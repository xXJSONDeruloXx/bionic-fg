// Bionic FG Vulkan implicit layer
// Intercepts vkCreateSwapchainKHR + vkQueuePresentKHR to inject generated frames.
// Image sharing between producer device and framegen device is done via AHardwareBuffer.
// Enabled by BIONIC_FG_ENABLE=1 and configured by conf.toml.

#include "../framegen_context.hpp"
#include "../logging.hpp"
#include "../vulkan/vk_types.hpp"

#include <vulkan/vulkan.h>
#ifdef __ANDROID__
#include <android/hardware_buffer.h>
#include <android/log.h>
#include <vulkan/vulkan_android.h>
#endif

#include <algorithm>
#include <chrono>
#include <condition_variable>
#include <cctype>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <deque>
#include <fstream>
#include <iomanip>
#include <iterator>
#include <memory>
#include <mutex>
#include <sstream>
#include <stdexcept>
#include <string>
#include <sys/stat.h>
#include <thread>
#include <unordered_map>
#include <vector>

#define BFG_LAYER_NAME "VK_LAYER_BIONIC_framegen"
#ifdef __ANDROID__
#define BFG_LAYER(...) __android_log_print(ANDROID_LOG_INFO, BFG_LAYER_NAME, __VA_ARGS__)
#define BFG_LAYER_E(...) __android_log_print(ANDROID_LOG_ERROR, BFG_LAYER_NAME, __VA_ARGS__)
#else
#define BFG_LAYER(...)
#define BFG_LAYER_E(...)
#endif

#if defined(__GNUC__)
#define BFG_EXPORT __attribute__((visibility("default"), used))
#else
#define BFG_EXPORT
#endif

namespace bfg::layer {

// Minimal loader-link structs normally provided by vulkan/vk_layer.h. The
// Android NDK only ships vulkan_core.h, so define the ABI-compatible subset we
// need for vkCreateInstance/vkCreateDevice trampoline chaining.
typedef enum VkLayerFunction_ {
    VK_LAYER_LINK_INFO = 0,
    VK_LOADER_DATA_CALLBACK = 1,
} VkLayerFunction;

struct VkLayerInstanceLink_ {
    VkLayerInstanceLink_* pNext;
    PFN_vkGetInstanceProcAddr pfnNextGetInstanceProcAddr;
    PFN_vkGetDeviceProcAddr   pfnNextGetDeviceProcAddr;
};
using VkLayerInstanceLink = VkLayerInstanceLink_;

struct VkLayerInstanceCreateInfo {
    VkStructureType sType;
    const void* pNext;
    VkLayerFunction function;
    union { VkLayerInstanceLink* pLayerInfo; void* pfnSetInstanceLoaderData; } u;
};
using VkLayerDeviceCreateInfo = VkLayerInstanceCreateInfo;

// ─── Dispatch key helper ──────────────────────────────────────────────────────

static void* dispatchKey(void* h) { return *reinterpret_cast<void**>(h); }

// ─── Config ───────────────────────────────────────────────────────────────────

struct LayerConf {
    bool        enabled    = false;
    int         multiplier = 2;
    float       flowScale  = 0.6f;
    int         model      = 0;
    bool        debugTiming = false;
    int         debugSummaryEvery = 60;
    bool        pacePresent = false;
    double      paceIntervalMs = 8.333;
    std::string configPath;
    int64_t     configStamp = 0;
};

static constexpr int kMaxHotReloadMultiplier = 4;
static constexpr int kMaxHotReloadGeneratedFrames = kMaxHotReloadMultiplier - 1;
static constexpr int kMaxHotReloadPresentOps = kMaxHotReloadMultiplier;
static constexpr uint32_t kHotReloadProvisionedSwapchainImages =
    static_cast<uint32_t>(kMaxHotReloadMultiplier + 1);

static int64_t steadyNowNs() {
    return std::chrono::duration_cast<std::chrono::nanoseconds>(
        std::chrono::steady_clock::now().time_since_epoch()).count();
}

static double nsToMs(int64_t ns) {
    return static_cast<double>(ns) / 1000000.0;
}

static std::string trim(std::string s) {
    auto notSpace = [](unsigned char ch) { return !std::isspace(ch); };
    s.erase(s.begin(), std::find_if(s.begin(), s.end(), notSpace));
    s.erase(std::find_if(s.rbegin(), s.rend(), notSpace).base(), s.end());
    return s;
}

static std::string stripComment(const std::string& line) {
    bool inString = false;
    for (size_t i = 0; i < line.size(); ++i) {
        char ch = line[i];
        if (ch == '"' && (i == 0 || line[i - 1] != '\\')) inString = !inString;
        if (ch == '#' && !inString) return line.substr(0, i);
    }
    return line;
}

static std::string unquote(std::string value) {
    value = trim(value);
    if (value.size() >= 2 && value.front() == '"' && value.back() == '"') {
        value = value.substr(1, value.size() - 2);
        std::string out;
        out.reserve(value.size());
        bool escaped = false;
        for (char ch : value) {
            if (escaped) {
                out.push_back(ch);
                escaped = false;
            } else if (ch == '\\') {
                escaped = true;
            } else {
                out.push_back(ch);
            }
        }
        return out;
    }
    return value;
}

static bool parseBool(std::string value, bool fallback) {
    value = unquote(value);
    std::transform(value.begin(), value.end(), value.begin(),
                   [](unsigned char ch) { return static_cast<char>(std::tolower(ch)); });
    if (value == "true" || value == "1" || value == "yes" || value == "on") return true;
    if (value == "false" || value == "0" || value == "no" || value == "off") return false;
    return fallback;
}

static int64_t fileStamp(const std::string& path) {
    struct stat st{};
    if (path.empty() || stat(path.c_str(), &st) != 0) return 0;
#if defined(__APPLE__)
    return static_cast<int64_t>(st.st_mtimespec.tv_sec) * 1000000000LL + st.st_mtimespec.tv_nsec;
#else
    return static_cast<int64_t>(st.st_mtim.tv_sec) * 1000000000LL + st.st_mtim.tv_nsec;
#endif
}

static std::string defaultConfigPath() {
    if (const char* explicitPath = std::getenv("BIONIC_FG_CONFIG")) {
        if (explicitPath[0] != '\0') return explicitPath;
    }
    if (const char* home = std::getenv("HOME")) {
        if (home[0] != '\0') return std::string(home) + "/.config/bionic-fg/conf.toml";
    }
    return {};
}

static void parseConfigFile(const std::string& path, LayerConf& c) {
    std::ifstream in(path);
    if (!in) return;

    bool sawEnabled = false;
    std::string line;
    while (std::getline(in, line)) {
        line = trim(stripComment(line));
        if (line.empty() || line.front() == '[') continue;

        size_t eq = line.find('=');
        if (eq == std::string::npos) continue;
        std::string key = trim(line.substr(0, eq));
        std::string value = trim(line.substr(eq + 1));
        std::transform(key.begin(), key.end(), key.begin(),
                       [](unsigned char ch) { return static_cast<char>(std::tolower(ch)); });

        try {
            if (key == "enabled") {
                c.enabled = parseBool(value, c.enabled);
                sawEnabled = true;
            } else if (key == "multiplier") {
                c.multiplier = std::max(0, std::min(4, std::atoi(unquote(value).c_str())));
            } else if (key == "flow_scale" || key == "flowscale") {
                c.flowScale = std::max(0.2f, std::min(1.0f, static_cast<float>(std::atof(unquote(value).c_str()))));
            } else if (key == "model") {
                c.model = std::max(0, std::min(1, std::atoi(unquote(value).c_str())));
            } else if (key == "debug_timing" || key == "debugtiming") {
                c.debugTiming = parseBool(value, c.debugTiming);
            } else if (key == "debug_summary_every" || key == "debugsummaryevery") {
                c.debugSummaryEvery = std::max(1, std::min(600, std::atoi(unquote(value).c_str())));
            } else if (key == "pace_present" || key == "pacepresent") {
                c.pacePresent = parseBool(value, c.pacePresent);
            } else if (key == "pace_interval_ms" || key == "paceintervalms") {
                c.paceIntervalMs = std::max(1.0, std::min(50.0, std::atof(unquote(value).c_str())));
            }
        } catch (...) {
            BFG_LAYER_E("Ignoring invalid config value: %s", line.c_str());
        }
    }

    // If a config exists but omits enabled, keep the launch activation state.
    (void)sawEnabled;
}

static LayerConf readConf() {
    LayerConf c;

    const char* en = std::getenv("BIONIC_FG_ENABLE");
    c.enabled = en && en[0] == '1';

    // Backwards-compatible env fallback. A config file, when present, wins.
    if (auto* v = std::getenv("BIONIC_FG_MULTIPLIER"))
        c.multiplier = std::max(0, std::min(4, std::atoi(v)));
    if (auto* v = std::getenv("BIONIC_FG_FLOW_SCALE"))
        c.flowScale = std::max(0.2f, std::min(1.0f, static_cast<float>(std::atof(v))));
    if (auto* v = std::getenv("BIONIC_FG_MODEL"))
        c.model = std::max(0, std::min(1, std::atoi(v)));
    if (auto* v = std::getenv("BIONIC_FG_DEBUG_TIMING"))
        c.debugTiming = parseBool(v, c.debugTiming);
    if (auto* v = std::getenv("BIONIC_FG_DEBUG_SUMMARY_EVERY"))
        c.debugSummaryEvery = std::max(1, std::min(600, std::atoi(v)));
    if (auto* v = std::getenv("BIONIC_FG_PACE_PRESENT"))
        c.pacePresent = parseBool(v, c.pacePresent);
    if (auto* v = std::getenv("BIONIC_FG_PACE_INTERVAL_MS"))
        c.paceIntervalMs = std::max(1.0, std::min(50.0, std::atof(v)));

    c.configPath = defaultConfigPath();
    if (!c.configPath.empty()) {
        c.configStamp = fileStamp(c.configPath);
        if (c.configStamp != 0) parseConfigFile(c.configPath, c);
    }
    return c;
}

static bool configNeedsSwapchainRecreate(const LayerConf& oldConf, const LayerConf& newConf) {
    return oldConf.enabled != newConf.enabled;
}

#ifdef __ANDROID__
struct DisableLayerEnvGuard {
    bool hadDisable = false;
    std::string oldDisable;

    DisableLayerEnvGuard() {
        if (const char* v = std::getenv("DISABLE_BIONIC_FG")) {
            hadDisable = true;
            oldDisable = v;
        }
        setenv("DISABLE_BIONIC_FG", "1", 1);
    }

    ~DisableLayerEnvGuard() {
        if (hadDisable) setenv("DISABLE_BIONIC_FG", oldDisable.c_str(), 1);
        else unsetenv("DISABLE_BIONIC_FG");
    }
};
#endif

// ─── Instance data ────────────────────────────────────────────────────────────

struct InstanceData {
    PFN_vkGetInstanceProcAddr  next = nullptr;
    PFN_vkDestroyInstance      destroyInstance = nullptr;
    PFN_vkEnumeratePhysicalDevices enumeratePhysicalDevices = nullptr;
};

// ─── Device data ──────────────────────────────────────────────────────────────

struct DeviceData {
    VkInstance       instance  = VK_NULL_HANDLE;
    VkPhysicalDevice physical  = VK_NULL_HANDLE;
    uint32_t         queueFamilyIdx = 0;
    PFN_vkGetDeviceProcAddr        next                   = nullptr;
    PFN_vkDestroyDevice            destroyDevice          = nullptr;
    PFN_vkCreateSwapchainKHR       createSwapchain        = nullptr;
    PFN_vkDestroySwapchainKHR      destroySwapchain       = nullptr;
    PFN_vkGetSwapchainImagesKHR    getSwapchainImages     = nullptr;
    PFN_vkQueuePresentKHR          queuePresent           = nullptr;
    PFN_vkAcquireNextImageKHR      acquireNextImage       = nullptr;
    PFN_vkAcquireNextImage2KHR     acquireNextImage2      = nullptr;
    PFN_vkGetDeviceQueue           getDeviceQueue         = nullptr;
    PFN_vkGetDeviceQueue2          getDeviceQueue2        = nullptr;
    PFN_vkQueueSubmit              queueSubmit            = nullptr;
    PFN_vkDeviceWaitIdle           deviceWaitIdle         = nullptr;
    PFN_vkCreateCommandPool        createCmdPool          = nullptr;
    PFN_vkDestroyCommandPool       destroyCmdPool         = nullptr;
    PFN_vkAllocateCommandBuffers   allocCmdBufs           = nullptr;
    PFN_vkResetCommandBuffer       resetCmdBuf            = nullptr;
    PFN_vkBeginCommandBuffer       beginCmdBuf            = nullptr;
    PFN_vkEndCommandBuffer         endCmdBuf              = nullptr;
    PFN_vkCmdPipelineBarrier       cmdPipelineBarrier     = nullptr;
    PFN_vkCmdBlitImage             cmdBlitImage           = nullptr;
    PFN_vkCreateImage              createImage            = nullptr;
    PFN_vkDestroyImage             destroyImage           = nullptr;
    PFN_vkAllocateMemory           allocMemory            = nullptr;
    PFN_vkFreeMemory               freeMemory             = nullptr;
    PFN_vkBindImageMemory          bindImageMemory        = nullptr;
    PFN_vkCreateImageView          createImageView        = nullptr;
    PFN_vkDestroyImageView         destroyImageView       = nullptr;
    PFN_vkCreateFence              createFence            = nullptr;
    PFN_vkDestroyFence             destroyFence           = nullptr;
    PFN_vkWaitForFences            waitForFences          = nullptr;
    PFN_vkResetFences              resetFences            = nullptr;
    PFN_vkCreateSemaphore          createSemaphore        = nullptr;
    PFN_vkDestroySemaphore         destroySemaphore       = nullptr;
    PFN_vkGetPhysicalDeviceMemoryProperties getPhysDevMemProps = nullptr;
#ifdef __ANDROID__
    PFN_vkGetAndroidHardwareBufferPropertiesANDROID getAhbProps = nullptr;
#endif
};

// ─── AHB-backed image on an external device ───────────────────────────────────

struct ExtImage {
    VkImage        img  = VK_NULL_HANDLE;
    VkDeviceMemory mem  = VK_NULL_HANDLE;
    VkImageView    view = VK_NULL_HANDLE;
    VkImageLayout  layout = VK_IMAGE_LAYOUT_UNDEFINED;
    bool           externalOwner = false;
};

#ifdef __ANDROID__
static ExtImage importAhb(const DeviceData& dd,
                          VkDevice device,
                          AHardwareBuffer* ahb,
                          VkExtent2D extent,
                          VkFormat format,
                          VkImageUsageFlags usage) {
    ExtImage out;
    VkAndroidHardwareBufferPropertiesANDROID ahbProps{};
    ahbProps.sType = VK_STRUCTURE_TYPE_ANDROID_HARDWARE_BUFFER_PROPERTIES_ANDROID;
    if (!dd.getAhbProps || dd.getAhbProps(device, ahb, &ahbProps) != VK_SUCCESS)
        return out;

    VkExternalMemoryImageCreateInfo emici{};
    emici.sType      = VK_STRUCTURE_TYPE_EXTERNAL_MEMORY_IMAGE_CREATE_INFO;
    emici.handleTypes= VK_EXTERNAL_MEMORY_HANDLE_TYPE_ANDROID_HARDWARE_BUFFER_BIT_ANDROID;

    VkImageCreateInfo ici{};
    ici.sType         = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO;
    ici.pNext         = &emici;
    ici.imageType     = VK_IMAGE_TYPE_2D;
    ici.format        = format;
    ici.extent        = {extent.width, extent.height, 1};
    ici.mipLevels     = 1;
    ici.arrayLayers   = 1;
    ici.samples       = VK_SAMPLE_COUNT_1_BIT;
    ici.tiling        = VK_IMAGE_TILING_OPTIMAL;
    ici.usage         = usage;
    ici.sharingMode   = VK_SHARING_MODE_EXCLUSIVE;
    ici.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;

    if (dd.createImage(device, &ici, nullptr, &out.img) != VK_SUCCESS) return out;

    VkMemoryDedicatedAllocateInfo dedicated{};
    dedicated.sType = VK_STRUCTURE_TYPE_MEMORY_DEDICATED_ALLOCATE_INFO;
    dedicated.image = out.img;
    VkImportAndroidHardwareBufferInfoANDROID importInfo{};
    importInfo.sType  = VK_STRUCTURE_TYPE_IMPORT_ANDROID_HARDWARE_BUFFER_INFO_ANDROID;
    importInfo.pNext  = &dedicated;
    importInfo.buffer = ahb;
    VkMemoryAllocateInfo mai{};
    mai.sType          = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
    mai.pNext          = &importInfo;
    mai.allocationSize = ahbProps.allocationSize;
    VkPhysicalDeviceMemoryProperties memProps{};
    dd.getPhysDevMemProps(dd.physical, &memProps);
    for (uint32_t i = 0; i < memProps.memoryTypeCount; ++i)
        if (ahbProps.memoryTypeBits & (1u << i)) { mai.memoryTypeIndex = i; break; }

    if (dd.allocMemory(device, &mai, nullptr, &out.mem) != VK_SUCCESS) {
        dd.destroyImage(device, out.img, nullptr);
        out.img = VK_NULL_HANDLE;
        return out;
    }
    dd.bindImageMemory(device, out.img, out.mem, 0);

    VkImageViewCreateInfo vci{};
    vci.sType                        = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
    vci.image                        = out.img;
    vci.viewType                     = VK_IMAGE_VIEW_TYPE_2D;
    vci.format                       = format;
    vci.components.r                 = VK_COMPONENT_SWIZZLE_IDENTITY;
    vci.components.g                 = VK_COMPONENT_SWIZZLE_IDENTITY;
    vci.components.b                 = VK_COMPONENT_SWIZZLE_IDENTITY;
    vci.components.a                 = VK_COMPONENT_SWIZZLE_IDENTITY;
    vci.subresourceRange.aspectMask  = VK_IMAGE_ASPECT_COLOR_BIT;
    vci.subresourceRange.levelCount  = 1;
    vci.subresourceRange.layerCount  = 1;
    dd.createImageView(device, &vci, nullptr, &out.view);
    return out;
}

static void freeExtImage(const DeviceData& dd, VkDevice device, ExtImage& img) {
    if (img.view) dd.destroyImageView(device, img.view, nullptr);
    if (img.mem)  dd.freeMemory(device, img.mem, nullptr);
    if (img.img)  dd.destroyImage(device, img.img, nullptr);
    img = {};
}
#endif

// ─── Swapchain state ──────────────────────────────────────────────────────────

struct SwapState {
    VkDevice   device = VK_NULL_HANDLE;
    VkExtent2D extent = {};
    VkFormat   format = VK_FORMAT_UNDEFINED;
    std::vector<VkImage> images;
    LayerConf  conf;

#ifdef __ANDROID__
    AHardwareBuffer* prevAhb = nullptr;
    AHardwareBuffer* currAhb = nullptr;
    std::vector<AHardwareBuffer*> outAhbs;

    ExtImage prevProducerImg, currProducerImg;
    std::vector<ExtImage> outProducerImgs;

    std::unique_ptr<bionic_fg::FramegenContext> fgCtx;

    // Copy cmd infrastructure (on producer device)
    VkCommandPool   copyPool   = VK_NULL_HANDLE;
    uint32_t        copyQueueFamily = VK_QUEUE_FAMILY_IGNORED;
    VkCommandBuffer copyCmds[2]= {};
    VkFence         copyFences[2]={};
    VkCommandBuffer genCmds[4] = {};    // one per possible presented output slot
    VkFence         genFences[4] = {};
    VkSemaphore     acquireSems[4] = {};
    VkSemaphore     presentSems[4] = {};

    uint64_t frameCount = 0;
    bool     inPresent  = false;

    uint64_t debugSourceSeq = 0;
    uint64_t debugOutputSeq = 0;
    uint32_t debugTraceFramesRemaining = 24;
    int64_t  debugLastSourceStartNs = 0;
    int64_t  debugLastOutputPresentNs = 0;
    double   debugOutputGapSumNs = 0.0;
    uint64_t debugOutputGapCount = 0;
    int64_t  debugOutputGapMinNs = 0;
    int64_t  debugOutputGapMaxNs = 0;
    uint64_t debugOutputGapUnder10ms = 0;
    uint64_t debugOutputGapUnder20ms = 0;
    uint64_t debugOutputGapOver20ms = 0;
    double   debugAcquireWaitSumNs = 0.0;
    uint64_t debugAcquireWaitCount = 0;
    int64_t  debugAcquireWaitMaxNs = 0;
    double   debugFramegenSumNs = 0.0;
    uint64_t debugFramegenCount = 0;
    int64_t  debugFramegenMaxNs = 0;
    double   debugCallbackSumNs = 0.0;
    uint64_t debugCallbackCount = 0;
    int64_t  debugCallbackMaxNs = 0;

    void cleanup(const DeviceData& dd, VkDevice dev) {
        if (dd.deviceWaitIdle) dd.deviceWaitIdle(dev);
        for (auto& s : acquireSems) if (s) { dd.destroySemaphore(dev, s, nullptr); s = VK_NULL_HANDLE; }
        for (auto& s : presentSems) if (s) { dd.destroySemaphore(dev, s, nullptr); s = VK_NULL_HANDLE; }
        for (auto& f : copyFences) if (f) { dd.destroyFence(dev, f, nullptr); f = VK_NULL_HANDLE; }
        for (auto& f : genFences) if (f) { dd.destroyFence(dev, f, nullptr); f = VK_NULL_HANDLE; }
        if (copyPool) dd.destroyCmdPool(dev, copyPool, nullptr);
        // Free AHB images on producer device
        freeExtImage(dd, dev, prevProducerImg);
        freeExtImage(dd, dev, currProducerImg);
        for (auto& img : outProducerImgs) freeExtImage(dd, dev, img);
        // Destroy framegen context (frees its own device)
        if (fgCtx) { fgCtx->destroy(); fgCtx.reset(); }
        // Release AHBs
        if (prevAhb) { AHardwareBuffer_release(prevAhb); prevAhb = nullptr; }
        if (currAhb) { AHardwareBuffer_release(currAhb); currAhb = nullptr; }
        for (auto* ahb : outAhbs) if (ahb) AHardwareBuffer_release(ahb);
        outAhbs.clear(); outProducerImgs.clear();
        copyPool = VK_NULL_HANDLE;
        copyQueueFamily = VK_QUEUE_FAMILY_IGNORED;
    }
#endif
};

static bionic_fg::Config makeFramegenConfig(const SwapState& st, const LayerConf& conf) {
    bionic_fg::Config cfg;
    cfg.width = st.extent.width;
    cfg.height = st.extent.height;
    cfg.multiplier = static_cast<uint32_t>(std::max(2, std::min(kMaxHotReloadMultiplier, conf.multiplier)));
    cfg.flowScale = conf.flowScale;
    cfg.model = static_cast<uint32_t>(std::max(0, std::min(1, conf.model)));
    return cfg;
}

#ifdef __ANDROID__
static std::unique_ptr<bionic_fg::FramegenContext> createFramegenContext(
        const SwapState& st,
        const LayerConf& conf) {
    DisableLayerEnvGuard disableLayerForInternalVulkan;
    const int64_t t0 = steadyNowNs();
    auto ctx = bionic_fg::FramegenContext::create(
        st.prevAhb,
        st.currAhb,
        st.outAhbs,
        st.extent,
        st.format,
        makeFramegenConfig(st, conf));
    BFG_LAYER("FramegenContext create mult=%d flow=%.2f model=%d result=%s took=%.3fms",
              conf.multiplier, conf.flowScale, conf.model,
              ctx ? "ok" : "null", nsToMs(steadyNowNs() - t0));
    return ctx;
}

static void resetFrameHistory(SwapState& st) {
    st.frameCount = 0;
    st.debugTraceFramesRemaining = 24;
    st.debugLastSourceStartNs = 0;
}

static bool rebuildFramegenContext(SwapState& st, const LayerConf& conf) {
    if (!st.prevAhb || !st.currAhb || st.outAhbs.empty()) {
        BFG_LAYER_E("cannot rebuild FramegenContext: AHBs not provisioned");
        return false;
    }

    if (st.fgCtx) st.fgCtx->waitIdle();

    auto newCtx = createFramegenContext(st, conf);
    if (!newCtx) {
        BFG_LAYER_E("FramegenContext rebuild failed for mult=%d flow=%.2f model=%d",
                    conf.multiplier, conf.flowScale, conf.model);
        return false;
    }

    if (st.fgCtx) {
        st.fgCtx->destroy();
    }
    st.fgCtx = std::move(newCtx);
    resetFrameHistory(st);
    BFG_LAYER("FramegenContext rebuilt: mult=%d flow=%.2f model=%d",
              conf.multiplier, conf.flowScale, conf.model);
    return true;
}
#endif

// ─── Global state ─────────────────────────────────────────────────────────────

static std::mutex g_mtx;
static std::mutex g_queueHostMtx;
static std::unordered_map<void*, InstanceData> g_instances;
static std::unordered_map<void*, DeviceData>   g_devices;
static std::unordered_map<VkPhysicalDevice, VkInstance> g_physInstances;
struct QueueData { void* deviceKey = nullptr; uint32_t family = VK_QUEUE_FAMILY_IGNORED; };
static std::unordered_map<VkQueue, QueueData> g_queues;
static std::unordered_map<VkSwapchainKHR, SwapState> g_swapchains;
static std::unordered_map<VkSwapchainKHR, void*>     g_swapDevice;

static VkResult queueSubmitLocked(const DeviceData& dd, VkQueue queue,
                                  uint32_t submitCount, const VkSubmitInfo* submits,
                                  VkFence fence) {
    std::lock_guard<std::mutex> lk(g_queueHostMtx);
    return dd.queueSubmit(queue, submitCount, submits, fence);
}

static VkResult queuePresentLocked(PFN_vkQueuePresentKHR present, VkQueue queue,
                                   const VkPresentInfoKHR* presentInfo) {
    std::lock_guard<std::mutex> lk(g_queueHostMtx);
    return present(queue, presentInfo);
}

static VkResult queuePresentLocked(const DeviceData& dd, VkQueue queue,
                                   const VkPresentInfoKHR* presentInfo) {
    return queuePresentLocked(dd.queuePresent, queue, presentInfo);
}

#ifdef __ANDROID__
struct ScheduledPresent {
    int64_t targetNs = 0;
    int64_t intervalNs = 0;
    PFN_vkQueuePresentKHR present = nullptr;
    VkQueue queue = VK_NULL_HANDLE;
    VkSwapchainKHR swapchain = VK_NULL_HANDLE;
    uint32_t imageIndex = UINT32_MAX;
    uint64_t sourceSeq = 0;
    int slot = 0;
    char label[16] = {};
};

static std::mutex g_presentSchedulerMtx;
static std::condition_variable g_presentSchedulerCv;
static std::deque<ScheduledPresent> g_presentSchedulerQueue;
static std::thread g_presentSchedulerThread;
static bool g_presentSchedulerStarted = false;
static int64_t g_presentSchedulerLastPresentNs = 0;
static uint64_t g_presentSchedulerSeq = 0;

static void presentSchedulerLoop() {
    for (;;) {
        ScheduledPresent item;
        {
            std::unique_lock<std::mutex> lk(g_presentSchedulerMtx);
            g_presentSchedulerCv.wait(lk, [] { return !g_presentSchedulerQueue.empty(); });
            for (;;) {
                auto best = g_presentSchedulerQueue.begin();
                for (auto it = g_presentSchedulerQueue.begin(); it != g_presentSchedulerQueue.end(); ++it)
                    if (it->targetNs < best->targetNs) best = it;
                int64_t effectiveTargetNs = best->targetNs;
                if (g_presentSchedulerLastPresentNs != 0 && best->intervalNs > 0)
                    effectiveTargetNs = std::max(effectiveTargetNs, g_presentSchedulerLastPresentNs + best->intervalNs);
                const int64_t nowNs = steadyNowNs();
                if (effectiveTargetNs > nowNs) {
                    g_presentSchedulerCv.wait_for(lk, std::chrono::nanoseconds(effectiveTargetNs - nowNs));
                    continue;
                }
                item = *best;
                g_presentSchedulerQueue.erase(best);
                break;
            }
        }

        int64_t effectiveTargetNs = item.targetNs;
        {
            std::lock_guard<std::mutex> lk(g_presentSchedulerMtx);
            if (g_presentSchedulerLastPresentNs != 0 && item.intervalNs > 0)
                effectiveTargetNs = std::max(effectiveTargetNs, g_presentSchedulerLastPresentNs + item.intervalNs);
        }
        const int64_t nowBeforePresentNs = steadyNowNs();
        if (effectiveTargetNs > nowBeforePresentNs)
            std::this_thread::sleep_for(std::chrono::nanoseconds(effectiveTargetNs - nowBeforePresentNs));

        VkPresentInfoKHR presentInfo{};
        presentInfo.sType = VK_STRUCTURE_TYPE_PRESENT_INFO_KHR;
        presentInfo.swapchainCount = 1;
        presentInfo.pSwapchains = &item.swapchain;
        presentInfo.pImageIndices = &item.imageIndex;

        const int64_t presentStartNs = steadyNowNs();
        VkResult res = item.present
            ? queuePresentLocked(item.present, item.queue, &presentInfo)
            : VK_ERROR_INITIALIZATION_FAILED;
        const int64_t presentEndNs = steadyNowNs();
        double gapMs = -1.0;
        {
            std::lock_guard<std::mutex> lk(g_presentSchedulerMtx);
            if (g_presentSchedulerLastPresentNs != 0)
                gapMs = nsToMs(presentEndNs - g_presentSchedulerLastPresentNs);
            g_presentSchedulerLastPresentNs = presentEndNs;
        }
        const uint64_t seq = ++g_presentSchedulerSeq;
        BFG_LAYER("paced present#%llu src#%llu %s slot=%d img=%u targetErr=%.3fms origErr=%.3fms present=%.3fms gapPrev=%.3fms res=%d",
                  static_cast<unsigned long long>(seq),
                  static_cast<unsigned long long>(item.sourceSeq),
                  item.label,
                  item.slot,
                  item.imageIndex,
                  nsToMs(presentStartNs - effectiveTargetNs),
                  nsToMs(presentStartNs - item.targetNs),
                  nsToMs(presentEndNs - presentStartNs),
                  gapMs,
                  static_cast<int>(res));
    }
}

static void ensurePresentSchedulerStarted() {
    std::lock_guard<std::mutex> lk(g_presentSchedulerMtx);
    if (g_presentSchedulerStarted) return;
    g_presentSchedulerStarted = true;
    g_presentSchedulerThread = std::thread(presentSchedulerLoop);
    g_presentSchedulerThread.detach();
}

static void schedulePresent(const ScheduledPresent& item) {
    ensurePresentSchedulerStarted();
    {
        std::lock_guard<std::mutex> lk(g_presentSchedulerMtx);
        g_presentSchedulerQueue.push_back(item);
    }
    g_presentSchedulerCv.notify_one();
}

static void cancelScheduledPresents(VkSwapchainKHR swapchain) {
    std::lock_guard<std::mutex> lk(g_presentSchedulerMtx);
    for (auto it = g_presentSchedulerQueue.begin(); it != g_presentSchedulerQueue.end(); ) {
        if (it->swapchain == swapchain) it = g_presentSchedulerQueue.erase(it);
        else ++it;
    }
}

static void debugRecordAcquireWait(SwapState& st, int64_t waitNs) {
    if (!st.conf.debugTiming) return;
    st.debugAcquireWaitSumNs += static_cast<double>(waitNs);
    st.debugAcquireWaitCount++;
    st.debugAcquireWaitMaxNs = std::max(st.debugAcquireWaitMaxNs, waitNs);
}

static void debugRecordFramegenTime(SwapState& st, int64_t fgNs) {
    if (!st.conf.debugTiming) return;
    st.debugFramegenSumNs += static_cast<double>(fgNs);
    st.debugFramegenCount++;
    st.debugFramegenMaxNs = std::max(st.debugFramegenMaxNs, fgNs);
}

static void debugRecordCallbackTime(SwapState& st, int64_t callbackNs) {
    if (!st.conf.debugTiming) return;
    st.debugCallbackSumNs += static_cast<double>(callbackNs);
    st.debugCallbackCount++;
    st.debugCallbackMaxNs = std::max(st.debugCallbackMaxNs, callbackNs);
}

static double debugRecordOutputPresent(SwapState& st, int64_t presentNs) {
    if (!st.conf.debugTiming) return -1.0;

    double gapMs = -1.0;
    if (st.debugLastOutputPresentNs != 0) {
        int64_t gapNs = presentNs - st.debugLastOutputPresentNs;
        gapMs = nsToMs(gapNs);
        st.debugOutputGapSumNs += static_cast<double>(gapNs);
        st.debugOutputGapCount++;
        st.debugOutputGapMinNs = (st.debugOutputGapMinNs == 0)
            ? gapNs
            : std::min(st.debugOutputGapMinNs, gapNs);
        st.debugOutputGapMaxNs = std::max(st.debugOutputGapMaxNs, gapNs);
        if (gapMs < 10.0) st.debugOutputGapUnder10ms++;
        else if (gapMs < 20.0) st.debugOutputGapUnder20ms++;
        else st.debugOutputGapOver20ms++;
    }
    st.debugLastOutputPresentNs = presentNs;
    st.debugOutputSeq++;

    if (st.conf.debugSummaryEvery > 0 && st.debugOutputSeq > 0 &&
        (st.debugOutputSeq % static_cast<uint64_t>(st.conf.debugSummaryEvery) == 0)) {
        const double avgGapMs = st.debugOutputGapCount > 0
            ? nsToMs(static_cast<int64_t>(st.debugOutputGapSumNs / static_cast<double>(st.debugOutputGapCount)))
            : 0.0;
        const double avgAcquireMs = st.debugAcquireWaitCount > 0
            ? nsToMs(static_cast<int64_t>(st.debugAcquireWaitSumNs / static_cast<double>(st.debugAcquireWaitCount)))
            : 0.0;
        const double avgFgMs = st.debugFramegenCount > 0
            ? nsToMs(static_cast<int64_t>(st.debugFramegenSumNs / static_cast<double>(st.debugFramegenCount)))
            : 0.0;
        const double avgCallbackMs = st.debugCallbackCount > 0
            ? nsToMs(static_cast<int64_t>(st.debugCallbackSumNs / static_cast<double>(st.debugCallbackCount)))
            : 0.0;
        const uint64_t totalGapBuckets = st.debugOutputGapUnder10ms + st.debugOutputGapUnder20ms + st.debugOutputGapOver20ms;
        const double under10Pct = totalGapBuckets > 0 ? (100.0 * static_cast<double>(st.debugOutputGapUnder10ms) / static_cast<double>(totalGapBuckets)) : 0.0;
        const double under20Pct = totalGapBuckets > 0 ? (100.0 * static_cast<double>(st.debugOutputGapUnder20ms) / static_cast<double>(totalGapBuckets)) : 0.0;
        const double over20Pct = totalGapBuckets > 0 ? (100.0 * static_cast<double>(st.debugOutputGapOver20ms) / static_cast<double>(totalGapBuckets)) : 0.0;
        BFG_LAYER(
            "timing summary outputs=%llu gaps<10=%.1f%% gaps10to20=%.1f%% gaps>=20=%.1f%% gapAvg=%.3fms gapMin=%.3fms gapMax=%.3fms acquireAvg=%.3fms acquireMax=%.3fms fgAvg=%.3fms fgMax=%.3fms cbAvg=%.3fms cbMax=%.3fms",
            static_cast<unsigned long long>(st.debugOutputSeq),
            under10Pct,
            under20Pct,
            over20Pct,
            avgGapMs,
            nsToMs(st.debugOutputGapMinNs),
            nsToMs(st.debugOutputGapMaxNs),
            avgAcquireMs,
            nsToMs(st.debugAcquireWaitMaxNs),
            avgFgMs,
            nsToMs(st.debugFramegenMaxNs),
            avgCallbackMs,
            nsToMs(st.debugCallbackMaxNs));
    }

    return gapMs;
}
#endif

// ─── Barrier helper ───────────────────────────────────────────────────────────

static void layerImageBarrier(const DeviceData& dd, VkCommandBuffer cmd,
        VkImage img,
        VkPipelineStageFlags srcStage, VkAccessFlags srcAccess,
        VkPipelineStageFlags dstStage, VkAccessFlags dstAccess,
        VkImageLayout oldLayout, VkImageLayout newLayout,
        uint32_t srcQueueFamily = VK_QUEUE_FAMILY_IGNORED,
        uint32_t dstQueueFamily = VK_QUEUE_FAMILY_IGNORED) {
    VkImageMemoryBarrier b{};
    b.sType               = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
    b.srcAccessMask       = srcAccess;
    b.dstAccessMask       = dstAccess;
    b.oldLayout           = oldLayout;
    b.newLayout           = newLayout;
    b.srcQueueFamilyIndex = srcQueueFamily;
    b.dstQueueFamilyIndex = dstQueueFamily;
    b.image               = img;
    b.subresourceRange    = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};
    dd.cmdPipelineBarrier(cmd, srcStage, dstStage, 0, 0, nullptr, 0, nullptr, 1, &b);
}

// ─── CreateInstance ───────────────────────────────────────────────────────────

VKAPI_ATTR VkResult VKAPI_CALL BionicFG_CreateInstance(
        const VkInstanceCreateInfo* pCreateInfo,
        const VkAllocationCallbacks* pAllocator,
        VkInstance* pInstance) {
    // Walk pNext for the layer link info
    auto* linkInfo = reinterpret_cast<const VkLayerInstanceCreateInfo*>(pCreateInfo->pNext);
    while (linkInfo && !(linkInfo->sType == VK_STRUCTURE_TYPE_LOADER_INSTANCE_CREATE_INFO
                         && linkInfo->function == VK_LAYER_LINK_INFO)) {
        linkInfo = reinterpret_cast<const VkLayerInstanceCreateInfo*>(linkInfo->pNext);
    }
    if (!linkInfo) return VK_ERROR_INITIALIZATION_FAILED;

    PFN_vkGetInstanceProcAddr nextGIPA = linkInfo->u.pLayerInfo->pfnNextGetInstanceProcAddr;
    // advance layer link
    const_cast<VkLayerInstanceCreateInfo*>(linkInfo)->u.pLayerInfo =
        linkInfo->u.pLayerInfo->pNext;

    auto* createInstance = reinterpret_cast<PFN_vkCreateInstance>(
        nextGIPA(VK_NULL_HANDLE, "vkCreateInstance"));
    if (!createInstance) return VK_ERROR_INITIALIZATION_FAILED;

    VkResult res = createInstance(pCreateInfo, pAllocator, pInstance);
    if (res != VK_SUCCESS) return res;

    InstanceData data;
    data.next            = nextGIPA;
    data.destroyInstance = reinterpret_cast<PFN_vkDestroyInstance>(
        nextGIPA(*pInstance, "vkDestroyInstance"));
    data.enumeratePhysicalDevices = reinterpret_cast<PFN_vkEnumeratePhysicalDevices>(
        nextGIPA(*pInstance, "vkEnumeratePhysicalDevices"));
    {
        std::lock_guard<std::mutex> lk(g_mtx);
        g_instances[dispatchKey(*pInstance)] = data;
    }
    return VK_SUCCESS;
}

VKAPI_ATTR void VKAPI_CALL BionicFG_DestroyInstance(
        VkInstance instance, const VkAllocationCallbacks* pAllocator) {
    std::lock_guard<std::mutex> lk(g_mtx);
    auto it = g_instances.find(dispatchKey(instance));
    if (it != g_instances.end()) {
        for (auto pit = g_physInstances.begin(); pit != g_physInstances.end(); ) {
            pit = (pit->second == instance) ? g_physInstances.erase(pit) : std::next(pit);
        }
        it->second.destroyInstance(instance, pAllocator);
        g_instances.erase(it);
    }
}

VKAPI_ATTR VkResult VKAPI_CALL BionicFG_EnumeratePhysicalDevices(
        VkInstance instance, uint32_t* pPhysicalDeviceCount,
        VkPhysicalDevice* pPhysicalDevices) {
    InstanceData id;
    {
        std::lock_guard<std::mutex> lk(g_mtx);
        auto it = g_instances.find(dispatchKey(instance));
        if (it == g_instances.end() || !it->second.enumeratePhysicalDevices)
            return VK_ERROR_INITIALIZATION_FAILED;
        id = it->second;
    }
    VkResult res = id.enumeratePhysicalDevices(instance, pPhysicalDeviceCount, pPhysicalDevices);
    if ((res == VK_SUCCESS || res == VK_INCOMPLETE) && pPhysicalDevices && pPhysicalDeviceCount) {
        std::lock_guard<std::mutex> lk(g_mtx);
        for (uint32_t i = 0; i < *pPhysicalDeviceCount; ++i)
            g_physInstances[pPhysicalDevices[i]] = instance;
    }
    return res;
}

// ─── CreateDevice ─────────────────────────────────────────────────────────────

static const char* kAhbExts[] = {
    "VK_ANDROID_external_memory_android_hardware_buffer",
    "VK_KHR_external_memory",
    "VK_KHR_sampler_ycbcr_conversion",
    "VK_KHR_dedicated_allocation",
    "VK_KHR_get_memory_requirements2",
    "VK_KHR_bind_memory2",
    "VK_KHR_maintenance1",
};

VKAPI_ATTR VkResult VKAPI_CALL BionicFG_CreateDevice(
        VkPhysicalDevice physicalDevice,
        const VkDeviceCreateInfo* pCreateInfo,
        const VkAllocationCallbacks* pAllocator,
        VkDevice* pDevice) {
    auto* linkInfo = reinterpret_cast<const VkLayerDeviceCreateInfo*>(pCreateInfo->pNext);
    while (linkInfo && !(linkInfo->sType == VK_STRUCTURE_TYPE_LOADER_DEVICE_CREATE_INFO
                         && linkInfo->function == VK_LAYER_LINK_INFO)) {
        linkInfo = reinterpret_cast<const VkLayerDeviceCreateInfo*>(linkInfo->pNext);
    }
    if (!linkInfo) return VK_ERROR_INITIALIZATION_FAILED;

    PFN_vkGetInstanceProcAddr nextGIPA = linkInfo->u.pLayerInfo->pfnNextGetInstanceProcAddr;
    PFN_vkGetDeviceProcAddr   nextGDPA = linkInfo->u.pLayerInfo->pfnNextGetDeviceProcAddr;
    const_cast<VkLayerDeviceCreateInfo*>(linkInfo)->u.pLayerInfo =
        linkInfo->u.pLayerInfo->pNext;

    // Inject AHB extensions if not present (needed for cross-device sharing)
    std::vector<const char*> exts(
        pCreateInfo->ppEnabledExtensionNames,
        pCreateInfo->ppEnabledExtensionNames + pCreateInfo->enabledExtensionCount);
    for (const char* e : kAhbExts) {
        bool found = false;
        for (const char* x : exts) if (std::strcmp(x, e) == 0) { found = true; break; }
        if (!found) exts.push_back(e);
    }
    VkDeviceCreateInfo dci = *pCreateInfo;
    dci.enabledExtensionCount   = static_cast<uint32_t>(exts.size());
    dci.ppEnabledExtensionNames = exts.data();

    // Look up the real instance that produced this physical device.
    VkInstance instance = VK_NULL_HANDLE;
    {
        std::lock_guard<std::mutex> lk(g_mtx);
        auto pit = g_physInstances.find(physicalDevice);
        if (pit != g_physInstances.end()) instance = pit->second;
    }

    auto* createDevice = reinterpret_cast<PFN_vkCreateDevice>(
        nextGIPA(instance, "vkCreateDevice"));
    if (!createDevice) return VK_ERROR_INITIALIZATION_FAILED;

    VkResult res = createDevice(physicalDevice, &dci, pAllocator, pDevice);
    if (res != VK_SUCCESS) return res;

    // Find compute/graphics queue family through the next instance dispatch.
    // Calling the global loader symbol from inside a layer can reject wrapped
    // physical-device handles as invalid.
    auto* getQueueFamilyProps = reinterpret_cast<PFN_vkGetPhysicalDeviceQueueFamilyProperties>(
        nextGIPA(instance, "vkGetPhysicalDeviceQueueFamilyProperties"));
    if (!getQueueFamilyProps) return VK_ERROR_INITIALIZATION_FAILED;
    uint32_t qfCount = 0;
    getQueueFamilyProps(physicalDevice, &qfCount, nullptr);
    std::vector<VkQueueFamilyProperties> qfProps(qfCount);
    getQueueFamilyProps(physicalDevice, &qfCount, qfProps.data());
    uint32_t qf = 0;
    for (uint32_t i = 0; i < qfCount; ++i) {
        if (qfProps[i].queueFlags & (VK_QUEUE_GRAPHICS_BIT | VK_QUEUE_COMPUTE_BIT)) {
            qf = i; break;
        }
    }

    auto load = [&](const char* name) {
        return reinterpret_cast<PFN_vkVoidFunction>(nextGDPA(*pDevice, name));
    };

    DeviceData dd;
    dd.instance       = instance;
    dd.physical       = physicalDevice;
    dd.queueFamilyIdx = qf;
    dd.next                 = nextGDPA;
    dd.destroyDevice        = reinterpret_cast<PFN_vkDestroyDevice>(load("vkDestroyDevice"));
    dd.createSwapchain      = reinterpret_cast<PFN_vkCreateSwapchainKHR>(load("vkCreateSwapchainKHR"));
    dd.destroySwapchain     = reinterpret_cast<PFN_vkDestroySwapchainKHR>(load("vkDestroySwapchainKHR"));
    dd.getSwapchainImages   = reinterpret_cast<PFN_vkGetSwapchainImagesKHR>(load("vkGetSwapchainImagesKHR"));
    dd.queuePresent         = reinterpret_cast<PFN_vkQueuePresentKHR>(load("vkQueuePresentKHR"));
    dd.acquireNextImage     = reinterpret_cast<PFN_vkAcquireNextImageKHR>(load("vkAcquireNextImageKHR"));
    dd.acquireNextImage2    = reinterpret_cast<PFN_vkAcquireNextImage2KHR>(load("vkAcquireNextImage2KHR"));
    dd.getDeviceQueue       = reinterpret_cast<PFN_vkGetDeviceQueue>(load("vkGetDeviceQueue"));
    dd.getDeviceQueue2      = reinterpret_cast<PFN_vkGetDeviceQueue2>(load("vkGetDeviceQueue2"));
    dd.queueSubmit          = reinterpret_cast<PFN_vkQueueSubmit>(load("vkQueueSubmit"));
    dd.deviceWaitIdle       = reinterpret_cast<PFN_vkDeviceWaitIdle>(load("vkDeviceWaitIdle"));
    dd.createCmdPool        = reinterpret_cast<PFN_vkCreateCommandPool>(load("vkCreateCommandPool"));
    dd.destroyCmdPool       = reinterpret_cast<PFN_vkDestroyCommandPool>(load("vkDestroyCommandPool"));
    dd.allocCmdBufs         = reinterpret_cast<PFN_vkAllocateCommandBuffers>(load("vkAllocateCommandBuffers"));
    dd.resetCmdBuf          = reinterpret_cast<PFN_vkResetCommandBuffer>(load("vkResetCommandBuffer"));
    dd.beginCmdBuf          = reinterpret_cast<PFN_vkBeginCommandBuffer>(load("vkBeginCommandBuffer"));
    dd.endCmdBuf            = reinterpret_cast<PFN_vkEndCommandBuffer>(load("vkEndCommandBuffer"));
    dd.cmdPipelineBarrier   = reinterpret_cast<PFN_vkCmdPipelineBarrier>(load("vkCmdPipelineBarrier"));
    dd.cmdBlitImage         = reinterpret_cast<PFN_vkCmdBlitImage>(load("vkCmdBlitImage"));
    dd.createImage          = reinterpret_cast<PFN_vkCreateImage>(load("vkCreateImage"));
    dd.destroyImage         = reinterpret_cast<PFN_vkDestroyImage>(load("vkDestroyImage"));
    dd.allocMemory          = reinterpret_cast<PFN_vkAllocateMemory>(load("vkAllocateMemory"));
    dd.freeMemory           = reinterpret_cast<PFN_vkFreeMemory>(load("vkFreeMemory"));
    dd.bindImageMemory      = reinterpret_cast<PFN_vkBindImageMemory>(load("vkBindImageMemory"));
    dd.createImageView      = reinterpret_cast<PFN_vkCreateImageView>(load("vkCreateImageView"));
    dd.destroyImageView     = reinterpret_cast<PFN_vkDestroyImageView>(load("vkDestroyImageView"));
    dd.createFence          = reinterpret_cast<PFN_vkCreateFence>(load("vkCreateFence"));
    dd.destroyFence         = reinterpret_cast<PFN_vkDestroyFence>(load("vkDestroyFence"));
    dd.waitForFences        = reinterpret_cast<PFN_vkWaitForFences>(load("vkWaitForFences"));
    dd.resetFences          = reinterpret_cast<PFN_vkResetFences>(load("vkResetFences"));
    dd.createSemaphore      = reinterpret_cast<PFN_vkCreateSemaphore>(load("vkCreateSemaphore"));
    dd.destroySemaphore     = reinterpret_cast<PFN_vkDestroySemaphore>(load("vkDestroySemaphore"));
    dd.getPhysDevMemProps   = reinterpret_cast<PFN_vkGetPhysicalDeviceMemoryProperties>(
        nextGIPA(instance, "vkGetPhysicalDeviceMemoryProperties"));
#ifdef __ANDROID__
    dd.getAhbProps = reinterpret_cast<PFN_vkGetAndroidHardwareBufferPropertiesANDROID>(
        load("vkGetAndroidHardwareBufferPropertiesANDROID"));
#endif
    {
        std::lock_guard<std::mutex> lk(g_mtx);
        g_devices[dispatchKey(*pDevice)] = dd;
    }
    BFG_LAYER("Device created, queueFamily=%u", qf);
    return VK_SUCCESS;
}

VKAPI_ATTR void VKAPI_CALL BionicFG_DestroyDevice(
        VkDevice device, const VkAllocationCallbacks* pAllocator) {
    std::lock_guard<std::mutex> lk(g_mtx);
    void* devKey = dispatchKey(device);
    for (auto itq = g_queues.begin(); itq != g_queues.end(); ) {
        itq = (itq->second.deviceKey == devKey) ? g_queues.erase(itq) : std::next(itq);
    }
    auto it = g_devices.find(devKey);
    if (it != g_devices.end()) {
        it->second.destroyDevice(device, pAllocator);
        g_devices.erase(it);
    }
}

VKAPI_ATTR void VKAPI_CALL BionicFG_GetDeviceQueue(
        VkDevice device, uint32_t queueFamilyIndex, uint32_t queueIndex,
        VkQueue* pQueue) {
    DeviceData dd;
    void* devKey = dispatchKey(device);
    {
        std::lock_guard<std::mutex> lk(g_mtx);
        auto it = g_devices.find(devKey);
        if (it == g_devices.end() || !it->second.getDeviceQueue) return;
        dd = it->second;
    }
    dd.getDeviceQueue(device, queueFamilyIndex, queueIndex, pQueue);
    if (pQueue && *pQueue) {
        std::lock_guard<std::mutex> lk(g_mtx);
        g_queues[*pQueue] = QueueData{devKey, queueFamilyIndex};
    }
}

VKAPI_ATTR void VKAPI_CALL BionicFG_GetDeviceQueue2(
        VkDevice device, const VkDeviceQueueInfo2* pQueueInfo, VkQueue* pQueue) {
    DeviceData dd;
    void* devKey = dispatchKey(device);
    {
        std::lock_guard<std::mutex> lk(g_mtx);
        auto it = g_devices.find(devKey);
        if (it == g_devices.end() || !it->second.getDeviceQueue2) return;
        dd = it->second;
    }
    dd.getDeviceQueue2(device, pQueueInfo, pQueue);
    if (pQueueInfo && pQueue && *pQueue) {
        std::lock_guard<std::mutex> lk(g_mtx);
        g_queues[*pQueue] = QueueData{devKey, pQueueInfo->queueFamilyIndex};
    }
}

VKAPI_ATTR VkResult VKAPI_CALL BionicFG_QueueSubmit(
        VkQueue queue,
        uint32_t submitCount,
        const VkSubmitInfo* pSubmits,
        VkFence fence) {
    DeviceData dd;
    {
        std::lock_guard<std::mutex> lk(g_mtx);
        auto qit = g_queues.find(queue);
        if (qit == g_queues.end() || !qit->second.deviceKey)
            return vkQueueSubmit(queue, submitCount, pSubmits, fence);
        auto it = g_devices.find(qit->second.deviceKey);
        if (it == g_devices.end() || !it->second.queueSubmit)
            return vkQueueSubmit(queue, submitCount, pSubmits, fence);
        dd = it->second;
    }
    return queueSubmitLocked(dd, queue, submitCount, pSubmits, fence);
}

VKAPI_ATTR VkResult VKAPI_CALL BionicFG_AcquireNextImage2KHR(
        VkDevice device, const VkAcquireNextImageInfoKHR* pAcquireInfo,
        uint32_t* pImageIndex) {
    DeviceData dd;
    {
        std::lock_guard<std::mutex> lk(g_mtx);
        auto it = g_devices.find(dispatchKey(device));
        if (it == g_devices.end() || !it->second.acquireNextImage2)
            return VK_ERROR_EXTENSION_NOT_PRESENT;
        dd = it->second;
    }
    return dd.acquireNextImage2(device, pAcquireInfo, pImageIndex);
}

// ─── CreateSwapchainKHR ───────────────────────────────────────────────────────

VKAPI_ATTR VkResult VKAPI_CALL BionicFG_CreateSwapchainKHR(
        VkDevice device,
        const VkSwapchainCreateInfoKHR* pCreateInfo,
        const VkAllocationCallbacks* pAllocator,
        VkSwapchainKHR* pSwapchain) {
    LayerConf conf = readConf();

    DeviceData dd;
    {
        std::lock_guard<std::mutex> lk(g_mtx);
        auto it = g_devices.find(dispatchKey(device));
        if (it == g_devices.end()) return VK_ERROR_INITIALIZATION_FAILED;
        dd = it->second;
    }

    // Modify swapchain: add transfer usage, request enough images up front for
    // full 2x..4x hot-reload without requiring a swapchain recreation later.
    VkSwapchainCreateInfoKHR ci = *pCreateInfo;
    ci.imageUsage |= VK_IMAGE_USAGE_TRANSFER_SRC_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT;
    if (conf.enabled)
        ci.minImageCount = std::max(ci.minImageCount, kHotReloadProvisionedSwapchainImages);

    // Retire old swapchain state if recreating
    if (pCreateInfo->oldSwapchain) {
        std::lock_guard<std::mutex> lk(g_mtx);
        auto old = g_swapchains.find(pCreateInfo->oldSwapchain);
        if (old != g_swapchains.end()) old->second.cleanup(dd, old->second.device);
        g_swapchains.erase(pCreateInfo->oldSwapchain);
        g_swapDevice.erase(pCreateInfo->oldSwapchain);
    }

    VkResult res = dd.createSwapchain(device, &ci, pAllocator, pSwapchain);
    if (res != VK_SUCCESS) return res;

    // Get swapchain images
    uint32_t imgCount = 0;
    dd.getSwapchainImages(device, *pSwapchain, &imgCount, nullptr);
    std::vector<VkImage> imgs(imgCount);
    dd.getSwapchainImages(device, *pSwapchain, &imgCount, imgs.data());

    {
        std::lock_guard<std::mutex> lk(g_mtx);
        g_swapDevice[*pSwapchain] = dispatchKey(device);
    }

    if (!conf.enabled) {
        BFG_LAYER("Layer disabled (BIONIC_FG_ENABLE not set)");
        std::lock_guard<std::mutex> lk(g_mtx);
        SwapState st; st.device = device; st.extent = ci.imageExtent;
        st.format = ci.imageFormat; st.images = imgs; st.conf = conf;
        g_swapchains[*pSwapchain] = std::move(st);
        return VK_SUCCESS;
    }

#ifdef __ANDROID__
    SwapState st;
    st.device = device;
    st.extent = ci.imageExtent;
    st.format = ci.imageFormat;
    st.images = imgs;
    st.conf   = conf;

    try {
        const uint32_t W = ci.imageExtent.width, H = ci.imageExtent.height;
        const int requestedGeneratedFrames = std::max(conf.multiplier - 1, 0);
        const int provisionedGeneratedFrames = kMaxHotReloadGeneratedFrames;

        // Allocate AHBs
        auto allocAhb = [&](AHardwareBuffer** out) {
            AHardwareBuffer_Desc d{};
            d.width  = W; d.height = H; d.layers = 1;
            d.format = AHARDWAREBUFFER_FORMAT_R8G8B8A8_UNORM;
            d.usage  = AHARDWAREBUFFER_USAGE_GPU_FRAMEBUFFER
                     | AHARDWAREBUFFER_USAGE_GPU_SAMPLED_IMAGE;
            if (AHardwareBuffer_allocate(&d, out) != 0 || !*out)
                throw std::runtime_error("failed to allocate AHardwareBuffer");
        };
        allocAhb(&st.prevAhb);
        allocAhb(&st.currAhb);
        for (int k = 0; k < provisionedGeneratedFrames; ++k) {
            AHardwareBuffer* o = nullptr; allocAhb(&o);
            st.outAhbs.push_back(o);
        }

        // Import AHBs on producer device (for copy ops)
        const VkImageUsageFlags copyUsage = VK_IMAGE_USAGE_TRANSFER_DST_BIT
                                           | VK_IMAGE_USAGE_SAMPLED_BIT
                                           | VK_IMAGE_USAGE_STORAGE_BIT;
        const VkImageUsageFlags outUsage  = VK_IMAGE_USAGE_TRANSFER_SRC_BIT
                                           | VK_IMAGE_USAGE_SAMPLED_BIT
                                           | VK_IMAGE_USAGE_STORAGE_BIT;
        st.prevProducerImg = importAhb(dd, device, st.prevAhb, {W,H}, VK_FORMAT_R8G8B8A8_UNORM, copyUsage);
        st.currProducerImg = importAhb(dd, device, st.currAhb, {W,H}, VK_FORMAT_R8G8B8A8_UNORM, copyUsage);
        if (!st.prevProducerImg.img || !st.currProducerImg.img)
            throw std::runtime_error("failed to import input AHBs on producer device");
        for (int k = 0; k < provisionedGeneratedFrames; ++k) {
            st.outProducerImgs.push_back(importAhb(dd, device, st.outAhbs[size_t(k)], {W,H}, VK_FORMAT_R8G8B8A8_UNORM, outUsage));
            if (!st.outProducerImgs.back().img)
                throw std::runtime_error("failed to import output AHB on producer device");
        }

        // Create copy command pool
        VkCommandPoolCreateInfo cpci{};
        cpci.sType            = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO;
        cpci.queueFamilyIndex = dd.queueFamilyIdx;
        st.copyQueueFamily    = dd.queueFamilyIdx;
        cpci.flags            = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;
        dd.createCmdPool(device, &cpci, nullptr, &st.copyPool);

        VkCommandBufferAllocateInfo cbai{};
        cbai.sType              = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
        cbai.commandPool        = st.copyPool;
        cbai.level              = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
        cbai.commandBufferCount = 2;
        if (dd.allocCmdBufs(device, &cbai, st.copyCmds) != VK_SUCCESS)
            throw std::runtime_error("failed to allocate copy command buffers");
        cbai.commandBufferCount = kMaxHotReloadPresentOps;
        if (dd.allocCmdBufs(device, &cbai, st.genCmds) != VK_SUCCESS)
            throw std::runtime_error("failed to allocate output command buffers");

        VkFenceCreateInfo fci{};
        fci.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO;
        fci.flags = VK_FENCE_CREATE_SIGNALED_BIT;
        if (dd.createFence(device, &fci, nullptr, &st.copyFences[0]) != VK_SUCCESS ||
            dd.createFence(device, &fci, nullptr, &st.copyFences[1]) != VK_SUCCESS)
            throw std::runtime_error("failed to create copy fences");
        for (int k = 0; k < kMaxHotReloadPresentOps && k < 4; ++k) {
            if (dd.createFence(device, &fci, nullptr, &st.genFences[k]) != VK_SUCCESS)
                throw std::runtime_error("failed to create output fence");
        }

        VkSemaphoreCreateInfo sci{};
        sci.sType = VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO;
        for (int k = 0; k < kMaxHotReloadPresentOps && k < 4; ++k) {
            if (dd.createSemaphore(device, &sci, nullptr, &st.acquireSems[k]) != VK_SUCCESS ||
                dd.createSemaphore(device, &sci, nullptr, &st.presentSems[k]) != VK_SUCCESS)
                throw std::runtime_error("failed to create output semaphores");
        }

        // Create FramegenContext only when generation is currently active.
        // The swapchain/AHB set is still provisioned for full 2x..4x hot-reload,
        // so turning framegen on later does not require app-side swapchain rebuilds.
        if (requestedGeneratedFrames > 0) {
            st.fgCtx = createFramegenContext(st, conf);
            if (!st.fgCtx) {
                BFG_LAYER_E("FramegenContext creation failed — layer will passthrough");
            } else {
                BFG_LAYER("SwapchainState ready: %ux%u mult=%d provisionedOutputs=%d shaders=embedded",
                          W, H, requestedGeneratedFrames + 1, provisionedGeneratedFrames);
            }
        } else {
            BFG_LAYER("SwapchainState provisioned: %ux%u framegen=off provisionedOutputs=%d",
                      W, H, provisionedGeneratedFrames);
        }
    } catch (const std::exception& e) {
        BFG_LAYER_E("Exception in CreateSwapchainKHR setup — layer will passthrough: %s", e.what());
        st.cleanup(dd, device);
    } catch (...) {
        BFG_LAYER_E("Unknown exception in CreateSwapchainKHR setup — layer will passthrough");
        st.cleanup(dd, device);
    }

    {
        std::lock_guard<std::mutex> lk(g_mtx);
        g_swapchains[*pSwapchain] = std::move(st);
    }
#endif
    return VK_SUCCESS;
}

VKAPI_ATTR void VKAPI_CALL BionicFG_DestroySwapchainKHR(
        VkDevice device,
        VkSwapchainKHR swapchain,
        const VkAllocationCallbacks* pAllocator) {
    {
        std::lock_guard<std::mutex> lk(g_mtx);
        auto dit = g_devices.find(dispatchKey(device));
        auto sit = g_swapchains.find(swapchain);
#ifdef __ANDROID__
        cancelScheduledPresents(swapchain);
#endif
        if (dit != g_devices.end() && sit != g_swapchains.end())
            sit->second.cleanup(dit->second, sit->second.device);
        g_swapchains.erase(swapchain);
        g_swapDevice.erase(swapchain);
    }
    auto it = g_devices.find(dispatchKey(device));
    if (it != g_devices.end())
        it->second.destroySwapchain(device, swapchain, pAllocator);
}

// ─── Present / acquire helpers ───────────────────────────────────────────────

static bool findDeviceForQueueOrSwapchain(VkQueue queue, VkSwapchainKHR swapchain,
                                           void** outDevKey, QueueData* outQueueData = nullptr) {
    std::lock_guard<std::mutex> lk(g_mtx);
    auto qit = g_queues.find(queue);
    if (qit != g_queues.end()) {
        if (outDevKey) *outDevKey = qit->second.deviceKey;
        if (outQueueData) *outQueueData = qit->second;
        return qit->second.deviceKey != nullptr;
    }
    auto sit = g_swapDevice.find(swapchain);
    if (sit != g_swapDevice.end()) {
        if (outDevKey) *outDevKey = sit->second;
        if (outQueueData) *outQueueData = QueueData{sit->second, VK_QUEUE_FAMILY_IGNORED};
        return sit->second != nullptr;
    }
    return false;
}

static VkResult callNextPresent(VkQueue queue, const VkPresentInfoKHR* presentInfo) {
    if (!presentInfo || presentInfo->swapchainCount == 0)
        return vkQueuePresentKHR(queue, presentInfo);
    void* devKey = nullptr;
    findDeviceForQueueOrSwapchain(queue, presentInfo->pSwapchains[0], &devKey, nullptr);
    PFN_vkQueuePresentKHR nextPresent = nullptr;
    {
        std::lock_guard<std::mutex> lk(g_mtx);
        auto dit = g_devices.find(devKey);
        if (dit != g_devices.end()) nextPresent = dit->second.queuePresent;
    }
    return nextPresent ? nextPresent(queue, presentInfo) : vkQueuePresentKHR(queue, presentInfo);
}

static VkResult acquireGeneratedImage(const DeviceData& dd, VkDevice device,
                                      VkSwapchainKHR swapchain, VkSemaphore signalSemaphore,
                                      uint32_t* imageIndex) {
    // Pace generated presents by blocking until the presentation engine releases
    // the next FIFO slot/image instead of best-effort grabbing whatever is
    // immediately available. This mirrors lsfg-vk's Android path and avoids the
    // bursty "present everything right now or skip it" behavior from timeout=0.
    constexpr uint64_t kAcquireTimeoutNs = UINT64_MAX;

    if (dd.acquireNextImage) {
        return dd.acquireNextImage(device, swapchain, kAcquireTimeoutNs,
                                   signalSemaphore, VK_NULL_HANDLE, imageIndex);
    }
    if (dd.acquireNextImage2) {
        VkAcquireNextImageInfoKHR info{};
        info.sType      = VK_STRUCTURE_TYPE_ACQUIRE_NEXT_IMAGE_INFO_KHR;
        info.swapchain  = swapchain;
        info.timeout    = kAcquireTimeoutNs;
        info.semaphore  = signalSemaphore;
        info.fence      = VK_NULL_HANDLE;
        info.deviceMask = 1;
        return dd.acquireNextImage2(device, &info, imageIndex);
    }
    return VK_ERROR_EXTENSION_NOT_PRESENT;
}

#ifdef __ANDROID__
static VkResult presentSwapchainImage(const DeviceData& dd, VkQueue queue,
                                      VkSwapchainKHR swapchain, uint32_t imageIndex,
                                      const void* pNext = nullptr) {
    VkPresentInfoKHR presentInfo{};
    presentInfo.sType = VK_STRUCTURE_TYPE_PRESENT_INFO_KHR;
    presentInfo.pNext = pNext;
    presentInfo.swapchainCount = 1;
    presentInfo.pSwapchains = &swapchain;
    presentInfo.pImageIndices = &imageIndex;
    return queuePresentLocked(dd, queue, &presentInfo);
}

static VkResult blitProducerToSwapchainAndWait(
        const DeviceData& dd,
        SwapState& st,
        VkQueue queue,
        const VkCommandBufferBeginInfo& cbi,
        int slot,
        ExtImage& src,
        uint32_t dstImageIndex,
        VkImageLayout dstOldLayout,
        uint32_t waitSemaphoreCount = 0,
        const VkSemaphore* waitSemaphores = nullptr,
        VkFilter filter = VK_FILTER_LINEAR) {
    if (slot < 0 || slot >= kMaxHotReloadPresentOps)
        return VK_ERROR_INITIALIZATION_FAILED;
    if (dstImageIndex >= st.images.size())
        return VK_ERROR_OUT_OF_DATE_KHR;
    if (!st.genCmds[slot] || !st.genFences[slot])
        return VK_ERROR_INITIALIZATION_FAILED;

    VkFence fence = st.genFences[slot];
    VkResult waitRes = dd.waitForFences(st.device, 1, &fence, VK_TRUE, UINT64_MAX);
    if (waitRes != VK_SUCCESS) return waitRes;
    VkResult resetRes = dd.resetFences(st.device, 1, &fence);
    if (resetRes != VK_SUCCESS) return resetRes;

    VkCommandBuffer cmd = st.genCmds[slot];
    dd.resetCmdBuf(cmd, 0);
    if (dd.beginCmdBuf(cmd, &cbi) != VK_SUCCESS)
        return VK_ERROR_INITIALIZATION_FAILED;

    const VkImageLayout oldSrcLayout = src.layout;
    const bool oldSrcExternalOwner = src.externalOwner;

    uint32_t srcSrcFamily = src.externalOwner ? VK_QUEUE_FAMILY_EXTERNAL : VK_QUEUE_FAMILY_IGNORED;
    uint32_t srcDstFamily = src.externalOwner ? st.copyQueueFamily : VK_QUEUE_FAMILY_IGNORED;
    layerImageBarrier(dd, cmd, src.img,
        VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, 0,
        VK_PIPELINE_STAGE_TRANSFER_BIT, VK_ACCESS_TRANSFER_READ_BIT,
        src.layout, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
        srcSrcFamily, srcDstFamily);
    src.layout = VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL;
    src.externalOwner = false;

    layerImageBarrier(dd, cmd, st.images[dstImageIndex],
        VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, 0,
        VK_PIPELINE_STAGE_TRANSFER_BIT, VK_ACCESS_TRANSFER_WRITE_BIT,
        dstOldLayout, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL);

    VkImageBlit blit{};
    blit.srcSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1};
    blit.srcOffsets[1]  = {(int32_t)st.extent.width, (int32_t)st.extent.height, 1};
    blit.dstSubresource = blit.srcSubresource;
    blit.dstOffsets[1]  = blit.srcOffsets[1];
    dd.cmdBlitImage(cmd,
        src.img, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
        st.images[dstImageIndex], VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
        1, &blit, filter);

    layerImageBarrier(dd, cmd, src.img,
        VK_PIPELINE_STAGE_TRANSFER_BIT, VK_ACCESS_TRANSFER_READ_BIT,
        VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT, 0,
        VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, VK_IMAGE_LAYOUT_GENERAL,
        st.copyQueueFamily, VK_QUEUE_FAMILY_EXTERNAL);
    src.layout = VK_IMAGE_LAYOUT_GENERAL;
    src.externalOwner = true;

    layerImageBarrier(dd, cmd, st.images[dstImageIndex],
        VK_PIPELINE_STAGE_TRANSFER_BIT, VK_ACCESS_TRANSFER_WRITE_BIT,
        VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT, 0,
        VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, VK_IMAGE_LAYOUT_PRESENT_SRC_KHR);

    if (dd.endCmdBuf(cmd) != VK_SUCCESS) {
        src.layout = oldSrcLayout;
        src.externalOwner = oldSrcExternalOwner;
        return VK_ERROR_INITIALIZATION_FAILED;
    }

    std::vector<VkPipelineStageFlags> waitStages(waitSemaphoreCount,
                                                 VK_PIPELINE_STAGE_TRANSFER_BIT);
    VkSubmitInfo si{};
    si.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
    si.waitSemaphoreCount = waitSemaphoreCount;
    si.pWaitSemaphores = waitSemaphores;
    si.pWaitDstStageMask = waitStages.empty() ? nullptr : waitStages.data();
    si.commandBufferCount = 1;
    si.pCommandBuffers = &cmd;
    VkResult submitRes = queueSubmitLocked(dd, queue, 1, &si, fence);
    if (submitRes != VK_SUCCESS) {
        src.layout = oldSrcLayout;
        src.externalOwner = oldSrcExternalOwner;
        return submitRes;
    }

    return dd.waitForFences(st.device, 1, &fence, VK_TRUE, UINT64_MAX);
}
#endif

// ─── QueuePresentKHR ──────────────────────────────────────────────────────────

VKAPI_ATTR VkResult VKAPI_CALL BionicFG_QueuePresentKHR(
        VkQueue queue,
        const VkPresentInfoKHR* pPresentInfo) {
    if (!pPresentInfo || pPresentInfo->swapchainCount != 1) {
        if (pPresentInfo && pPresentInfo->swapchainCount > 1)
            BFG_LAYER("multi-swapchain present (%u) is not transformed; passing through", pPresentInfo->swapchainCount);
        return callNextPresent(queue, pPresentInfo);
    }

    VkSwapchainKHR swapchain = pPresentInfo->pSwapchains[0];
    uint32_t imgIdx = pPresentInfo->pImageIndices ? pPresentInfo->pImageIndices[0] : UINT32_MAX;

    void* devKey = nullptr;
    QueueData qd{};
    if (!findDeviceForQueueOrSwapchain(queue, swapchain, &devKey, &qd))
        return callNextPresent(queue, pPresentInfo);

    DeviceData dd;
    {
        std::lock_guard<std::mutex> lk(g_mtx);
        auto dit = g_devices.find(devKey);
        if (dit == g_devices.end()) return callNextPresent(queue, pPresentInfo);
        dd = dit->second;
    }

    SwapState* stPtr = nullptr;
    {
        std::lock_guard<std::mutex> lk(g_mtx);
        auto it = g_swapchains.find(swapchain);
        if (it == g_swapchains.end()) return callNextPresent(queue, pPresentInfo);
        stPtr = &it->second;
    }
    SwapState& st = *stPtr;

    if (!st.conf.configPath.empty()) {
        int64_t currentStamp = fileStamp(st.conf.configPath);
        if (currentStamp != 0 && currentStamp != st.conf.configStamp) {
            const LayerConf oldConf = st.conf;
            LayerConf newConf = readConf();

            if (configNeedsSwapchainRecreate(oldConf, newConf)) {
                st.conf = newConf;
                BFG_LAYER("config changed; requesting swapchain recreate enabled=%d mult=%d flow=%.2f model=%d",
                          st.conf.enabled ? 1 : 0, st.conf.multiplier, st.conf.flowScale, st.conf.model);
                return VK_ERROR_OUT_OF_DATE_KHR;
            }

#ifdef __ANDROID__
            const bool oldActive = oldConf.multiplier >= 2;
            const bool newActive = newConf.multiplier >= 2;

            if (!newActive) {
                if (st.fgCtx) {
                    st.fgCtx->destroy();
                    st.fgCtx.reset();
                }
                resetFrameHistory(st);
                st.conf = newConf;
                BFG_LAYER("config hot-reloaded framegen=off flow=%.2f model=%d",
                          st.conf.flowScale, st.conf.model);
            } else {
                const bool needsContextRebuild =
                    !oldActive ||
                    !st.fgCtx ||
                    (oldConf.multiplier != newConf.multiplier) ||
                    (oldConf.model != newConf.model);

                if (needsContextRebuild) {
                    if (rebuildFramegenContext(st, newConf)) {
                        st.conf = newConf;
                        BFG_LAYER("config hot-reloaded mult=%d flow=%.2f model=%d via context rebuild",
                                  st.conf.multiplier, st.conf.flowScale, st.conf.model);
                    } else {
                        BFG_LAYER_E("config rebuild failed; keeping old config enabled=%d mult=%d flow=%.2f model=%d",
                                    oldConf.enabled ? 1 : 0, oldConf.multiplier, oldConf.flowScale, oldConf.model);
                    }
                } else {
                    st.conf = newConf;
                    if (st.fgCtx) {
                        st.fgCtx->updateConfig(makeFramegenConfig(st, st.conf));
                        BFG_LAYER("config hot-reloaded flow=%.2f", st.conf.flowScale);
                    }
                }
            }
#else
            {
                st.conf = newConf;
            }
#endif
        }
    }

    if (st.inPresent || !st.conf.enabled || !st.fgCtx)
        return callNextPresent(queue, pPresentInfo);
    if (imgIdx >= st.images.size()) {
        BFG_LAYER_E("present image index %u out of range (%zu); passing through", imgIdx, st.images.size());
        return callNextPresent(queue, pPresentInfo);
    }
    if (qd.family != VK_QUEUE_FAMILY_IGNORED && st.copyQueueFamily != VK_QUEUE_FAMILY_IGNORED &&
        qd.family != st.copyQueueFamily) {
        BFG_LAYER_E("present queue family changed from %u to %u; passing through to avoid invalid command pool",
                    st.copyQueueFamily, qd.family);
        return callNextPresent(queue, pPresentInfo);
    }

#ifndef __ANDROID__
    return callNextPresent(queue, pPresentInfo);
#else
    st.inPresent = true;
    struct Guard { SwapState& s; ~Guard(){ s.inPresent = false; } } guard{st};

    const int64_t callbackStartNs = steadyNowNs();
    const uint64_t sourceSeq = ++st.debugSourceSeq;
    double srcDeltaMs = -1.0;
    if (st.debugLastSourceStartNs != 0)
        srcDeltaMs = nsToMs(callbackStartNs - st.debugLastSourceStartNs);
    st.debugLastSourceStartNs = callbackStartNs;
    const bool traceThisFrame = st.conf.debugTiming &&
        (st.debugTraceFramesRemaining > 0 || (sourceSeq % 30ull) == 0ull);
    if (traceThisFrame && st.debugTraceFramesRemaining > 0)
        st.debugTraceFramesRemaining--;

    std::ostringstream trace;
    if (traceThisFrame) {
        trace << std::fixed << std::setprecision(3)
              << "timing src#" << sourceSeq
              << " fc=" << st.frameCount
              << " mult=" << st.conf.multiplier
              << " model=" << st.conf.model
              << " img=" << imgIdx;
        if (srcDeltaMs >= 0.0)
            trace << " srcDelta=" << srcDeltaMs << "ms";
        if (st.conf.pacePresent)
            trace << " pace=" << st.conf.paceIntervalMs << "ms";
    }

    auto finish = [&](VkResult res) -> VkResult {
        const int64_t callbackNs = steadyNowNs() - callbackStartNs;
        debugRecordCallbackTime(st, callbackNs);
        if (traceThisFrame) {
            trace << " cbTotal=" << nsToMs(callbackNs) << "ms"
                  << " result=" << static_cast<int>(res);
            BFG_LAYER("%s", trace.str().c_str());
        }
        return res;
    };

    const uint32_t fi = static_cast<uint32_t>(st.frameCount & 1u);
    VkDevice device = st.device;
    const int N = std::min(st.conf.multiplier - 1, 4);
    bool consumedOriginalWaits = false;
    VkCommandBufferBeginInfo cbi{};

    int64_t pacedPresentBaseNs = 0;
    const int64_t paceIntervalNs = static_cast<int64_t>(st.conf.paceIntervalMs * 1000000.0 + 0.5);

    auto recordPresentTrace = [&](const char* label, int slot, uint32_t presentImage,
                                  VkResult res, int64_t presentStartNs) {
        const int64_t presentEndNs = steadyNowNs();
        const double gapMs = debugRecordOutputPresent(st, presentEndNs);
        if (traceThisFrame) {
            trace << ' ' << label
                  << "{slot=" << slot
                  << ",img=" << presentImage
                  << ",present=" << nsToMs(presentEndNs - presentStartNs) << "ms";
            if (gapMs >= 0.0)
                trace << ",gapPrev=" << gapMs << "ms";
            trace << ",res=" << static_cast<int>(res) << '}';
        }
    };

    auto presentInfoAndTrace = [&](const char* label, int slot, const VkPresentInfoKHR& info, uint32_t presentImage) -> VkResult {
        const int64_t presentStartNs = steadyNowNs();
        VkResult res = queuePresentLocked(dd, queue, &info);
        recordPresentTrace(label, slot, presentImage, res, presentStartNs);
        return res;
    };

    auto presentImageAndTrace = [&](const char* label, int slot, uint32_t presentImage, const void* pNext) -> VkResult {
        const int64_t presentStartNs = steadyNowNs();
        VkResult res = presentSwapchainImage(dd, queue, swapchain, presentImage, pNext);
        recordPresentTrace(label, slot, presentImage, res, presentStartNs);
        if (st.conf.pacePresent && slot == 0 && (res == VK_SUCCESS || res == VK_SUBOPTIMAL_KHR))
            pacedPresentBaseNs = steadyNowNs();
        return res;
    };

    auto scheduleImageAndTrace = [&](const char* label, int slot, uint32_t presentImage, const void* pNext = nullptr) {
        if (!st.conf.pacePresent || slot <= 0 || pacedPresentBaseNs == 0 || paceIntervalNs <= 0) {
            return presentImageAndTrace(label, slot, presentImage, pNext);
        }
        ScheduledPresent item{};
        item.targetNs = pacedPresentBaseNs + static_cast<int64_t>(slot) * paceIntervalNs;
        item.intervalNs = paceIntervalNs;
        item.present = dd.queuePresent;
        item.queue = queue;
        item.swapchain = swapchain;
        item.imageIndex = presentImage;
        item.sourceSeq = sourceSeq;
        item.slot = slot;
        std::snprintf(item.label, sizeof(item.label), "%s", label);
        schedulePresent(item);
        if (traceThisFrame) {
            trace << ' ' << label
                  << "{slot=" << slot
                  << ",img=" << presentImage
                  << ",scheduled=+" << nsToMs(item.targetNs - pacedPresentBaseNs) << "ms}";
        }
        return VK_SUCCESS;
    };

    auto acquireAndTrace = [&](int slot, VkSemaphore semaphore, uint32_t* outImageIndex) -> VkResult {
        const int64_t acquireStartNs = steadyNowNs();
        VkResult res = acquireGeneratedImage(dd, device, swapchain, semaphore, outImageIndex);
        const int64_t acquireNs = steadyNowNs() - acquireStartNs;
        debugRecordAcquireWait(st, acquireNs);
        if (traceThisFrame) {
            trace << " slot" << slot << "Acq=" << nsToMs(acquireNs) << "ms";
            if (outImageIndex && *outImageIndex != UINT32_MAX)
                trace << "->" << *outImageIndex;
            trace << "(" << static_cast<int>(res) << ')';
        }
        return res;
    };

    auto blitAndTrace = [&](const char* label, int slot, ExtImage& src, uint32_t dstImageIndex,
                            VkImageLayout dstOldLayout, uint32_t waitSemaphoreCount,
                            const VkSemaphore* waitSemaphores, VkFilter filter) -> VkResult {
        const int64_t copyStartNs = steadyNowNs();
        VkResult res = blitProducerToSwapchainAndWait(
            dd, st, queue, cbi, slot, src, dstImageIndex,
            dstOldLayout, waitSemaphoreCount, waitSemaphores, filter);
        if (traceThisFrame) {
            trace << ' ' << label
                  << "Copy=" << nsToMs(steadyNowNs() - copyStartNs) << "ms"
                  << "->" << dstImageIndex
                  << "(" << static_cast<int>(res) << ')';
        }
        return res;
    };

    auto advanceFrameHistory = [&]() {
        std::swap(st.prevAhb, st.currAhb);
        std::swap(st.prevProducerImg, st.currProducerImg);
        if (st.fgCtx) {
            st.fgCtx->updateConfig(makeFramegenConfig(st, st.conf));
        }
        st.frameCount++;
    };

    VkResult overallResult = VK_SUCCESS;
    auto noteResult = [&](VkResult res) {
        if (res == VK_SUBOPTIMAL_KHR && overallResult == VK_SUCCESS) {
            overallResult = VK_SUBOPTIMAL_KHR;
        }
    };

    if (!dd.resetCmdBuf || !dd.beginCmdBuf || !dd.endCmdBuf || !dd.queueSubmit || !dd.waitForFences ||
        !dd.resetFences || !dd.cmdBlitImage || !st.copyCmds[fi] || !st.copyFences[fi]) {
        BFG_LAYER_E("copy command infrastructure incomplete; passing through");
        return finish(callNextPresent(queue, pPresentInfo));
    }

    const int64_t inputCopyStartNs = steadyNowNs();

    // --- Step 1: Copy real swapchain image into current AHB input. The copy
    // waits on the application's original present semaphores, so the final real
    // present must not wait on those same binary semaphores again.
    if (dd.waitForFences(device, 1, &st.copyFences[fi], VK_TRUE, UINT64_MAX) != VK_SUCCESS ||
        dd.resetFences(device, 1, &st.copyFences[fi]) != VK_SUCCESS) {
        BFG_LAYER_E("copy fence wait/reset failed; passing through");
        return finish(callNextPresent(queue, pPresentInfo));
    }

    dd.resetCmdBuf(st.copyCmds[fi], 0);
    cbi.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
    cbi.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
    if (dd.beginCmdBuf(st.copyCmds[fi], &cbi) != VK_SUCCESS) {
        BFG_LAYER_E("begin copy command buffer failed; passing through");
        return finish(callNextPresent(queue, pPresentInfo));
    }

    VkCommandBuffer cmd = st.copyCmds[fi];
    layerImageBarrier(dd, cmd, st.images[imgIdx],
        VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, 0,
        VK_PIPELINE_STAGE_TRANSFER_BIT, VK_ACCESS_TRANSFER_READ_BIT,
        VK_IMAGE_LAYOUT_PRESENT_SRC_KHR, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL);

    uint32_t currSrcFamily = st.currProducerImg.externalOwner ? VK_QUEUE_FAMILY_EXTERNAL : VK_QUEUE_FAMILY_IGNORED;
    uint32_t currDstFamily = st.currProducerImg.externalOwner ? st.copyQueueFamily : VK_QUEUE_FAMILY_IGNORED;
    layerImageBarrier(dd, cmd, st.currProducerImg.img,
        VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, 0,
        VK_PIPELINE_STAGE_TRANSFER_BIT, VK_ACCESS_TRANSFER_WRITE_BIT,
        st.currProducerImg.layout, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
        currSrcFamily, currDstFamily);
    st.currProducerImg.layout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
    st.currProducerImg.externalOwner = false;

    VkImageBlit blit{};
    blit.srcSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1};
    blit.srcOffsets[1]  = {(int32_t)st.extent.width, (int32_t)st.extent.height, 1};
    blit.dstSubresource = blit.srcSubresource;
    blit.dstOffsets[1]  = blit.srcOffsets[1];
    dd.cmdBlitImage(cmd, st.images[imgIdx], VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
        st.currProducerImg.img, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
        1, &blit, VK_FILTER_NEAREST);

    layerImageBarrier(dd, cmd, st.images[imgIdx],
        VK_PIPELINE_STAGE_TRANSFER_BIT, VK_ACCESS_TRANSFER_READ_BIT,
        VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT, 0,
        VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, VK_IMAGE_LAYOUT_PRESENT_SRC_KHR);
    layerImageBarrier(dd, cmd, st.currProducerImg.img,
        VK_PIPELINE_STAGE_TRANSFER_BIT, VK_ACCESS_TRANSFER_WRITE_BIT,
        VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT, 0,
        VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, VK_IMAGE_LAYOUT_GENERAL,
        st.copyQueueFamily, VK_QUEUE_FAMILY_EXTERNAL);
    st.currProducerImg.layout = VK_IMAGE_LAYOUT_GENERAL;
    st.currProducerImg.externalOwner = true;

    if (dd.endCmdBuf(cmd) != VK_SUCCESS) {
        BFG_LAYER_E("end copy command buffer failed; passing through");
        return finish(callNextPresent(queue, pPresentInfo));
    }

    std::vector<VkPipelineStageFlags> waitStages(pPresentInfo->waitSemaphoreCount,
                                                 VK_PIPELINE_STAGE_TRANSFER_BIT);
    VkSubmitInfo si{};
    si.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
    si.waitSemaphoreCount = pPresentInfo->waitSemaphoreCount;
    si.pWaitSemaphores = pPresentInfo->pWaitSemaphores;
    si.pWaitDstStageMask = waitStages.empty() ? nullptr : waitStages.data();
    si.commandBufferCount = 1;
    si.pCommandBuffers = &cmd;
    VkResult submitRes = queueSubmitLocked(dd, queue, 1, &si, st.copyFences[fi]);
    if (submitRes != VK_SUCCESS) {
        BFG_LAYER_E("copy queue submit failed: %d; passing through", submitRes);
        return finish(callNextPresent(queue, pPresentInfo));
    }
    consumedOriginalWaits = pPresentInfo->waitSemaphoreCount > 0;

    VkPresentInfoKHR finalPresent = *pPresentInfo;
    if (consumedOriginalWaits) {
        finalPresent.waitSemaphoreCount = 0;
        finalPresent.pWaitSemaphores = nullptr;
    }

    VkResult copyWait = dd.waitForFences(device, 1, &st.copyFences[fi], VK_TRUE, UINT64_MAX);
    if (copyWait != VK_SUCCESS) {
        BFG_LAYER_E("copy fence wait failed after submit: %d", copyWait);
        return finish(presentInfoAndTrace("copywaitFallback", -1, finalPresent, imgIdx));
    }
    if (traceThisFrame)
        trace << " inputCopy=" << nsToMs(steadyNowNs() - inputCopyStartNs) << "ms";

    // Warmup frame: seed previous/current history, but still present the app's
    // original image normally because no generated output exists yet.
    if (st.frameCount == 0 || N <= 0) {
        if (traceThisFrame) trace << " warmup";
        advanceFrameHistory();
        return finish(presentInfoAndTrace("warmupReal", -1, finalPresent, imgIdx));
    }

    // --- Step 2: Run framegen after we have both previous and current inputs.
    const int64_t fgStartNs = steadyNowNs();
    st.fgCtx->present(st.prevAhb, st.currAhb);
    st.fgCtx->waitIdle();
    const int64_t fgNs = steadyNowNs() - fgStartNs;
    debugRecordFramegenTime(st, fgNs);
    if (traceThisFrame)
        trace << " fg=" << nsToMs(fgNs) << "ms";
    for (auto& out : st.outProducerImgs) {
        out.layout = VK_IMAGE_LAYOUT_GENERAL;
        out.externalOwner = true;
    }

    // --- Step 3: Use the app-owned present image for the first generated frame
    // so the app's acquired image is still released this callback. Then acquire
    // future swapchain images for the remaining generated frames and the copied
    // current real frame, which lands on the last future slot.
    VkResult firstCopy = blitAndTrace(
        "gen0", 0, st.outProducerImgs[0], imgIdx,
        VK_IMAGE_LAYOUT_PRESENT_SRC_KHR,
        0, nullptr, VK_FILTER_LINEAR);
    if (firstCopy != VK_SUCCESS) {
        BFG_LAYER_E("first generated frame copy failed: %d; presenting original real frame", firstCopy);
        advanceFrameHistory();
        return finish(presentInfoAndTrace("fallbackReal", -1, finalPresent, imgIdx));
    }

    VkResult firstPresent = presentImageAndTrace("gen0", 0, imgIdx, pPresentInfo->pNext);
    noteResult(firstPresent);
    if (firstPresent != VK_SUCCESS && firstPresent != VK_SUBOPTIMAL_KHR) {
        BFG_LAYER_E("first generated frame present failed: %d", firstPresent);
        advanceFrameHistory();
        return finish(firstPresent);
    }

    for (int k = 1; k < N; ++k) {
        if (!st.acquireSems[k] || !st.genCmds[k] || !st.genFences[k]) {
            BFG_LAYER_E("generated frame slot %d is not provisioned", k);
            advanceFrameHistory();
            return finish(overallResult);
        }

        uint32_t genIdx = UINT32_MAX;
        VkResult acqRes = acquireAndTrace(k, st.acquireSems[k], &genIdx);
        noteResult(acqRes);
        if (acqRes != VK_SUCCESS && acqRes != VK_SUBOPTIMAL_KHR) {
            BFG_LAYER_E("generated frame acquire failed at slot %d: %d", k, acqRes);
            advanceFrameHistory();
            return finish(overallResult);
        }
        if (genIdx >= st.images.size()) {
            BFG_LAYER_E("generated image index %u out of range (%zu)", genIdx, st.images.size());
            advanceFrameHistory();
            return finish(overallResult);
        }

        VkResult genCopy = blitAndTrace(
            (std::string("gen") + std::to_string(k)).c_str(), k,
            st.outProducerImgs[size_t(k)], genIdx,
            VK_IMAGE_LAYOUT_UNDEFINED,
            1, &st.acquireSems[k], VK_FILTER_LINEAR);
        if (genCopy != VK_SUCCESS) {
            BFG_LAYER_E("generated frame copy failed at slot %d: %d", k, genCopy);
            advanceFrameHistory();
            return finish(overallResult);
        }

        VkResult genPresent = scheduleImageAndTrace((std::string("gen") + std::to_string(k)).c_str(), k, genIdx);
        noteResult(genPresent);
        if (genPresent != VK_SUCCESS && genPresent != VK_SUBOPTIMAL_KHR) {
            BFG_LAYER_E("generated frame present failed at slot %d: %d", k, genPresent);
            advanceFrameHistory();
            return finish(overallResult);
        }
    }

    const int realSlot = N;
    if (!st.acquireSems[realSlot] || !st.genCmds[realSlot] || !st.genFences[realSlot]) {
        BFG_LAYER_E("real frame slot %d is not provisioned", realSlot);
        advanceFrameHistory();
        return finish(overallResult);
    }

    uint32_t realIdx = UINT32_MAX;
    VkResult realAcq = acquireAndTrace(realSlot, st.acquireSems[realSlot], &realIdx);
    noteResult(realAcq);
    if (realAcq != VK_SUCCESS && realAcq != VK_SUBOPTIMAL_KHR) {
        BFG_LAYER_E("real frame acquire failed at slot %d: %d", realSlot, realAcq);
        advanceFrameHistory();
        return finish(overallResult);
    }
    if (realIdx >= st.images.size()) {
        BFG_LAYER_E("real frame image index %u out of range (%zu)", realIdx, st.images.size());
        advanceFrameHistory();
        return finish(overallResult);
    }

    VkResult realCopy = blitAndTrace(
        "real", realSlot, st.currProducerImg, realIdx,
        VK_IMAGE_LAYOUT_UNDEFINED,
        1, &st.acquireSems[realSlot], VK_FILTER_NEAREST);
    if (realCopy != VK_SUCCESS) {
        BFG_LAYER_E("real frame copy failed at slot %d: %d", realSlot, realCopy);
        advanceFrameHistory();
        return finish(overallResult);
    }

    VkResult realPresent = scheduleImageAndTrace("real", realSlot, realIdx);
    noteResult(realPresent);
    if (realPresent != VK_SUCCESS && realPresent != VK_SUBOPTIMAL_KHR) {
        BFG_LAYER_E("real frame present failed at slot %d: %d", realSlot, realPresent);
        advanceFrameHistory();
        return finish(overallResult);
    }

    advanceFrameHistory();
    return finish(overallResult);
#endif
}

// ─── GetProcAddr entry points ─────────────────────────────────────────────────

#define HOOK(fn) if (std::strcmp(name, "vk" #fn) == 0 || std::strcmp(name, #fn) == 0) return reinterpret_cast<PFN_vkVoidFunction>(BionicFG_##fn)

VKAPI_ATTR PFN_vkVoidFunction VKAPI_CALL BionicFG_GetDeviceProcAddr(
        VkDevice device, const char* name) {
    HOOK(DestroyDevice);
    HOOK(GetDeviceQueue);
    HOOK(GetDeviceQueue2);
    HOOK(QueueSubmit);
    HOOK(CreateSwapchainKHR);
    HOOK(DestroySwapchainKHR);
    HOOK(AcquireNextImage2KHR);
    HOOK(QueuePresentKHR);

    std::lock_guard<std::mutex> lk(g_mtx);
    auto it = g_devices.find(dispatchKey(device));
    if (it != g_devices.end() && it->second.next)
        return it->second.next(device, name);
    return nullptr;
}

VKAPI_ATTR PFN_vkVoidFunction VKAPI_CALL BionicFG_GetInstanceProcAddr(
        VkInstance instance, const char* name) {
    HOOK(CreateInstance);
    HOOK(DestroyInstance);
    HOOK(EnumeratePhysicalDevices);
    HOOK(CreateDevice);
    HOOK(DestroyDevice);
    HOOK(GetDeviceQueue);
    HOOK(GetDeviceQueue2);
    HOOK(QueueSubmit);
    HOOK(CreateSwapchainKHR);
    HOOK(DestroySwapchainKHR);
    HOOK(AcquireNextImage2KHR);
    HOOK(QueuePresentKHR);
    if (std::strcmp(name, "vkGetDeviceProcAddr") == 0)
        return reinterpret_cast<PFN_vkVoidFunction>(BionicFG_GetDeviceProcAddr);
    if (std::strcmp(name, "vkGetInstanceProcAddr") == 0)
        return reinterpret_cast<PFN_vkVoidFunction>(BionicFG_GetInstanceProcAddr);

    if (instance == VK_NULL_HANDLE) return nullptr;
    std::lock_guard<std::mutex> lk(g_mtx);
    auto it = g_instances.find(dispatchKey(instance));
    if (it != g_instances.end() && it->second.next)
        return it->second.next(instance, name);
    return nullptr;
}

#undef HOOK

} // namespace bfg::layer

// ─── C-linkage exports (referenced by manifest JSON) ─────────────────────────

extern "C" BFG_EXPORT VKAPI_ATTR PFN_vkVoidFunction VKAPI_CALL
BionicFG_GetInstanceProcAddr(VkInstance inst, const char* name) {
    return bfg::layer::BionicFG_GetInstanceProcAddr(inst, name);
}

extern "C" BFG_EXPORT VKAPI_ATTR PFN_vkVoidFunction VKAPI_CALL
BionicFG_GetDeviceProcAddr(VkDevice dev, const char* name) {
    return bfg::layer::BionicFG_GetDeviceProcAddr(dev, name);
}
