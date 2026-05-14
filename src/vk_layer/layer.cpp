// Bionic FG Vulkan implicit layer
// Intercepts vkCreateSwapchainKHR + vkQueuePresentKHR to inject generated frames.
// Image sharing between producer device and framegen device is done via AHardwareBuffer.
// Enabled only when BIONIC_FG_ENABLE=1 is set in the environment.

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
#include <cstdlib>
#include <cstring>
#include <iterator>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <string>
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
    bool    enabled    = false;
    int     multiplier = 2;
    float   flowScale  = 0.6f;
    int     model      = 0;
};

static LayerConf readConf() {
    LayerConf c;
    const char* en = std::getenv("BIONIC_FG_ENABLE");
    if (!en || en[0] != '1') return c;
    c.enabled = true;
    if (auto* v = std::getenv("BIONIC_FG_MULTIPLIER"))
        c.multiplier = std::max(2, std::min(4, std::atoi(v)));
    if (auto* v = std::getenv("BIONIC_FG_FLOW_SCALE"))
        c.flowScale = std::max(0.2f, std::min(1.0f, (float)std::atof(v)));
    if (auto* v = std::getenv("BIONIC_FG_MODEL"))
        c.model = std::max(0, std::min(1, std::atoi(v)));
    return c;
}

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
    VkCommandBuffer genCmds[4] = {};    // one per possible generated frame
    VkFence         genFences[4] = {};
    VkSemaphore     acquireSems[4] = {};
    VkSemaphore     presentSems[4] = {};

    uint64_t frameCount = 0;
    bool     inPresent  = false;

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

// ─── Global state ─────────────────────────────────────────────────────────────

static std::mutex g_mtx;
static std::unordered_map<void*, InstanceData> g_instances;
static std::unordered_map<void*, DeviceData>   g_devices;
static std::unordered_map<VkPhysicalDevice, VkInstance> g_physInstances;
struct QueueData { void* deviceKey = nullptr; uint32_t family = VK_QUEUE_FAMILY_IGNORED; };
static std::unordered_map<VkQueue, QueueData> g_queues;
static std::unordered_map<VkSwapchainKHR, SwapState> g_swapchains;
static std::unordered_map<VkSwapchainKHR, void*>     g_swapDevice;

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

    // Modify swapchain: add transfer usage, request extra images for generated frames
    VkSwapchainCreateInfoKHR ci = *pCreateInfo;
    ci.imageUsage |= VK_IMAGE_USAGE_TRANSFER_SRC_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT;
    if (conf.enabled)
        ci.minImageCount = std::max(ci.minImageCount, static_cast<uint32_t>(conf.multiplier + 1));

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
        const int N = conf.multiplier - 1;

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
        for (int k = 0; k < N; ++k) {
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
        for (int k = 0; k < N; ++k) {
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
        cbai.commandBufferCount = 4;
        if (dd.allocCmdBufs(device, &cbai, st.genCmds) != VK_SUCCESS)
            throw std::runtime_error("failed to allocate generated-frame command buffers");

        VkFenceCreateInfo fci{};
        fci.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO;
        fci.flags = VK_FENCE_CREATE_SIGNALED_BIT;
        if (dd.createFence(device, &fci, nullptr, &st.copyFences[0]) != VK_SUCCESS ||
            dd.createFence(device, &fci, nullptr, &st.copyFences[1]) != VK_SUCCESS)
            throw std::runtime_error("failed to create copy fences");
        for (int k = 0; k < N && k < 4; ++k) {
            if (dd.createFence(device, &fci, nullptr, &st.genFences[k]) != VK_SUCCESS)
                throw std::runtime_error("failed to create generated-frame fence");
        }

        VkSemaphoreCreateInfo sci{};
        sci.sType = VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO;
        for (int k = 0; k < N && k < 4; ++k) {
            if (dd.createSemaphore(device, &sci, nullptr, &st.acquireSems[k]) != VK_SUCCESS ||
                dd.createSemaphore(device, &sci, nullptr, &st.presentSems[k]) != VK_SUCCESS)
                throw std::runtime_error("failed to create generated-frame semaphores");
        }

        // Create FramegenContext (uses its own device, imports AHBs). Avoid
        // recursively loading this implicit layer into Bionic-FG's private
        // Vulkan instance/device.
        bionic_fg::Config cfg;
        cfg.width      = W; cfg.height = H;
        cfg.multiplier = static_cast<uint32_t>(conf.multiplier);
        cfg.flowScale  = conf.flowScale;
        cfg.model      = static_cast<uint32_t>(conf.model);
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
        } disableLayerForInternalVulkan;
        st.fgCtx = bionic_fg::FramegenContext::create(
            st.prevAhb, st.currAhb, st.outAhbs, {W,H}, VK_FORMAT_R8G8B8A8_UNORM, cfg);

        if (!st.fgCtx) {
            BFG_LAYER_E("FramegenContext creation failed — layer will passthrough");
        } else {
            BFG_LAYER("SwapchainState ready: %ux%u mult=%d shaders=embedded", W, H, N+1);
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
    if (dd.acquireNextImage) {
        return dd.acquireNextImage(device, swapchain, 0, signalSemaphore, VK_NULL_HANDLE, imageIndex);
    }
    if (dd.acquireNextImage2) {
        VkAcquireNextImageInfoKHR info{};
        info.sType      = VK_STRUCTURE_TYPE_ACQUIRE_NEXT_IMAGE_INFO_KHR;
        info.swapchain  = swapchain;
        info.timeout    = 0;
        info.semaphore  = signalSemaphore;
        info.fence      = VK_NULL_HANDLE;
        info.deviceMask = 1;
        return dd.acquireNextImage2(device, &info, imageIndex);
    }
    return VK_ERROR_EXTENSION_NOT_PRESENT;
}

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

    const uint32_t fi = static_cast<uint32_t>(st.frameCount & 1u);
    VkDevice device = st.device;
    const int N = std::min(st.conf.multiplier - 1, 4);
    bool consumedOriginalWaits = false;

    if (!dd.resetCmdBuf || !dd.beginCmdBuf || !dd.endCmdBuf || !dd.queueSubmit || !dd.waitForFences ||
        !dd.resetFences || !dd.cmdBlitImage || !st.copyCmds[fi] || !st.copyFences[fi]) {
        BFG_LAYER_E("copy command infrastructure incomplete; passing through");
        return callNextPresent(queue, pPresentInfo);
    }

    // --- Step 1: Copy real swapchain image into current AHB input. The copy
    // waits on the application's original present semaphores, so the final real
    // present must not wait on those same binary semaphores again.
    if (dd.waitForFences(device, 1, &st.copyFences[fi], VK_TRUE, UINT64_MAX) != VK_SUCCESS ||
        dd.resetFences(device, 1, &st.copyFences[fi]) != VK_SUCCESS) {
        BFG_LAYER_E("copy fence wait/reset failed; passing through");
        return callNextPresent(queue, pPresentInfo);
    }

    dd.resetCmdBuf(st.copyCmds[fi], 0);
    VkCommandBufferBeginInfo cbi{};
    cbi.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
    cbi.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
    if (dd.beginCmdBuf(st.copyCmds[fi], &cbi) != VK_SUCCESS) {
        BFG_LAYER_E("begin copy command buffer failed; passing through");
        return callNextPresent(queue, pPresentInfo);
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
        return callNextPresent(queue, pPresentInfo);
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
    VkResult submitRes = dd.queueSubmit(queue, 1, &si, st.copyFences[fi]);
    if (submitRes != VK_SUCCESS) {
        BFG_LAYER_E("copy queue submit failed: %d; passing through", submitRes);
        return callNextPresent(queue, pPresentInfo);
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
        return dd.queuePresent(queue, &finalPresent);
    }

    // --- Step 2: Run framegen after we have both previous and current inputs.
    if (st.frameCount > 0) {
        st.fgCtx->present(st.prevAhb, st.currAhb);
        st.fgCtx->waitIdle();
        for (auto& out : st.outProducerImgs) {
            out.layout = VK_IMAGE_LAYOUT_GENERAL;
            out.externalOwner = true;
        }

        // --- Step 3: Inject generated frames. Acquire/write/present are kept
        // on distinct binary semaphores to avoid waiting and signaling the same
        // semaphore in one submit.
        for (int k = 0; k < N; ++k) {
            if (!st.acquireSems[k] || !st.presentSems[k] || !st.genCmds[k] || !st.genFences[k])
                continue;

            uint32_t genIdx = UINT32_MAX;
            VkResult acqRes = acquireGeneratedImage(dd, device, swapchain, st.acquireSems[k], &genIdx);
            if (acqRes != VK_SUCCESS && acqRes != VK_SUBOPTIMAL_KHR) {
                BFG_LAYER("generated frame acquire skipped: %d", acqRes);
                continue;
            }
            if (genIdx >= st.images.size()) {
                BFG_LAYER_E("generated image index %u out of range (%zu)", genIdx, st.images.size());
                continue;
            }

            dd.waitForFences(device, 1, &st.genFences[k], VK_TRUE, UINT64_MAX);
            dd.resetFences(device, 1, &st.genFences[k]);
            VkCommandBuffer genCmd = st.genCmds[k];
            dd.resetCmdBuf(genCmd, 0);
            if (dd.beginCmdBuf(genCmd, &cbi) != VK_SUCCESS) {
                BFG_LAYER_E("begin generated command buffer failed");
                continue;
            }

            ExtImage& out = st.outProducerImgs[size_t(k)];
            uint32_t outSrcFamily = out.externalOwner ? VK_QUEUE_FAMILY_EXTERNAL : VK_QUEUE_FAMILY_IGNORED;
            uint32_t outDstFamily = out.externalOwner ? st.copyQueueFamily : VK_QUEUE_FAMILY_IGNORED;
            layerImageBarrier(dd, genCmd, out.img,
                VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, 0,
                VK_PIPELINE_STAGE_TRANSFER_BIT, VK_ACCESS_TRANSFER_READ_BIT,
                out.layout, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
                outSrcFamily, outDstFamily);
            out.layout = VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL;
            out.externalOwner = false;

            layerImageBarrier(dd, genCmd, st.images[genIdx],
                VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, 0,
                VK_PIPELINE_STAGE_TRANSFER_BIT, VK_ACCESS_TRANSFER_WRITE_BIT,
                VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL);

            dd.cmdBlitImage(genCmd,
                out.img, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
                st.images[genIdx], VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
                1, &blit, VK_FILTER_LINEAR);

            layerImageBarrier(dd, genCmd, out.img,
                VK_PIPELINE_STAGE_TRANSFER_BIT, VK_ACCESS_TRANSFER_READ_BIT,
                VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT, 0,
                VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, VK_IMAGE_LAYOUT_GENERAL,
                st.copyQueueFamily, VK_QUEUE_FAMILY_EXTERNAL);
            out.layout = VK_IMAGE_LAYOUT_GENERAL;
            out.externalOwner = true;

            layerImageBarrier(dd, genCmd, st.images[genIdx],
                VK_PIPELINE_STAGE_TRANSFER_BIT, VK_ACCESS_TRANSFER_WRITE_BIT,
                VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT, 0,
                VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, VK_IMAGE_LAYOUT_PRESENT_SRC_KHR);

            if (dd.endCmdBuf(genCmd) != VK_SUCCESS) {
                BFG_LAYER_E("end generated command buffer failed");
                continue;
            }

            VkPipelineStageFlags waitMask = VK_PIPELINE_STAGE_TRANSFER_BIT;
            VkSubmitInfo gsi{};
            gsi.sType                = VK_STRUCTURE_TYPE_SUBMIT_INFO;
            gsi.waitSemaphoreCount   = 1;
            gsi.pWaitSemaphores      = &st.acquireSems[k];
            gsi.pWaitDstStageMask    = &waitMask;
            gsi.commandBufferCount   = 1;
            gsi.pCommandBuffers      = &genCmd;
            gsi.signalSemaphoreCount = 1;
            gsi.pSignalSemaphores    = &st.presentSems[k];
            VkResult genSubmit = dd.queueSubmit(queue, 1, &gsi, st.genFences[k]);
            if (genSubmit != VK_SUCCESS) {
                BFG_LAYER_E("generated submit failed: %d", genSubmit);
                continue;
            }

            VkPresentInfoKHR genPresent{};
            genPresent.sType              = VK_STRUCTURE_TYPE_PRESENT_INFO_KHR;
            genPresent.waitSemaphoreCount = 1;
            genPresent.pWaitSemaphores    = &st.presentSems[k];
            genPresent.swapchainCount     = 1;
            genPresent.pSwapchains        = &swapchain;
            genPresent.pImageIndices      = &genIdx;
            dd.queuePresent(queue, &genPresent);
        }
    }

    std::swap(st.prevAhb, st.currAhb);
    std::swap(st.prevProducerImg, st.currProducerImg);
    st.fgCtx->updateConfig(bionic_fg::Config{st.extent.width, st.extent.height,
        uint32_t(st.conf.multiplier), st.conf.flowScale, uint32_t(st.conf.model)});
    st.frameCount++;

    return dd.queuePresent(queue, &finalPresent);
#endif
}

// ─── GetProcAddr entry points ─────────────────────────────────────────────────

#define HOOK(fn) if (std::strcmp(name, "vk" #fn) == 0 || std::strcmp(name, #fn) == 0) return reinterpret_cast<PFN_vkVoidFunction>(BionicFG_##fn)

VKAPI_ATTR PFN_vkVoidFunction VKAPI_CALL BionicFG_GetDeviceProcAddr(
        VkDevice device, const char* name) {
    HOOK(DestroyDevice);
    HOOK(GetDeviceQueue);
    HOOK(GetDeviceQueue2);
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
