#include "framegen_context.hpp"
#include "logging.hpp"

#include <cstring>
#include <sstream>
#include <stdexcept>
#include <algorithm>
#include <array>
#include <cstdio>
#include <string>
#include <unordered_map>

namespace bionic_fg {

// ─── Pass ────────────────────────────────────────────────────────────────────

Pass::Pass(const vk::Device& dev,
           VkDescriptorPool descPool,
           int shaderIdx,
           const std::vector<vk::DescriptorBinding>& bindings) {
    if (shaderIdx < 0 || static_cast<size_t>(shaderIdx) >= embedded::kShaderRegistry.size())
        throw std::runtime_error("Pass: shader index out of range");
    const auto& blob = embedded::kShaderRegistry[static_cast<size_t>(shaderIdx)];
    if (!embedded::IsValidSpirv(blob))
        throw std::runtime_error(std::string("Pass: invalid SPIR-V for ") + blob.name);
    layout_   = vk::DescriptorSetLayout(dev, bindings);
    descSet_  = vk::DescriptorSet(dev, descPool, layout_.handle());
    pipeline_ = vk::ComputePipeline(dev, blob.data, blob.size, layout_.handle());
    BFG_LOGI("Pass: loaded %s", blob.name);
}
void Pass::destroy(const vk::Device& dev) {
    pipeline_.destroy(dev);
    layout_.destroy(dev);
}
void Pass::bindUBO(const vk::Device& dev, uint32_t b, const vk::Buffer& buf) {
    descSet_.bindUBO(dev, b, buf);
}
void Pass::bindSampled(const vk::Device& dev, uint32_t b, const vk::Image& img,
                       const vk::Sampler& sampler) {
    descSet_.bindCombinedImageSampler(dev, b, img, sampler,
        img.external() ? VK_IMAGE_LAYOUT_GENERAL : VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);
}
void Pass::bindSampledGeneral(const vk::Device& dev, uint32_t b, const vk::Image& img,
                              const vk::Sampler& sampler) {
    descSet_.bindCombinedImageSampler(dev, b, img, sampler, VK_IMAGE_LAYOUT_GENERAL);
}
void Pass::bindStorage(const vk::Device& dev, uint32_t b, const vk::Image& img) {
    descSet_.bindStorageImage(dev, b, img);
}
void Pass::dispatch(VkCommandBuffer cmd, uint32_t gx, uint32_t gy) const {
    pipeline_.bind(cmd);
    pipeline_.bindDescriptorSet(cmd, descSet_.handle());
    pipeline_.dispatch(cmd, gx, gy, 1);
}

// ─── Helpers ─────────────────────────────────────────────────────────────────

static void computeBarrier(VkCommandBuffer cmd) {
    VkMemoryBarrier mb{};
    mb.sType         = VK_STRUCTURE_TYPE_MEMORY_BARRIER;
    mb.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
    mb.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;
    vkCmdPipelineBarrier(cmd,
        VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
        VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
        0, 1, &mb, 0, nullptr, 0, nullptr);
}

static void toStorage(VkCommandBuffer cmd, vk::Image& img) {
    if (img.external() || img.layout == VK_IMAGE_LAYOUT_GENERAL) return;
    vk::imageBarrier(cmd, img.handle(),
        img.layout == VK_IMAGE_LAYOUT_UNDEFINED
            ? VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT : VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
        0,
        VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_ACCESS_SHADER_WRITE_BIT,
        img.layout, VK_IMAGE_LAYOUT_GENERAL);
    img.layout = VK_IMAGE_LAYOUT_GENERAL;
}

static void toShaderRead(VkCommandBuffer cmd, vk::Image& img) {
    if (img.external()) return;
    if (img.layout == VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL) return;
    vk::imageBarrier(cmd, img.handle(),
        VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_ACCESS_SHADER_WRITE_BIT,
        VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_ACCESS_SHADER_READ_BIT,
        img.layout, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);
    img.layout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
}

static VkDescriptorPool makeDescPool(const vk::Device& dev, uint32_t maxSets) {
    VkDescriptorPoolSize sizes[3]{};
    sizes[0].type = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;            sizes[0].descriptorCount = maxSets * 4;
    sizes[1].type = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;    sizes[1].descriptorCount = maxSets * 16;
    sizes[2].type = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;             sizes[2].descriptorCount = maxSets * 16;
    VkDescriptorPoolCreateInfo pci{};
    pci.sType         = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
    pci.flags         = VK_DESCRIPTOR_POOL_CREATE_FREE_DESCRIPTOR_SET_BIT;
    pci.maxSets       = maxSets;
    pci.poolSizeCount = 3;
    pci.pPoolSizes    = sizes;
    VkDescriptorPool pool = VK_NULL_HANDLE;
    vkCreateDescriptorPool(dev.handle(), &pci, nullptr, &pool);
    return pool;
}

static void clearImageWhite(const vk::Device& dev, const vk::CommandPool& pool, vk::Image& img) {
    vk::CommandBuffer cb(dev, pool);
    cb.begin();
    vk::imageBarrier(cb.handle(), img.handle(),
        VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, 0,
        VK_PIPELINE_STAGE_TRANSFER_BIT, VK_ACCESS_TRANSFER_WRITE_BIT,
        VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL);
    img.layout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
    VkClearColorValue c{}; c.float32[0]=c.float32[1]=c.float32[2]=c.float32[3]=1.0f;
    VkImageSubresourceRange r{}; r.aspectMask=VK_IMAGE_ASPECT_COLOR_BIT; r.levelCount=1; r.layerCount=1;
    vkCmdClearColorImage(cb.handle(), img.handle(), VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, &c, 1, &r);
    vk::imageBarrier(cb.handle(), img.handle(),
        VK_PIPELINE_STAGE_TRANSFER_BIT, VK_ACCESS_TRANSFER_WRITE_BIT,
        VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_ACCESS_SHADER_READ_BIT,
        VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, VK_IMAGE_LAYOUT_GENERAL);
    img.layout = VK_IMAGE_LAYOUT_GENERAL;
    cb.end(); cb.submitAndWait(dev);
    cb.destroy(dev, pool);
}

// Build a single-sampled + single-storage pass (most feature passes use this).
static Pass make1Tex1Img(const vk::Device& dev, VkDescriptorPool pool,
                         int shaderIdx, uint32_t sampledBinding, uint32_t storageBinding,
                         const vk::Image& src, const vk::Sampler& sampler, const vk::Image& dst) {
    std::vector<vk::DescriptorBinding> binds = {
        {sampledBinding,  VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 1},
        {storageBinding,  VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 1},
    };
    Pass p(dev, pool, shaderIdx, binds);
    p.bindSampled(dev, sampledBinding, src, sampler);
    p.bindStorage(dev, storageBinding, dst);
    return p;
}

// Build a UBO + 2-sampled + 2-storage pass (OF refinement passes).
static Pass make2Tex2Img_UBO(const vk::Device& dev, VkDescriptorPool pool,
                              int shaderIdx,
                              const vk::Buffer& ubo,
                              const vk::Image& inA, const vk::Image& inB,
                              const vk::Sampler& sampler,
                              vk::Image& outFwd, vk::Image& outBwd) {
    std::vector<vk::DescriptorBinding> binds = {
        {0,  VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, 1},
        {32, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 1},
        {33, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 1},
        {48, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 1},
        {49, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 1},
    };
    Pass p(dev, pool, shaderIdx, binds);
    p.bindUBO    (dev, 0,  ubo);
    p.bindSampled(dev, 32, inA, sampler);
    p.bindSampled(dev, 33, inB, sampler);
    p.bindStorage(dev, 48, outFwd);
    p.bindStorage(dev, 49, outBwd);
    return p;
}

static Pass makeModel1Pass(const vk::Device& dev,
                           VkDescriptorPool pool,
                           int shaderIdx,
                           const vk::Buffer* ubo,
                           const std::vector<const vk::Image*>& sampled,
                           const vk::Sampler& sampler,
                           const std::vector<vk::Image*>& storage) {
    std::vector<vk::DescriptorBinding> binds;
    binds.reserve((ubo ? 1u : 0u) + sampled.size() + storage.size());
    if (ubo) binds.push_back({0, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, 1});
    for (uint32_t i = 0; i < sampled.size(); ++i)
        binds.push_back({32u + i, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 1});
    for (uint32_t i = 0; i < storage.size(); ++i)
        binds.push_back({48u + i, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 1});

    Pass p(dev, pool, shaderIdx, binds);
    if (ubo) p.bindUBO(dev, 0, *ubo);
    for (uint32_t i = 0; i < sampled.size(); ++i)
        p.bindSampledGeneral(dev, 32u + i, *sampled[i], sampler);
    for (uint32_t i = 0; i < storage.size(); ++i)
        p.bindStorage(dev, 48u + i, *storage[i]);
    return p;
}

// ─── FramegenContext::create ─────────────────────────────────────────────────

#ifdef __ANDROID__
std::unique_ptr<FramegenContext> FramegenContext::create(
        AHardwareBuffer* prevAhb, AHardwareBuffer* currAhb,
        const std::vector<AHardwareBuffer*>& outputAhbs,
        VkExtent2D extent, VkFormat format, const Config& cfg) {
    if (!prevAhb || !currAhb || outputAhbs.empty()) {
        BFG_LOGE("FramegenContext::create: null AHB or empty outputs");
        return nullptr;
    }
    auto ctx = std::make_unique<FramegenContext>();
    ctx->cfg_ = cfg; ctx->cfg_.sanitize();
    ctx->extent_ = extent; ctx->format_ = format;
    ctx->prevAhbPtr_ = prevAhb;
    ctx->currAhbPtr_ = currAhb;
    const uint32_t W = extent.width, H = extent.height;
    const uint32_t W2 = std::max(1u, W >> 1), H2 = std::max(1u, H >> 1);
    const uint32_t outputs = ctx->cfg_.multiplier - 1;

    try {
        const bool useModel1 = ctx->cfg_.model == 1;
        if (useModel1) {
            BFG_LOGI("model=1 requested: using traced 100-dispatch graph (shader_03, 30..53, final 04), not model-0 hybrid fallback");
        }

        ctx->device_        = vk::Device::create();
        ctx->cmdPool_       = vk::CommandPool(ctx->device_);
        ctx->linearSampler_ = vk::Sampler(ctx->device_, VK_FILTER_LINEAR,
                                           VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_BORDER);
        ctx->nearestSampler_= vk::Sampler(ctx->device_, VK_FILTER_NEAREST,
                                           VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE);
        ctx->descPool_ = makeDescPool(ctx->device_, 128);

        // ── AHB-backed images ──────────────────────────────────────────────
        vk::ImageInfo ahbInfo;
        ahbInfo.extent = extent; ahbInfo.format = format;
        ahbInfo.usage  = VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_SAMPLED_BIT
                       | VK_IMAGE_USAGE_TRANSFER_SRC_BIT;
        ctx->prevFrame_ = vk::Image(ctx->device_, ahbInfo, prevAhb);
        ctx->currFrame_ = vk::Image(ctx->device_, ahbInfo, currAhb);
        ctx->outputImages_.reserve(outputAhbs.size());
        for (auto* ahb : outputAhbs)
            ctx->outputImages_.emplace_back(ctx->device_, ahbInfo, ahb);

        // ── Pyramid images (R8_UNORM, 6 levels) ───────────────────────────
        ctx->pyramidA_.reserve(6); ctx->pyramidB_.reserve(6);
        for (int i = 0; i < 6; ++i) {
            vk::ImageInfo pi;
            pi.extent = {std::max(1u,W>>i), std::max(1u,H>>i)};
            pi.format = VK_FORMAT_R8_UNORM;
            pi.usage  = VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_SAMPLED_BIT;
            ctx->pyramidA_.emplace_back(ctx->device_, pi);
            ctx->pyramidB_.emplace_back(ctx->device_, pi);
        }

        // ── Feature maps (W/2 × H/2, RGBA16F) ────────────────────────────
        vk::ImageInfo fi;
        fi.extent = {W2, H2};
        fi.format = VK_FORMAT_R16G16B16A16_SFLOAT;
        fi.usage  = VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_SAMPLED_BIT;
        ctx->featA_     = vk::Image(ctx->device_, fi);
        ctx->featB_     = vk::Image(ctx->device_, fi);
        ctx->featChanA_ = vk::Image(ctx->device_, fi);
        ctx->featChanB_ = vk::Image(ctx->device_, fi);
        ctx->featChanC_ = vk::Image(ctx->device_, fi);

        // ── Flow images (W × H, RGBA16F) ─────────────────────────────────
        vk::ImageInfo flow;
        flow.extent = extent;
        flow.format = VK_FORMAT_R16G16B16A16_SFLOAT;
        flow.usage  = VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_SAMPLED_BIT;
        ctx->flowFwd_        = vk::Image(ctx->device_, flow);
        ctx->flowBwd_        = vk::Image(ctx->device_, flow);
        ctx->flowRefinedFwd_ = vk::Image(ctx->device_, flow);
        ctx->flowRefinedBwd_ = vk::Image(ctx->device_, flow);
        ctx->flowMerged_     = vk::Image(ctx->device_, flow);
        ctx->flowAggregated_ = vk::Image(ctx->device_, flow);
        ctx->flowExpA_       = vk::Image(ctx->device_, flow);
        ctx->flowExpB_       = vk::Image(ctx->device_, flow);
        ctx->flowPyramid_.reserve(6);
        for (int i = 0; i < 6; ++i) {
            vk::ImageInfo fpi = flow;
            fpi.extent = {std::max(1u, W >> i), std::max(1u, H >> i)};
            ctx->flowPyramid_.emplace_back(ctx->device_, fpi);
        }

        // Confidence placeholder (RGBA8 all-ones)
        vk::ImageInfo ci;
        ci.extent = extent; ci.format = format;
        ci.usage  = VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_SAMPLED_BIT
                  | VK_IMAGE_USAGE_TRANSFER_DST_BIT;
        ctx->confidence_    = vk::Image(ctx->device_, ci);
        ctx->warpedPrev_     = vk::Image(ctx->device_, ci);
        ctx->warpedCurr_     = vk::Image(ctx->device_, ci);
        ctx->confidenceMap_  = vk::Image(ctx->device_, ci);
        clearImageWhite(ctx->device_, ctx->cmdPool_, ctx->confidence_);

        // ── UBOs ──────────────────────────────────────────────────────────
        PyramidUBO pyubo; pyubo.scale=2; pyubo.aspect=W/std::max(1u,H); pyubo.pad0=0; pyubo.pad1=0;
        ctx->uboPyramid_ = vk::Buffer(ctx->device_, sizeof(PyramidUBO),
                                       VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT, &pyubo);
        FlowUBO fubo; fubo.flowScale=cfg.flowScale; fubo.pad0=fubo.pad1=fubo.pad2=0;
        ctx->uboFlow_ = vk::Buffer(ctx->device_, sizeof(FlowUBO),
                                    VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT, &fubo);
        ctx->uboSynth_.reserve(outputs);
        for (uint32_t k = 0; k < outputs; ++k) {
            SynthUBO s;
            s.flowScale = cfg.flowScale;
            s.alpha     = float(k+1) / float(cfg.multiplier);
            s.epsilon   = 1e-5f;
            ctx->uboSynth_.emplace_back(ctx->device_, sizeof(SynthUBO),
                                         VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT, &s);
        }

        // ── Model 1: runtime-confirmed graph ───────────────────────
        // The reference implementation owns more persistent/history resources
        // than an implicit layer can see directly. The compute side still needs
        // to match the observed model-1 dispatch/resource graph: 99 internal
        // dispatches (shader_03, shader_30..53) plus final shader_04 per
        // generated output. Descriptor labels below follow the traced
        // descriptor-edge map used for this graph reconstruction.
        if (useModel1) {
            struct Ratio { uint32_t num; uint32_t den; };
            auto ratio = [](uint32_t num, uint32_t den) { return Ratio{num, den}; };
            auto extentFor = [&](Ratio r) -> VkExtent2D {
                const uint32_t ew = std::max(1u, static_cast<uint32_t>((uint64_t(W) * r.num + r.den - 1u) / r.den));
                const uint32_t eh = std::max(1u, static_cast<uint32_t>((uint64_t(H) * r.num + r.den - 1u) / r.den));
                return {ew, eh};
            };
            auto groupsFor = [&](Ratio r) -> VkExtent2D {
                VkExtent2D e = extentFor(r);
                return {std::max(1u, (e.width + 15u) / 16u),
                        std::max(1u, (e.height + 15u) / 16u)};
            };
            auto dLabel = [](int dispatch, int binding) -> std::string {
                char buf[16];
                std::snprintf(buf, sizeof(buf), "d%03d.b%d", dispatch, binding);
                return std::string(buf);
            };
            auto extLabel = [](int ext) -> std::string {
                char buf[16];
                std::snprintf(buf, sizeof(buf), "ext%d", ext);
                return std::string(buf);
            };
            auto dLabels = [&](int dispatch, int firstBinding, int count) {
                std::vector<std::string> out;
                out.reserve(static_cast<size_t>(count));
                for (int i = 0; i < count; ++i) out.push_back(dLabel(dispatch, firstBinding + i));
                return out;
            };
            auto extLabels = [&](int firstExt, int count) {
                std::vector<std::string> out;
                out.reserve(static_cast<size_t>(count));
                for (int i = 0; i < count; ++i) out.push_back(extLabel(firstExt + i));
                return out;
            };
            auto append = [](std::vector<std::string>& dst, std::vector<std::string> src) {
                dst.insert(dst.end(), src.begin(), src.end());
            };

            auto storageFormatForShader = [](int shaderIdx) -> VkFormat {
                switch (shaderIdx) {
                    case 3:
                    case 38:
                        return VK_FORMAT_R8_UNORM;
                    case 43:
                    case 48:
                    case 53:
                        return VK_FORMAT_R16G16B16A16_SFLOAT;
                    default:
                        return VK_FORMAT_R8G8B8A8_UNORM;
                }
            };
            auto uboForShader = [&](int shaderIdx) -> const vk::Buffer* {
                switch (shaderIdx) {
                    case 3:
                        return &ctx->uboPyramid_;
                    case 38:
                    case 39:
                    case 43:
                    case 44:
                    case 48:
                    case 49:
                        return &ctx->uboFlow_;
                    default:
                        return nullptr;
                }
            };

            ctx->model1Resources_.reserve(320);
            std::unordered_map<std::string, size_t> resourceIndex;
            auto allocateResource = [&](const std::string& label, VkFormat fmt, VkExtent2D e) -> vk::Image& {
                auto found = resourceIndex.find(label);
                if (found != resourceIndex.end()) return ctx->model1Resources_[found->second];
                vk::ImageInfo info;
                info.extent = e;
                info.format = fmt;
                info.usage  = VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_SAMPLED_BIT;
                const size_t idx = ctx->model1Resources_.size();
                ctx->model1Resources_.emplace_back(ctx->device_, info);
                resourceIndex.emplace(label, idx);
                return ctx->model1Resources_.back();
            };
            auto resourceForD = [&](const std::string& label) -> vk::Image& {
                return ctx->model1Resources_[resourceIndex.at(label)];
            };

            // Most extN labels are persistent/history resources from the traced
            // native graph. In the implicit-layer adaptation we feed matching
            // current-frame graph products instead; ext1/ext35 remain the real
            // curr/prev frame pair used by shader_03 and final shader_04.
            auto mappedExternalLabel = [&](int ext) -> std::string {
                if (ext >= 2 && ext <= 5)   return dLabel(23, 48 + (ext - 2));
                if (ext >= 6 && ext <= 9)   return dLabel(23, 48 + (ext - 6));
                if (ext >= 10 && ext <= 13) return dLabel(29, 48 + (ext - 10));
                if (ext == 14)              return dLabel(34, 53);
                if (ext >= 15 && ext <= 18) return dLabel(28, 48 + (ext - 15));
                if (ext >= 19 && ext <= 22) return dLabel(27, 48 + (ext - 19));
                if (ext >= 23 && ext <= 26) return dLabel(26, 48 + (ext - 23));
                if (ext >= 27 && ext <= 30) return dLabel(25, 48 + (ext - 27));
                if (ext >= 31 && ext <= 34) return dLabel(24, 48 + (ext - 31));
                return dLabel(23, 48);
            };
            auto resolveSample = [&](const std::string& label) -> const vk::Image* {
                if (label.rfind("ext", 0) == 0) {
                    const int ext = std::stoi(label.substr(3));
                    if (ext == 1)  return &ctx->currFrame_;
                    if (ext == 35) return &ctx->prevFrame_;
                    return &resourceForD(mappedExternalLabel(ext));
                }
                return &resourceForD(label);
            };

            auto addPass = [&](int shaderIdx,
                               Ratio passRatio,
                               const std::vector<std::string>& sampledLabels,
                               const std::vector<std::string>& storageLabels,
                               const std::vector<Ratio>& storageRatios) {
                const VkFormat fmt = storageFormatForShader(shaderIdx);
                std::vector<vk::Image*> storage;
                storage.reserve(storageLabels.size());
                for (size_t i = 0; i < storageLabels.size(); ++i) {
                    const Ratio r = storageRatios.empty() ? passRatio : storageRatios[i];
                    storage.push_back(&allocateResource(storageLabels[i], fmt, extentFor(r)));
                }

                std::vector<const vk::Image*> sampled;
                sampled.reserve(sampledLabels.size());
                for (const auto& label : sampledLabels) sampled.push_back(resolveSample(label));

                ctx->model1GraphPasses_.push_back(
                    makeModel1Pass(ctx->device_, ctx->descPool_, shaderIdx, uboForShader(shaderIdx),
                                   sampled, ctx->linearSampler_, storage));
                ctx->model1GraphDispatch_.push_back(groupsFor(passRatio));
            };

            const std::vector<Ratio> pyramidOut = {
                ratio(1,5), ratio(1,10), ratio(1,20), ratio(1,40),
                ratio(1,80), ratio(1,160), ratio(1,320),
            };
            const std::array<Ratio, 7> expandRatios = {{
                ratio(2,5), ratio(1,5), ratio(1,10), ratio(1,20),
                ratio(1,40), ratio(1,80), ratio(1,160),
            }};
            const std::array<Ratio, 7> smallRatios = {{
                ratio(1,5), ratio(1,10), ratio(1,20), ratio(1,40),
                ratio(1,80), ratio(1,160), ratio(1,320),
            }};

            // 0x1a9ed4: shader_03, shader_30 x7, shader_31 x7,
            // shader_32 x7, shader_33 x7.
            addPass(3, ratio(1,5), {extLabel(1)}, dLabels(1, 48, 7), pyramidOut);
            for (int i = 0; i < 7; ++i)
                addPass(30, expandRatios[size_t(i)], {dLabel(1, 48 + i)}, dLabels(2 + i, 48, 2), {});
            for (int i = 0; i < 7; ++i)
                addPass(31, expandRatios[size_t(i)], dLabels(2 + i, 48, 2), dLabels(9 + i, 48, 2), {});
            for (int i = 0; i < 7; ++i)
                addPass(32, smallRatios[size_t(i)], dLabels(9 + i, 48, 2), dLabels(16 + i, 48, 4), {});
            for (int i = 0; i < 7; ++i)
                addPass(33, smallRatios[size_t(i)], dLabels(16 + i, 48, 4), dLabels(23 + i, 48, 4), {});

            // 0x1b004c: shader_34..38.
            {
                auto sampled = extLabels(2, 8);
                append(sampled, dLabels(23, 48, 4));
                addPass(34, ratio(1,5), sampled, dLabels(30, 48, 2), {});
            }
            addPass(35, ratio(1,5), dLabels(30, 48, 2), dLabels(31, 48, 2), {});
            addPass(36, ratio(1,5), dLabels(31, 48, 2), dLabels(32, 48, 2), {});
            addPass(37, ratio(1,5), dLabels(32, 48, 2), dLabels(33, 48, 2), {});
            addPass(38, ratio(1,10), dLabels(33, 48, 2), dLabels(34, 48, 6), {
                ratio(1,10), ratio(1,20), ratio(1,40), ratio(1,80), ratio(1,160), ratio(1,320)
            });

            auto addFivePassRound = [&](Ratio r, int extStart, int srcD,
                                        const std::string& carry, const std::string& aux,
                                        int firstDispatch) {
                auto sampled39 = extLabels(extStart, 4);
                append(sampled39, dLabels(srcD, 48, 4));
                sampled39.push_back(carry);
                addPass(39, r, sampled39, dLabels(firstDispatch, 48, 3), {});
                addPass(40, r, dLabels(firstDispatch, 48, 3), dLabels(firstDispatch + 1, 48, 4), {});
                addPass(41, r, dLabels(firstDispatch + 1, 48, 4), dLabels(firstDispatch + 2, 48, 4), {});
                addPass(42, r, dLabels(firstDispatch + 2, 48, 4), dLabels(firstDispatch + 3, 48, 4), {});
                auto sampled43 = dLabels(firstDispatch + 3, 48, 4);
                sampled43.push_back(carry);
                sampled43.push_back(aux);
                addPass(43, r, sampled43, dLabels(firstDispatch + 4, 48, 1), {});
            };

            // 0x1b0708: three 1x1 rounds plus one 2x2 round using shader_39..43.
            addFivePassRound(ratio(1,320), 10, 29, extLabel(14),      dLabel(34, 53), 35);
            addFivePassRound(ratio(1,160), 15, 28, dLabel(39, 48),   dLabel(34, 53), 40);
            addFivePassRound(ratio(1,80),  19, 27, dLabel(44, 48),   dLabel(34, 52), 45);
            addFivePassRound(ratio(1,40),  23, 26, dLabel(49, 48),   dLabel(34, 51), 50);

            auto addFifteenPassRound = [&](Ratio r, int extStart, int srcD,
                                           const std::string& carry,
                                           const std::string& aux,
                                           const std::string& shader49Extra,
                                           const std::string& shader53Extra,
                                           int firstDispatch) {
                addFivePassRound(r, extStart, srcD, carry, aux, firstDispatch);

                auto sampled44 = extLabels(extStart, 4);
                append(sampled44, dLabels(srcD, 48, 4));
                sampled44.push_back(carry);
                addPass(44, r, sampled44, dLabels(firstDispatch + 5, 48, 3), {});
                addPass(45, r, dLabels(firstDispatch + 5, 48, 3), dLabels(firstDispatch + 6, 48, 4), {});
                addPass(46, r, dLabels(firstDispatch + 6, 48, 4), dLabels(firstDispatch + 7, 48, 4), {});
                addPass(47, r, dLabels(firstDispatch + 7, 48, 4), dLabels(firstDispatch + 8, 48, 4), {});
                auto sampled48 = dLabels(firstDispatch + 8, 48, 4);
                sampled48.push_back(carry);
                sampled48.push_back(aux);
                addPass(48, r, sampled48, dLabels(firstDispatch + 9, 48, 1), {});

                auto sampled49 = extLabels(extStart, 4);
                append(sampled49, dLabels(srcD, 48, 4));
                sampled49.push_back(carry);
                sampled49.push_back(shader49Extra);
                addPass(49, r, sampled49, dLabels(firstDispatch + 10, 48, 2), {});
                addPass(50, r, dLabels(firstDispatch + 10, 48, 2), dLabels(firstDispatch + 11, 48, 2), {});
                addPass(51, r, dLabels(firstDispatch + 11, 48, 2), dLabels(firstDispatch + 12, 48, 2), {});
                addPass(52, r, dLabels(firstDispatch + 12, 48, 2), dLabels(firstDispatch + 13, 48, 2), {});
                auto sampled53 = dLabels(firstDispatch + 13, 48, 2);
                sampled53.push_back(shader53Extra);
                addPass(53, r, sampled53, dLabels(firstDispatch + 14, 48, 1), {});
            };

            addFifteenPassRound(ratio(1,20), 27, 25, dLabel(54, 48), dLabel(34, 50),
                                extLabel(14), extLabel(14), 55);
            addFifteenPassRound(ratio(1,10), 31, 24, dLabel(59, 48), dLabel(34, 49),
                                dLabel(64, 48), dLabel(69, 48), 70);
            addFifteenPassRound(ratio(1,5), 6, 23, dLabel(74, 48), dLabel(34, 48),
                                dLabel(79, 48), dLabel(84, 48), 85);

            // Final full-resolution shader_04. The traced native path then
            // performs a barrier/copy/barrier in 0x1b1b10; as a layer we bind
            // the storage output directly to the generated AHB image and let
            // layer.cpp blit it into swapchain images.
            ctx->model1FinalPassStart_ = ctx->model1GraphPasses_.size();
            for (uint32_t k = 0; k < outputs; ++k) {
                std::vector<const vk::Image*> sampled = {
                    &ctx->prevFrame_,
                    &ctx->currFrame_,
                    &resourceForD(dLabel(89, 48)),
                    &resourceForD(dLabel(94, 48)),
                    &resourceForD(dLabel(99, 48)),
                };
                std::vector<vk::Image*> storage = {&ctx->outputImages_[k]};
                ctx->model1GraphPasses_.push_back(
                    makeModel1Pass(ctx->device_, ctx->descPool_, 4, &ctx->uboSynth_[k],
                                   sampled, ctx->linearSampler_, storage));
                ctx->model1GraphDispatch_.push_back(groupsFor(ratio(1,1)));
            }

            BFG_LOGI("model=1 graph: %zu passes, %zu resources, finalStart=%zu",
                     ctx->model1GraphPasses_.size(), ctx->model1Resources_.size(),
                     ctx->model1FinalPassStart_);
        }

        // ── Model-0 stages: skip entirely for model=1 (has its own graph above) ──
        if (!useModel1) {

        // ── Stage 1: Pyramid (separate Pass per frame to avoid descriptor aliasing) ──
        {
            std::vector<vk::DescriptorBinding> binds = {
                {0,  VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, 1},
                {32, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 1},
                {48, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 1},
                {49, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 1},
                {50, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 1},
                {51, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 1},
                {52, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 1},
                {53, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 1},
            };
            // pyramidA_ ← currFrame (frame t)
            ctx->passPyramidA_ = Pass(ctx->device_, ctx->descPool_, 3, binds);
            ctx->passPyramidA_.bindUBO(ctx->device_, 0, ctx->uboPyramid_);
            ctx->passPyramidA_.bindSampled(ctx->device_, 32, ctx->currFrame_, ctx->linearSampler_);
            for (int i=0;i<6;++i) ctx->passPyramidA_.bindStorage(ctx->device_, uint32_t(48+i), ctx->pyramidA_[size_t(i)]);
            // pyramidB_ ← prevFrame (frame t-1)
            ctx->passPyramidB_ = Pass(ctx->device_, ctx->descPool_, 3, binds);
            ctx->passPyramidB_.bindUBO(ctx->device_, 0, ctx->uboPyramid_);
            ctx->passPyramidB_.bindSampled(ctx->device_, 32, ctx->prevFrame_, ctx->linearSampler_);
            for (int i=0;i<6;++i) ctx->passPyramidB_.bindStorage(ctx->device_, uint32_t(48+i), ctx->pyramidB_[size_t(i)]);
        }

        // ── Stage 2: Feature extraction ──────────────────────────────────────
        // shader_05: curr → featA  (half-res)
        ctx->passFeatA_     = make1Tex1Img(ctx->device_, ctx->descPool_, 5,  32, 48,
                                            ctx->currFrame_, ctx->linearSampler_, ctx->featA_);
        // shader_06: curr → featB
        ctx->passFeatB_     = make1Tex1Img(ctx->device_, ctx->descPool_, 6,  32, 48,
                                            ctx->currFrame_, ctx->linearSampler_, ctx->featB_);
        // shader_26,27,28: featA → channels A/B/C
        ctx->passFeatChanA_ = make1Tex1Img(ctx->device_, ctx->descPool_, 26, 32, 48,
                                            ctx->featA_, ctx->linearSampler_, ctx->featChanA_);
        ctx->passFeatChanB_ = make1Tex1Img(ctx->device_, ctx->descPool_, 27, 32, 48,
                                            ctx->featA_, ctx->linearSampler_, ctx->featChanB_);
        ctx->passFeatChanC_ = make1Tex1Img(ctx->device_, ctx->descPool_, 28, 32, 48,
                                            ctx->featA_, ctx->linearSampler_, ctx->featChanC_);

        // ── Stage 3: Coarse OF (shader_09) ──────────────────────────────────
        // bindings: 32-34 = prev pyramid (pyramidB_), 35-37 = curr pyramid (pyramidA_)
        {
            std::vector<vk::DescriptorBinding> binds = {
                {32,VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,1},
                {33,VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,1},
                {34,VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,1},
                {35,VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,1},
                {36,VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,1},
                {37,VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,1},
                {48,VK_DESCRIPTOR_TYPE_STORAGE_IMAGE,1},
                {49,VK_DESCRIPTOR_TYPE_STORAGE_IMAGE,1},
            };
            ctx->passCoarseOF_ = Pass(ctx->device_, ctx->descPool_, 9, binds);
            for (int i=0;i<3;++i) {
                ctx->passCoarseOF_.bindSampled(ctx->device_, uint32_t(32+i),
                    ctx->pyramidB_[size_t(i)], ctx->linearSampler_); // prev
                ctx->passCoarseOF_.bindSampled(ctx->device_, uint32_t(35+i),
                    ctx->pyramidA_[size_t(i)], ctx->linearSampler_); // curr
            }
            ctx->passCoarseOF_.bindStorage(ctx->device_, 48, ctx->flowFwd_);
            ctx->passCoarseOF_.bindStorage(ctx->device_, 49, ctx->flowBwd_);
        }

        // ── Stage 4: OF refinement chain ────────────────────────────────────
        // shaders 08, 10, 11, 12: UBO + featChanA(prev+curr) → refined flow
        // For our purposes, featChanA acts as both prev and curr feature inputs
        // (in a real pipeline the prev features come from the prior frame's extraction)
        ctx->passOFRefine0_    = make2Tex2Img_UBO(ctx->device_, ctx->descPool_, 8,
            ctx->uboFlow_, ctx->featChanA_, ctx->featChanB_, ctx->linearSampler_,
            ctx->flowRefinedFwd_, ctx->flowRefinedBwd_);
        ctx->passOFRefine1_    = make2Tex2Img_UBO(ctx->device_, ctx->descPool_, 10,
            ctx->uboFlow_, ctx->flowRefinedFwd_, ctx->featChanA_, ctx->linearSampler_,
            ctx->flowFwd_, ctx->flowBwd_);
        ctx->passOFRefine2_    = make2Tex2Img_UBO(ctx->device_, ctx->descPool_, 11,
            ctx->uboFlow_, ctx->flowFwd_, ctx->featChanB_, ctx->linearSampler_,
            ctx->flowRefinedFwd_, ctx->flowRefinedBwd_);
        ctx->passOFRefine3_    = make2Tex2Img_UBO(ctx->device_, ctx->descPool_, 12,
            ctx->uboFlow_, ctx->flowRefinedFwd_, ctx->featChanC_, ctx->linearSampler_,
            ctx->flowFwd_, ctx->flowBwd_);
        ctx->passOFRefineLarge_= make2Tex2Img_UBO(ctx->device_, ctx->descPool_, 17,
            ctx->uboFlow_, ctx->flowFwd_, ctx->flowBwd_, ctx->linearSampler_,
            ctx->flowRefinedFwd_, ctx->flowRefinedBwd_);

        // ── Stage 5: Flow post-processing ───────────────────────────────────
        {
            std::vector<vk::DescriptorBinding> binds = {
                {32,VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,1},
                {33,VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,1},
                {48,VK_DESCRIPTOR_TYPE_STORAGE_IMAGE,1},
            };
            ctx->passFlowMerge_ = Pass(ctx->device_, ctx->descPool_, 29, binds);
            ctx->passFlowMerge_.bindSampled(ctx->device_, 32, ctx->flowRefinedFwd_, ctx->linearSampler_);
            ctx->passFlowMerge_.bindSampled(ctx->device_, 33, ctx->flowRefinedBwd_, ctx->linearSampler_);
            ctx->passFlowMerge_.bindStorage(ctx->device_, 48, ctx->flowMerged_);
        }
        {
            std::vector<vk::DescriptorBinding> binds = {
                {0,  VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER,1},
                {32,VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,1},
                {33,VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,1},
                {48,VK_DESCRIPTOR_TYPE_STORAGE_IMAGE,1},
                {49,VK_DESCRIPTOR_TYPE_STORAGE_IMAGE,1},
                {50,VK_DESCRIPTOR_TYPE_STORAGE_IMAGE,1},
                {51,VK_DESCRIPTOR_TYPE_STORAGE_IMAGE,1},
                {52,VK_DESCRIPTOR_TYPE_STORAGE_IMAGE,1},
                {53,VK_DESCRIPTOR_TYPE_STORAGE_IMAGE,1},
            };
            ctx->passFlowPyramid_ = Pass(ctx->device_, ctx->descPool_, 13, binds);
            ctx->passFlowPyramid_.bindUBO(ctx->device_, 0, ctx->uboFlow_);
            ctx->passFlowPyramid_.bindSampled(ctx->device_, 32, ctx->flowMerged_, ctx->linearSampler_);
            ctx->passFlowPyramid_.bindSampled(ctx->device_, 33, ctx->featA_, ctx->linearSampler_);
            for (int i=0;i<6;++i)
                ctx->passFlowPyramid_.bindStorage(ctx->device_, uint32_t(48+i), ctx->flowPyramid_[size_t(i)]);
        }
        {
            std::vector<vk::DescriptorBinding> binds = {
                {0,  VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER,1},
                {32,VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,1},
                {33,VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,1},
                {34,VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,1},
                {35,VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,1},
                {36,VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,1},
                {37,VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,1},
                {48,VK_DESCRIPTOR_TYPE_STORAGE_IMAGE,1},
            };
            ctx->passFlowAggregate_ = Pass(ctx->device_, ctx->descPool_, 25, binds);
            ctx->passFlowAggregate_.bindUBO(ctx->device_, 0, ctx->uboFlow_);
            for (int i=0;i<6;++i)
                ctx->passFlowAggregate_.bindSampled(ctx->device_, uint32_t(32+i), ctx->flowPyramid_[size_t(i)], ctx->linearSampler_);
            ctx->passFlowAggregate_.bindStorage(ctx->device_, 48, ctx->flowAggregated_);
        }
        {
            std::vector<vk::DescriptorBinding> binds = {
                {32,VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,1},
                {48,VK_DESCRIPTOR_TYPE_STORAGE_IMAGE,1},
                {49,VK_DESCRIPTOR_TYPE_STORAGE_IMAGE,1},
            };
            ctx->passFlowExpand_ = Pass(ctx->device_, ctx->descPool_, 30, binds);
            ctx->passFlowExpand_.bindSampled(ctx->device_, 32, ctx->flowAggregated_, ctx->linearSampler_);
            ctx->passFlowExpand_.bindStorage(ctx->device_, 48, ctx->flowExpA_);
            ctx->passFlowExpand_.bindStorage(ctx->device_, 49, ctx->flowExpB_);
        }

        // ── Stage 6: Confidence warp + synthesis ─────────────────────────────
        ctx->passWarpBlend_.reserve(outputs);
        ctx->passSynth_.reserve(outputs);
        for (uint32_t k = 0; k < outputs; ++k) {
            // shader_14/20 generate the per-alpha occlusion/confidence map used
            // by shader_04 in the model-0 path. Model-1 never reaches this
            // block; it has its own traced graph above.
            std::vector<vk::DescriptorBinding> wbBinds = {
                {0,  VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, 1},
                {32, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 1},
                {33, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 1},
                {34, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 1},
                {35, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 1},
                {36, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 1},
                {48, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 1},
                {49, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 1},
                {50, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 1},
            };
            const int warpShader = (k & 1u) ? 20 : 14;
            ctx->passWarpBlend_.emplace_back(ctx->device_, ctx->descPool_, warpShader, wbBinds);
            auto& pw = ctx->passWarpBlend_.back();
            pw.bindUBO    (ctx->device_, 0,  ctx->uboSynth_[k]);
            pw.bindSampled(ctx->device_, 32, ctx->prevFrame_,    ctx->linearSampler_);
            pw.bindSampled(ctx->device_, 33, ctx->currFrame_,    ctx->linearSampler_);
            pw.bindSampled(ctx->device_, 34, ctx->flowExpA_,     ctx->linearSampler_);
            pw.bindSampled(ctx->device_, 35, ctx->flowExpB_,     ctx->linearSampler_);
            pw.bindSampled(ctx->device_, 36, ctx->confidence_,   ctx->linearSampler_);
            pw.bindStorage(ctx->device_, 48, ctx->warpedPrev_);
            pw.bindStorage(ctx->device_, 49, ctx->warpedCurr_);
            pw.bindStorage(ctx->device_, 50, ctx->confidenceMap_);

            std::vector<vk::DescriptorBinding> binds = {
                {0,  VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, 1},
                {32, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 1},
                {33, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 1},
                {34, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 1},
                {35, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 1},
                {36, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 1},
                {48, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 1},
            };
            ctx->passSynth_.emplace_back(ctx->device_, ctx->descPool_, 4, binds);
            auto& ps = ctx->passSynth_.back();
            ps.bindUBO    (ctx->device_, 0,  ctx->uboSynth_[k]);
            ps.bindSampled(ctx->device_, 32, ctx->prevFrame_,      ctx->linearSampler_);
            ps.bindSampled(ctx->device_, 33, ctx->currFrame_,      ctx->linearSampler_);
            ps.bindSampled(ctx->device_, 34, ctx->flowExpA_,       ctx->linearSampler_);
            ps.bindSampled(ctx->device_, 35, ctx->flowExpB_,       ctx->linearSampler_);
            ps.bindSampled(ctx->device_, 36, ctx->confidenceMap_,  ctx->linearSampler_);
            ps.bindStorage(ctx->device_, 48, ctx->outputImages_[k]);
        }

        } // end !useModel1 model-0 stages

        // ── Frame ring ────────────────────────────────────────────────────────
        for (auto& f : ctx->frames_) {
            f.cmd   = vk::CommandBuffer(ctx->device_, ctx->cmdPool_);
            f.fence = vk::Fence(ctx->device_, true);
        }

        BFG_LOGI("FramegenContext ready: %ux%u mult=%u model=%u graph=%s",
                  W, H, cfg.multiplier, cfg.model,
                  useModel1 ? "model1-table" : "model0-full-of-chain");
        return ctx;
    } catch (const vk::VkError& e) {
        BFG_LOGE("FramegenContext::create VkError %d: %s", e.code, e.msg.c_str());
        ctx->destroy();
        return nullptr;
    } catch (const std::exception& e) {
        BFG_LOGE("FramegenContext::create exception: %s", e.what());
        ctx->destroy();
        return nullptr;
    }
}

void FramegenContext::rebindFrameInputs() {
    if (cfg_.model == 1) {
        if (!model1GraphPasses_.empty())
            model1GraphPasses_[0].bindSampledGeneral(device_, 32, currFrame_, linearSampler_); // shader_03 ext1
        for (size_t i = model1FinalPassStart_; i < model1GraphPasses_.size(); ++i) {
            model1GraphPasses_[i].bindSampledGeneral(device_, 32, prevFrame_, linearSampler_); // shader_04 ext35
            model1GraphPasses_[i].bindSampledGeneral(device_, 33, currFrame_, linearSampler_); // shader_04 ext1
        }
        return;
    }

    passPyramidA_.bindSampled(device_, 32, currFrame_, linearSampler_);
    passPyramidB_.bindSampled(device_, 32, prevFrame_, linearSampler_);
    passFeatA_.bindSampled(device_, 32, currFrame_, linearSampler_);
    passFeatB_.bindSampled(device_, 32, currFrame_, linearSampler_);
    for (auto& pw : passWarpBlend_) {
        pw.bindSampled(device_, 32, prevFrame_, linearSampler_);
        pw.bindSampled(device_, 33, currFrame_, linearSampler_);
    }
    for (auto& ps : passSynth_) {
        ps.bindSampled(device_, 32, prevFrame_, linearSampler_);
        ps.bindSampled(device_, 33, currFrame_, linearSampler_);
    }
}

void FramegenContext::present(AHardwareBuffer* newPrev, AHardwareBuffer* newCurr) {
    if (newPrev && newCurr && (newPrev != prevAhbPtr_ || newCurr != currAhbPtr_)) {
        if (newPrev == currAhbPtr_ && newCurr == prevAhbPtr_) {
            std::swap(prevFrame_, currFrame_);
            std::swap(prevAhbPtr_, currAhbPtr_);
            rebindFrameInputs();
        } else {
            BFG_LOGW("FramegenContext::present: unexpected AHB input order; using existing descriptors");
        }
    }

    const uint32_t W  = extent_.width,  H  = extent_.height;
    const uint32_t W2 = std::max(1u,W>>1), H2 = std::max(1u,H>>1);
    const uint32_t fi = frameIdx_ & 1u;
    auto& fr = frames_[fi];
    fr.fence.wait(device_); fr.fence.reset(device_);

    vkResetCommandBuffer(fr.cmd.handle(), 0);
    fr.cmd.begin();
    VkCommandBuffer cmd = fr.cmd.handle();

    // Acquire AHB inputs from external
    if (prevFrame_.external()) vk::acquireFromExternal(cmd, prevFrame_, device_.computeFamily(), VK_ACCESS_SHADER_READ_BIT);
    if (currFrame_.external()) vk::acquireFromExternal(cmd, currFrame_, device_.computeFamily(), VK_ACCESS_SHADER_READ_BIT);

    if (cfg_.model == 1) {
        // Runtime-confirmed steady-state Model-1 order:
        // 0x1a9ed4 -> 0x1afe28(no compute) -> 0x1b004c -> 0x1b0708
        // -> 0x1b1b10(copy/barrier). We emit the internal dispatch graph plus
        // the active final shader_04 passes directly into the provisioned AHB
        // outputs required by the current multiplier.
        const size_t activeOutputs = std::min<size_t>(
            outputImages_.size(),
            cfg_.multiplier > 1 ? static_cast<size_t>(cfg_.multiplier - 1) : size_t{0});

        for (auto& img : model1Resources_) toStorage(cmd, img);
        for (size_t i = 0; i < activeOutputs; ++i) {
            auto& out = outputImages_[i];
            if (out.external())
                vk::acquireFromExternal(cmd, out, device_.computeFamily(), VK_ACCESS_SHADER_WRITE_BIT);
            toStorage(cmd, out);
        }

        const size_t passCount = std::min(model1GraphPasses_.size(), model1GraphDispatch_.size());
        for (size_t i = 0; i < passCount; ++i) {
            const VkExtent2D g = model1GraphDispatch_[i];
            model1GraphPasses_[i].dispatch(cmd, g.width, g.height);
            computeBarrier(cmd);
        }

        for (size_t i = 0; i < activeOutputs; ++i) {
            auto& out = outputImages_[i];
            if (out.external())
                vk::releaseToExternal(cmd, out, device_.computeFamily(), VK_ACCESS_SHADER_WRITE_BIT);
        }
        if (prevFrame_.external()) vk::releaseToExternal(cmd, prevFrame_, device_.computeFamily(), VK_ACCESS_SHADER_READ_BIT);
        if (currFrame_.external()) vk::releaseToExternal(cmd, currFrame_, device_.computeFamily(), VK_ACCESS_SHADER_READ_BIT);

        fr.cmd.end();
        fr.cmd.submit(device_, fr.fence.handle());
        frameIdx_++;
        return;
    }

    // ── Stage 1: Pyramid ─────────────────────────────────────────────────────
    for (int i=0;i<6;++i) { toStorage(cmd,pyramidA_[size_t(i)]); toStorage(cmd,pyramidB_[size_t(i)]); }
    passPyramidA_.dispatch(cmd, (W+15)/16, (H+15)/16);
    computeBarrier(cmd);
    passPyramidB_.dispatch(cmd, (W+15)/16, (H+15)/16);
    computeBarrier(cmd);
    for (int i=0;i<6;++i) { toShaderRead(cmd,pyramidA_[size_t(i)]); toShaderRead(cmd,pyramidB_[size_t(i)]); }

    // ── Stage 2: Feature extraction ─────────────────────────────────────────
    toStorage(cmd,featA_); toStorage(cmd,featB_);
    toStorage(cmd,featChanA_); toStorage(cmd,featChanB_); toStorage(cmd,featChanC_);
    passFeatA_.dispatch(cmd, (W2+15)/16, (H2+15)/16);    computeBarrier(cmd);
    passFeatB_.dispatch(cmd, (W2+15)/16, (H2+15)/16);    computeBarrier(cmd);
    toShaderRead(cmd,featA_); toShaderRead(cmd,featB_);
    passFeatChanA_.dispatch(cmd, (W2+15)/16, (H2+15)/16); computeBarrier(cmd);
    passFeatChanB_.dispatch(cmd, (W2+15)/16, (H2+15)/16); computeBarrier(cmd);
    passFeatChanC_.dispatch(cmd, (W2+15)/16, (H2+15)/16); computeBarrier(cmd);
    toShaderRead(cmd,featChanA_); toShaderRead(cmd,featChanB_); toShaderRead(cmd,featChanC_);

    // ── Stage 3: Coarse OF ───────────────────────────────────────────────────
    toStorage(cmd,flowFwd_); toStorage(cmd,flowBwd_);
    passCoarseOF_.dispatch(cmd, (W+15)/16, (H+15)/16);
    computeBarrier(cmd);
    toShaderRead(cmd,flowFwd_); toShaderRead(cmd,flowBwd_);

    // ── Stage 4: OF refinement chain ────────────────────────────────────────
    toStorage(cmd,flowRefinedFwd_); toStorage(cmd,flowRefinedBwd_);
    passOFRefine0_.dispatch(cmd, (W+15)/16, (H+15)/16);    computeBarrier(cmd);
    toShaderRead(cmd,flowRefinedFwd_); toShaderRead(cmd,flowRefinedBwd_);
    toStorage(cmd,flowFwd_); toStorage(cmd,flowBwd_);
    passOFRefine1_.dispatch(cmd, (W+15)/16, (H+15)/16);    computeBarrier(cmd);
    toShaderRead(cmd,flowFwd_); toShaderRead(cmd,flowBwd_);
    toStorage(cmd,flowRefinedFwd_); toStorage(cmd,flowRefinedBwd_);
    passOFRefine2_.dispatch(cmd, (W+15)/16, (H+15)/16);    computeBarrier(cmd);
    toShaderRead(cmd,flowRefinedFwd_); toShaderRead(cmd,flowRefinedBwd_);
    toStorage(cmd,flowFwd_); toStorage(cmd,flowBwd_);
    passOFRefine3_.dispatch(cmd, (W+15)/16, (H+15)/16);    computeBarrier(cmd);
    toShaderRead(cmd,flowFwd_); toShaderRead(cmd,flowBwd_);
    toStorage(cmd,flowRefinedFwd_); toStorage(cmd,flowRefinedBwd_);
    passOFRefineLarge_.dispatch(cmd, (W+15)/16, (H+15)/16); computeBarrier(cmd);
    toShaderRead(cmd,flowRefinedFwd_); toShaderRead(cmd,flowRefinedBwd_);

    // ── Stage 5: Flow post-processing ────────────────────────────────────────
    toStorage(cmd,flowMerged_);
    passFlowMerge_.dispatch(cmd, (W+15)/16, (H+15)/16);   computeBarrier(cmd);
    toShaderRead(cmd,flowMerged_);

    for (auto& img : flowPyramid_) toStorage(cmd, img);
    passFlowPyramid_.dispatch(cmd, (W+15)/16, (H+15)/16); computeBarrier(cmd);
    for (auto& img : flowPyramid_) toShaderRead(cmd, img);

    toStorage(cmd,flowAggregated_);
    passFlowAggregate_.dispatch(cmd, (W+15)/16, (H+15)/16); computeBarrier(cmd);
    toShaderRead(cmd,flowAggregated_);

    toStorage(cmd,flowExpA_); toStorage(cmd,flowExpB_);
    passFlowExpand_.dispatch(cmd, (W+15)/16, (H+15)/16);  computeBarrier(cmd);
    toShaderRead(cmd,flowExpA_); toShaderRead(cmd,flowExpB_);

    // ── Stage 6: Confidence warp + synthesis ────────────────────────────────
    for (size_t k=0; k<passSynth_.size(); ++k) {
        toStorage(cmd, warpedPrev_);
        toStorage(cmd, warpedCurr_);
        toStorage(cmd, confidenceMap_);
        passWarpBlend_[k].dispatch(cmd, (W+15)/16, (H+15)/16);
        computeBarrier(cmd);
        toShaderRead(cmd, confidenceMap_);

        if (outputImages_[k].external())
            vk::acquireFromExternal(cmd, outputImages_[k], device_.computeFamily(), VK_ACCESS_SHADER_WRITE_BIT);
        toStorage(cmd, outputImages_[k]);
        passSynth_[k].dispatch(cmd, (W+15)/16, (H+15)/16);
        computeBarrier(cmd);
        if (outputImages_[k].external())
            vk::releaseToExternal(cmd, outputImages_[k], device_.computeFamily(), VK_ACCESS_SHADER_WRITE_BIT);
    }

    // Release inputs to external
    if (prevFrame_.external()) vk::releaseToExternal(cmd, prevFrame_, device_.computeFamily(), VK_ACCESS_SHADER_READ_BIT);
    if (currFrame_.external()) vk::releaseToExternal(cmd, currFrame_, device_.computeFamily(), VK_ACCESS_SHADER_READ_BIT);

    fr.cmd.end();
    fr.cmd.submit(device_, fr.fence.handle());
    frameIdx_++;
}
#endif // __ANDROID__

void FramegenContext::updateConfig(const Config& cfg) {
    cfg_ = cfg; cfg_.sanitize();
    for (size_t k=0;k<uboSynth_.size();++k) {
        SynthUBO s; s.flowScale=cfg_.flowScale;
        s.alpha=float(k+1)/float(cfg_.multiplier); s.epsilon=1e-5f;
        uboSynth_[k].write(device_,&s,sizeof(s));
    }
    FlowUBO f; f.flowScale=cfg_.flowScale; f.pad0=f.pad1=f.pad2=0;
    uboFlow_.write(device_,&f,sizeof(f));
}

void FramegenContext::waitIdle() {
    if (device_.valid()) vkQueueWaitIdle(device_.computeQueue());
}

void FramegenContext::destroy() {
    waitIdle();
    for (auto& f:frames_) { f.cmd.destroy(device_,cmdPool_); f.fence.destroy(device_); }
    auto destroyPass=[&](Pass& p){ p.destroy(device_); };
    destroyPass(passPyramidA_); destroyPass(passPyramidB_);
    destroyPass(passFeatA_); destroyPass(passFeatB_);
    destroyPass(passFeatChanA_); destroyPass(passFeatChanB_); destroyPass(passFeatChanC_);
    destroyPass(passCoarseOF_);
    destroyPass(passOFRefine0_); destroyPass(passOFRefine1_);
    destroyPass(passOFRefine2_); destroyPass(passOFRefine3_);
    destroyPass(passOFRefineLarge_);
    destroyPass(passFlowMerge_); destroyPass(passFlowPyramid_);
    destroyPass(passFlowAggregate_); destroyPass(passFlowExpand_);
    for (auto& p:passWarpBlend_) p.destroy(device_); passWarpBlend_.clear();
    for (auto& p:passSynth_) p.destroy(device_); passSynth_.clear();
    for (auto& p:model1GraphPasses_) p.destroy(device_); model1GraphPasses_.clear();
    model1GraphDispatch_.clear();
    model1FinalPassStart_ = 0;
    uboPyramid_.destroy(device_); uboFlow_.destroy(device_);
    for (auto& b:uboSynth_) b.destroy(device_); uboSynth_.clear();
    // Images
    prevFrame_.destroy(device_); currFrame_.destroy(device_);
    for (auto& i:outputImages_) i.destroy(device_); outputImages_.clear();
    for (auto& i:pyramidA_) i.destroy(device_); for (auto& i:pyramidB_) i.destroy(device_);
    featA_.destroy(device_); featB_.destroy(device_);
    featChanA_.destroy(device_); featChanB_.destroy(device_); featChanC_.destroy(device_);
    flowFwd_.destroy(device_); flowBwd_.destroy(device_);
    flowRefinedFwd_.destroy(device_); flowRefinedBwd_.destroy(device_);
    flowMerged_.destroy(device_); flowAggregated_.destroy(device_);
    for (auto& i:flowPyramid_) i.destroy(device_); flowPyramid_.clear();
    flowExpA_.destroy(device_); flowExpB_.destroy(device_);
    confidence_.destroy(device_); warpedPrev_.destroy(device_);
    warpedCurr_.destroy(device_); confidenceMap_.destroy(device_);
    for (auto& i:model1Resources_) i.destroy(device_); model1Resources_.clear();
    if (descPool_) vkDestroyDescriptorPool(device_.handle(), descPool_, nullptr);
    descPool_ = VK_NULL_HANDLE;
    linearSampler_.destroy(device_); nearestSampler_.destroy(device_);
    cmdPool_.destroy(device_); device_.destroy();
}

std::string FramegenContext::describe() const {
    std::ostringstream o;
    o << "FramegenContext{" << extent_.width << "x" << extent_.height
      << " mult=" << cfg_.multiplier << " flowScale=" << cfg_.flowScale
      << " model=" << cfg_.model << " valid=" << (valid()?"true":"false") << "}";
    return o.str();
}

} // namespace bionic_fg
