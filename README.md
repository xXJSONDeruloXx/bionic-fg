# bionic-fg

Standalone Android/Bionic Vulkan frame generation layer.

## What is here

- `libbionic_fg.so`: native framegen runtime and Vulkan implicit layer
- `VK_LAYER_BIONIC_framegen`: intercepts swapchain presentation and injects generated frames
- Android Hardware Buffer sharing between the producer Vulkan device and the framegen device
- Embedded SPIR-V shader bundle with model 0 and model 1 selection
- Optional JNI bootstrap API under `io.github.bionicfg.BionicFGNative`

## Runtime environment

The Vulkan layer is enabled by environment variables:

```sh
BIONIC_FG_ENABLE=1
BIONIC_FG_MULTIPLIER=2   # 2..4
BIONIC_FG_FLOW_SCALE=0.6 # 0.2..1.0
BIONIC_FG_MODEL=0        # 0 or 1
VK_LAYER_PATH=/path/to/implicit_layer.d
```

Install layout expected by the manifest:

```text
.local/lib/libbionic_fg.so
.local/share/vulkan/implicit_layer.d/VkLayer_BIONIC_framegen.json
```

The manifest uses `../../../lib/libbionic_fg.so` relative to `implicit_layer.d`.

## Build

```sh
cmake -S . -B build/android-arm64 \
  -DANDROID_ABI=arm64-v8a \
  -DANDROID_PLATFORM=android-26 \
  -DCMAKE_TOOLCHAIN_FILE=$ANDROID_NDK_HOME/build/cmake/android.toolchain.cmake
cmake --build build/android-arm64 -j
```
