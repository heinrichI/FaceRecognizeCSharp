# GPU Status

## Current State
- **Runtime**: ONNX Runtime 1.27.0 (CPU only)
- **GPU**: NVIDIA GeForce RTX 4090 (available but not used)

## Why GPU is not working
FaceAiSharp.Bundle 0.6.35 depends on `Microsoft.ML.OnnxRuntime.Managed` >= 1.26.0.
The GPU packages have version constraints:

| Package | Max Version | CUDA Required | Compatible? |
|---------|-------------|---------------|-------------|
| Microsoft.ML.OnnxRuntime.Gpu | 1.27.0 | CUDA 13.x (cublasLt64_13.dll) | No — CUDA 12.4 installed |
| Microsoft.ML.OnnxRuntime.Gpu | 1.26.0 | CUDA 13.x | No — CUDA 12.4 installed |
| Microsoft.ML.OnnxRuntime.DirectML | 1.24.4 | DirectX 12 | No — managed wrapper 1.27.0 incompatible |

Installed CUDA: **12.4** (`C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.4`)
cuDNN: **v9.23** (`c:\Program Files\NVIDIA\CUDNN\v9.23\bin\12.9\x64`)

## How to enable GPU

### Option 1: Upgrade CUDA to 13.x
```bash
# Install CUDA Toolkit 13.x from NVIDIA
# Then in FaceRecognizer.cs, change:
var opts = new SessionOptions { GraphOptimizationLevel = GraphOptimizationLevel.ORT_ENABLE_ALL };
opts.AppendExecutionProvider_CUDA();
```
And replace OnnxRuntime with GPU version:
```bash
dotnet remove package Microsoft.ML.OnnxRuntime
dotnet add package Microsoft.ML.OnnxRuntime.Gpu --version 1.27.0
```

### Option 2: Downgrade FaceAiSharp to match CUDA 12.x
Use an older FaceAiSharp version that depends on OnnxRuntime.Managed < 1.26.0,
then install `Microsoft.ML.OnnxRuntime.Gpu` 1.21.x which supports CUDA 12.x.
Check NuGet for compatible versions.

### Option 3: Use DirectML when a compatible version is released
Monitor https://www.nuget.org/packages/Microsoft.ML.OnnxRuntime.DirectML for version 1.26+.

## Performance (CPU mode)
- ArcFace int8: ~10-50ms per face on modern CPU
- SCRFD 2.5G: ~5-20ms per face detection
- For <1000 images, CPU mode is perfectly adequate
