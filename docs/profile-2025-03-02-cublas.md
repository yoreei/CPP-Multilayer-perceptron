# cuBlas CPU profile, 2025-03-02

Summary of two Visual Studio profiler sessions (`.diagsession`) that used to be committed to the repo root. They were removed to save ~3 MiB; this note keeps what was in them.

## `preallocOnes.diagsession`

Empty file (0 bytes), committed in `b59c2bd` ("fix dupRows reallocation bug"). It held no data. The name refers to the "after preallocating ones" step in [benchCublas.txt](../benchCublas.txt).

## `Report20250302-1932.diagsession`

Committed in `523e419` ("cuComputeBias; equal perf"), recorded 2025-03-02 19:32 with VS 2022 17.12, using the **CPU Usage** tool (sampling, about 1 ms interval, with stacks).

- **Profiled program:** `x64/Instrumentation/cuBlasProj.exe` (cuBlas project, Instrumentation configuration)
- **Machine:** the Ryzen 9 5950X (32 logical cores) + RTX 3070 Ti desktop from the README
- **Run length:** 17.4 s process lifetime (includes MNIST loading and CUDA init). This is the stage between "after fixing dupRows reallocation bug" (14.9 s training) and "after computeBias kernel" (~11-14 s) in `benchCublas.txt`
- **CPU load:** about 1 core the whole time (process CPU 2.6% average, 3.5% max of 32 cores). One thread took 17,301 of the 17,325 samples

### Where the host thread's CPU time went (17,325 samples)

| Module | Share | What it is |
|---|---|---|
| `nvcuda64.dll` | 32.3% | CUDA driver (user mode) |
| `ntoskrnl.exe` | 21.3% | Windows kernel |
| `nvlddmkm.sys` | 12.8% | NVIDIA kernel-mode driver |
| `cublasLt64_12.dll` | 8.7% | cuBLASLt |
| `cuBlasProj.exe` | 8.7% | the project's own host code |
| `msvcp140.dll` | 4.3% | C++ runtime |
| `dxgmms2.sys` + `dxgkrnl.sys` | 6.3% | DirectX graphics kernel (GPU scheduling) |
| `ntdll.dll` | 3.2% | |
| `cublas64_12.dll` | 0.8% | cuBLAS |
| other | ~1.6% | |

### Takeaway

The host side was launch/sync bound: about 75% of the CPU time was spent in the CUDA driver, the NVIDIA kernel driver and the Windows kernel, and only about 9% in the project's own code. The CPU thread mostly fed the GPU many small kernel launches and synchronizations. This matches the later optimizations in `benchCublas.txt` (fusing work into custom kernels such as `a2Sub1Hot` and `computeBias`), which cut the training time from 14.9 s towards ~10 s.

The function-level hot spots inside `cuBlasProj.exe` were not recovered: that needs the matching `.pdb`, which was never committed. The function-level numbers that were recorded by hand (`CUPMatrix::CUPMatrix` 4.36%, `MLP::Backward` 25.25%, ...) are already in `benchCublas.txt`.
