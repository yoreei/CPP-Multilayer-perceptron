# CUDA version CPU profile, 2025-03-02

This is a summary of two Visual Studio profiler sessions (`.diagsession` files) that used to be in the repository root. I removed the files to keep the repository small.

## preallocOnes.diagsession

An empty file, committed in `33f03e8` ("fix dupRows reallocation bug"). It contained no data. The name refers to the "after preallocating ones" step in [benchCublas.txt](benchCublas.txt).

## Report20250302-1932.diagsession

Committed in `d21fcec` ("cuComputeBias; equal perf"). It was recorded on 2025-03-02 at 19:32 with the CPU Usage tool of Visual Studio 2022 17.12, which samples the call stack about once per millisecond.

The profiled program was `x64/Instrumentation/cuBlasProj.exe`, the CUDA version in the Instrumentation configuration, on the Ryzen 9 5950X desktop (32 logical cores, RTX 3070 Ti). The process ran for 17.4 s, including loading MNIST and initializing CUDA. In `benchCublas.txt` this falls between "after fixing dupRows reallocation bug" (14.9 s of training) and "after computeBias kernel" (11 to 14 s).

The process used about one core the whole time: 2.6% of the CPU on average and 3.5% at most, out of 32 cores. One thread took 17,301 of the 17,325 samples.

### Where the CPU time went

| Module | Share of samples | What it is |
|---|---|---|
| `nvcuda64.dll` | 32.3% | CUDA driver (user mode) |
| `ntoskrnl.exe` | 21.3% | Windows kernel |
| `nvlddmkm.sys` | 12.8% | NVIDIA kernel-mode driver |
| `cublasLt64_12.dll` | 8.7% | cuBLASLt |
| `cuBlasProj.exe` | 8.7% | my own host code |
| `dxgmms2.sys` and `dxgkrnl.sys` | 6.3% | DirectX graphics kernel (GPU scheduling) |
| `msvcp140.dll` | 4.3% | C++ runtime |
| `ntdll.dll` | 3.2% | |
| `cublas64_12.dll` | 0.8% | cuBLAS |
| other | 1.6% | |

About 75% of the CPU time went to the CUDA driver, the NVIDIA kernel driver and the Windows kernel, and only about 9% to my own code. The CPU was mostly busy launching many small GPU operations and waiting for them. The later changes in `benchCublas.txt` that moved more work into custom kernels (`a2Sub1Hot`, `computeBias`) brought training from 14.9 s down to about 10 s.

The profile can't show which functions inside `cuBlasProj.exe` were slow, because that needs the matching `.pdb` file, which was never committed. The function-level numbers I wrote down by hand at the time (`CUPMatrix::CUPMatrix` 4.36%, `MLP::Backward` 25.25% and others) are in `benchCublas.txt`.
