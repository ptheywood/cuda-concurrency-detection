# CUDA Concurrency Detection


Streams enable concurrency between grids of kernels or data movement, subject to there being enough SMs to allow concurrency and cuda calls being issued in a way that agrees with the driver model (i.e. wddm causes issues).


This repo aims to look at how it is possible to detect this concurrency between streams as a form of test, to avoid regressions which prevent concurrency.

This is likely to be tricky / error prone, as it may rely on using CUDA event timers and some bold assumptions.

It may also fail on smaller devices, or larger problem sizes - so the device being tested needs to be considered (I.e. concurrnecy for a given problem size is not possible on all devices.)



## Building using Cmake

For Compute Capability 61 using 4 concurrent build threads 

```bash
mkdir -p build
cd build
cmake .. 
make -j 4 -DSMS=61
```

## Execution

```bash
cd build
./cuda-concurrency-detection
```