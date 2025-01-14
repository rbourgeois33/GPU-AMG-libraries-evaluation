# Ginkgo Evaluation for TRUST

This project evaluates the [Ginkgo library](https://github.com/ginkgo-project/ginkgo) for solving linear systems derived from the TRUST platform CFD code ([TRUST platform](https://cea-trust-platform.github.io/)). 

The libraries required for this project are located in the `data/` directory, and the main file is inspired by the examples provided by the Ginkgo project.

Gingko requires MPI and cmake.

---

## Compilation on Petra (CUDA and OpenMP backends)

### Load Necessary Modules
```bash
module load cuda/12.1.0
module load cmake/3.18.4
module load openmpi/gcc_11.2.0/4.1.4
```
### Compile Ginkgo as an External Library
```bash
cd ginkgo
mkdir build
cd build
mkdir install
cmake -DGINKGO_BUILD_EXAMPLES=ON \
      -DGINKGO_BUILD_OMP=ON \
      -DGINKGO_BUILD_CUDA=ON \
      -DGINKGO_CUDA_ARCHITECTURES=Ampere \
      -DGINKGO_MIXED_PRECISION=ON \
      -DCMAKE_INSTALL_PREFIX=install/ ..
make -j 48 install
```

### Build the Main Program
```bash
mkdir build
cd build
cmake ..
make -j 12
```

### Run the Program
```bash
./main _backend_
```

Replace `_backend_` with one of the following options:
- `cuda`
- `omp`
- `reference`
- `hip`

---

## On Jean zay (messy)
For building ginkgo (long):
```bash
module load cuda/12.4.1 openmpi/4.1.5 cmake/3.18.0 
srun -p compil -A pri@v100 -t 00:30:00 -c 24 --hint=nomultithread make -j24 install
 ```