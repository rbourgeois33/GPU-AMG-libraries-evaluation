# GPU-AMG Libraries Evaluation

This project evaluates several libraries implementing an AMG-preconditioned conjugate gradient solver on the GPU for solving linear systems derived from the TRUST platform CFD code ([TRUST platform](https://cea-trust-platform.github.io/)).

## Libraries Evaluated
- [Ginkgo](https://github.com/ginkgo-project/ginkgo)
- [AMGX](https://github.com/NVIDIA/AMGX)
- [amgcl](https://github.com/ddemidov/amgcl)
- [Hypre](https://github.com/hypre-space/hypre)

## 1. Clone the Project and Submodules
```bash
git clone https://github.com/rbourgeois33/GPU-AMG-libraries-evaluation
cd GPU-AMG-libraries-evaluation
git submodule update --init --recursive
```

## 2. Load Required Modules

### NVIDIA A5000 (Petra)
```bash
module load cuda/12.1.0 \
            cmake/3.18.4 \
            openmpi/gcc_11.2.0/4.1.4
```
### NVIDIA Ada 6000 (Ada)
```bash
module load cuda/12.4.0 openmpi/gcc_13.3.0/
```

### NVIDIA A100 (Jean Zay)
```bash
module load cuda/12.4.1 \
            openmpi/4.1.5 \
            cmake/3.18.0
```
### AMD Radeon PRO W7900
```bash
module load rocm/6.2.0 \
            openmpi/gcc_13.3.0/ \
            cmake/3.21.1
```

## 3. Build All Libraries
### Ginkgo
```bash
cd lib/ginkgo
mkdir build && cd build
mkdir install
```
#### **CMake Configuration**
- **On Jean Zay, Petra & Ada**
```bash
cmake -DGINKGO_BUILD_EXAMPLES=ON \
      -DGINKGO_BUILD_OMP=ON \
      -DGINKGO_BUILD_CUDA=ON \
      -DGINKGO_CUDA_ARCHITECTURES=Ampere \
      -DCMAKE_INSTALL_PREFIX=install/ ..
```
- **On AMD GPU Machine**
```bash
cmake -DGINKGO_BUILD_EXAMPLES=ON \
      -DGINKGO_BUILD_OMP=ON \
      -DGINKGO_BUILD_HIP=ON \
      -DCMAKE_INSTALL_PREFIX=install/ ..
```
#### **Build and Install**
- **On Petra & AMD GPU Machine**
```bash
make -j 48 install
```
- **On Jean Zay**  
```bash
srun -p compil -A pri@v100 -t 00:30:00 -c 24 --hint=nomultithread make -j24 install
```

### AMGX
```bash
cd lib/AMGX
mkdir build && cd build
cmake ..
make -j 16
```

### Hypre
```bash
cd lib/hypre
mkdir build && cd build
mkdir install
cmake -DHYPRE_ENABLE_OPENMP=ON -DHYPRE_ENABLE_CUDA=ON -DCMAKE_INSTALL_PREFIX=install/ ../src/
make -j12 install
```

## 4. Build the Main Program
```bash
mkdir build
cd build
cmake ..
make -j 12
```

## 5. Run the Programs

### **Ginkgo Evaluation**
```bash
./ginkgo_eval _backend_
```
where `_backend_` can be:
- `cuda`
- `omp`
- `reference`
- `hip`

### **AMGX Evaluation**
Generate the AMGX Matrix Market formatted input file (AMGX requires A and b in the same file):
```bash
cd data/
./AMGX_formatter.sh A.mtx rhs.mtx
```
Run AMGX solver:
```bash
./AMGX_eval ../config_AMGX/file.json
```
Optional: Add `write_w` to the command to write the output of the resolution in `.mtx` format.

### **AMGCL CUDA Evaluation**
```bash
./amgcl_cuda_eval ../data/aij_2592000.mtx ../data/rhs_2592000.mtx
```

### **Hypre Evaluation**
Run the Hypre solver with appropriate input files and configuration.
