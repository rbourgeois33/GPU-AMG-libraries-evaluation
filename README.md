# GPU-AMG-librairies-evaluation


This project evaluates the following libraries that implement a AMG-preconditioned conjugate gradient solver on the gpu or solving linear systems derived from the TRUST platform CFD code ([TRUST platform](https://cea-trust-platform.github.io/)): 

-[Ginkgo library](https://github.com/ginkgo-project/ginkgo)
-[AMGX](https://github.com/NVIDIA/AMGX)

The compilation process will compile all the cpp files, there is one per library

## 0. Clone the project and its submodules
```bash
git clone git clone https://github.com/rbourgeois33/GPU-AMG-libraries-evaluation
cd GPU-AMG-libraries-evaluation
git submodule update --init --recursive
```

## 1. Load Required Modules

### Petra (NVIDIA A5000)

```bash
module load cuda/12.1.0 \
            cmake/3.18.4 \
            openmpi/gcc_11.2.0/4.1.4
```

### Jean Zay (NVIDIA A100)

```bash
module load cuda/12.4.1 \
            openmpi/4.1.5 \
            cmake/3.18.0
```

### AMD GPU Machine (AMD Radeon PRO W7900)

```bash
module load rocm/6.2.0 \
            openmpi/gcc_13.3.0/ \
            cmake/3.21.1
```

---
# eval_ginkgo

## 2. Compile Ginkgo as an External Library

### 1. Create Build Directories

```bash
cd lib/ginkgo
mkdir build
cd build
mkdir install
```

### 2. CMake Configuration

- **On Jean Zay & Petra**

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

### 3. Build and Install

- **On Petra & AMD GPU Machine**

  ```bash
  make -j 48 install
  ```

- **On Jean Zay**  
  Compiling Ginkgo is heavy, so you need to launch a compile job:
  
  ```bash
  srun -p compil -A pri@v100 -t 00:30:00 -c 24 --hint=nomultithread make -j24 install
  ```

---

## 3. Build the Main Program

After Ginkgo is installed, go back to your main program’s directory:

```bash
mkdir build
cd build
cmake ..
make -j 12
```

---

## 4. Run the Program

```bash
./eval_ginkgo _backend_
```

where `_backend_` can be:
- `cuda`
- `omp`
- `reference`
- `hip`

Choose the backend according to your machine’s capabilities and the Ginkgo libraries you built.

# AMGX_eval

## 2. Compile AMGX as an External Library

### 1. Create Build Directories and build

```bash
cd lib/AMGX
mkdir build ; cd build
cmake ..
make -j 16
```

## 3. Build the Main Program

After AMGX is installed, go back to your main program’s directory:

```bash
mkdir build
cd build
cmake ..
make -j 12
```

## 4. Run the Program
generate the AMGX Matrix Market formatted input file (AMGX can't read A and b separatly, they must be in the same file)
```bash
cd data/
./AMGX_formatter.sh A.mtx rhs.mtx
```
```bash
./AMGX_eval ../config_AMGX/file.json
```
optional: add `write_w`to the command line to write the output of the resolution in the mtx format.
