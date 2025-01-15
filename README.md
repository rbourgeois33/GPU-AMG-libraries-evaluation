# Linalg evaluation

This project evaluates the [Ginkgo library](https://github.com/ginkgo-project/ginkgo) for solving linear systems derived from the TRUST platform CFD code ([TRUST platform](https://cea-trust-platform.github.io/)). 

The libraries required for this project are located in the `data/` directory, and the main file is inspired by the examples provided by the Ginkgo project.

Gingko requires MPI and cmake.

## 0. Clone the project and its submodules
```bash
git clone https://gitlab.com/github.com/rbourgeois33/linalg_evaluation.git
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



