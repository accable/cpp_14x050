## Description

This is a **non-official** C++ port of University of Geneva's Deep Learning Course (14x050), which was originally written in Python and Pytorch. The course can be accessed here: https://fleuret.org/dlc/

This repository was made as a self-exercise for C++ and practicing the use of high-performance numerical libraries such as BLAS, LAPACK, and other hardware specific such as Apple Accelerate Framework, NVPL (Nvidia Performance Libraries), and Libtorch.

The source code is typed with experimentation in mind, thus the implementations will not be 100% correct but follows how the original Python code was structred (and expected to be structured) along with the expected output. Consider that the code for the assignment are self contained within the .cpp files, thus no helper files were required.

Due to the nature of C++, some of the assignments might be skipped or modified. Please refer to the assignment files with caution. Some of the functions from PyTorch, notably ```.eig()``` is replaced since LibTorch does not come with it.

**This repository does not accept any pull requests. Any requests would be ignored.**

## (Current) Dependencies and How to Use

**Currently, this repository assumes that the user have:**

1. A M-based (not tested on Intel) MacBook with at least 16GB of memory, clang++, and cmake compilers. Xcode is also necessary for Apple accelerate framework and clang)
2. Nvidia Grace Superchip w/ 240GB memory alongside ```nvhpc/24.7``` module (for ```nvc++``` compiler).

**Library dependencies:**

- Apple Accelerate (only for ```exercise_1_3.cpp```)
- OpenBLAS
- LAPACK
- NVPL (Using ```nvhpc/24.7```)
- Libtorch

**How to use:**

1.  Clone the repository:
    ```bash
    git clone https://github.com/accable/cpp_14x050.git
    ```

2.  Navigate to the project directory:
    ```bash
    cd cpp_14x050
    cd source
    ```

3.  To compile the .cpp files with Apple Accelerate Framework:
    ```bash
    clang++ -std=c++14 exercise_1_3.cpp -o exercise_1_3 -framework Accelerate -DACCELERATE_NEW_LAPACK
    ```

    To compile with NVHPC:
    ```bash
    nvc++ exercise_1_3_nvpl.cpp -o exercise_1_3_nvpl -lblas
    ```

4.  To compile the .cpp files with Libtorch support (requires CMake):
    ```bash
    mkdir build
    cd build
    cmake -DCMAKE_PREFIX_PATH=/absolute/path/to/libtorch ..
    cmake --build . --config Release -j 4  # Compiling with 4 cores
    ```
    Make sure to update CMakeLists.txt before compiling!

4.  Run the application:
    ```bash
    ./exercise_1_3
    ```
