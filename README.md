# cpp_14x050

## Description

This is a **non-official** C++ port of University of Geneva's Deep Learning Course (14x050), which is originally written in Python and Pytorch. The course can be accessed here: https://fleuret.org/dlc/

This repository was made as a self-exercise for C++ and practicing the use of high-performance numerical libraries such as BLAS, LAPACK, and other hardware specific such as Apple Accelerate Framework.

The source code is typed with experimentation in mind, thus the implementations will not be 100% correct but follows how the original Python code was structred (and expected to be structured) along with the expected output. Consider that the code for the assignment are self contained within the .cpp files, thus no helper files were required.

Due to the nature of C++, some of the assignments might be skipped or modified. Please refer to the assignment files with caution.

**This repository does not accept any pull requests. Any requests would be ignored.**

## (Current) Dependencies and How to Use

**Currently, this repository assumes that the user have:**

1. A MacBook (required for Accelerate Framework) with at least 16GB of memory with clang compiler (bundled with xcode) for Apple Accelerate Framework.

**Compute library dependencies:**

- Apple Accelerate
- OpenBLAS
- LAPACK

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
    clang++ -std=c++14 14x050_exercise_1_1.cpp -o 14x050_exercise_1_1 -framework Accelerate -DACCELERATE_NEW_LAPACK
    ```
5.  Run the application:
    ```bash
    ./14x050_exercise_1_1.cpp
    ```
