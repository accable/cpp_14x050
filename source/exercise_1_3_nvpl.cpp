#include <iostream>
#include <chrono>
#include "nvpl_blas_cblas.h"
#include <random>
#include <cstdlib>

// Optimized for matmul only. Nothing here is reusable.
int main(){

    // Time operations
    using std::chrono::high_resolution_clock;
    using std::chrono::duration_cast;
    using std::chrono::milliseconds;  // Apparently in order for one to get proper speed, you need to use ms

    int n = 5000;  // Matrix size
    float *a, *b, *c;

    // Memory allocation
    posix_memalign((void**)&a, 16, n * n * sizeof(float));
    posix_memalign((void**)&b, 16, n * n * sizeof(float));
    posix_memalign((void**)&c, 16, n * n * sizeof(float));

    // RNG setup for the matrix
    std::mt19937 gen(42);
    std::normal_distribution<double> dist(0.0, 1.0);
    
    // Filling the matrix with gaussian values
    for(int i = 0; i < n * n; i++){
            a[i] = dist(gen);
            b[i] = dist(gen);
    }

    // We are using Apple's vDSP_mmul instead of openBLAS
    auto t1 = high_resolution_clock::now();
    cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
                n, n, n,
                1.0f,
                a, n,
                b, n,
                0.0f,
                c, n);
    auto t2 = high_resolution_clock::now();

    // Calculating throughput
    // Python's time.perf_counter() returns time in seconds instead of miliseconds (but float)
    auto runtime = duration_cast<milliseconds>(t2 - t1);
    double runtime_seconds = static_cast<double>(runtime.count()) / 1000.0;  // Convert to seconds
    double flops = static_cast<double>(2.0 * n * n * n) / runtime_seconds;  // 2 * n^3 FLOP/s

    std::cout << "Throughput: " << flops << " flop/s" << std::endl;  // Comparable to PyTorch

    // Freeing memory
    free(a);
    free(b);
    free(c);

    return 0;
    
}
