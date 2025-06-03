#include <iostream>
#include <random>
#include <vector>
#include "nvpl_lapack.h"
#include "nvpl_blas_cblas.h"
#include <complex>
#include <algorithm>

// We are reusing some of the code from #1 but modified so it would be considered as a function

// Just to print the matrix
void printMatrix(const std::vector<std::vector<double>>& matrix){
    for(const auto& row : matrix){
        for(double value : row){
            std::cout << value;
            }
            std::cout << std::endl;
        }
}


// Generating the random matrix
std::vector<std::vector<double>> gaussian_matrix(int n, double mean=0.0, double std=1.0){
    std::vector<std::vector<double>> matrix(n, std::vector<double>(n));
    
    // RNG setup
    std::random_device rng;
    std::mt19937 gen(rng());
    std::normal_distribution<double> dist(mean, std);
    
    // Filling the matrix with gaussian values
    for(int i = 0; i < n; i++){
        for(int j = 0; j < n; j++){
            matrix[i][j] = dist(gen);
        }
    }
    
    return matrix;
}


// "Junkyard torch.arange()" implementation
// This outputs a 1-D tensor (or simply a vector) with values of start and end.
std::vector<double> arange(double start, double end, double step){
    std::vector<double> vector;
    
    for(double value = start; value < end; value += step){
        vector.push_back(value);
    }
    
    return vector;
}


// "Junkyard torch.diag(torch.arange())" implementation
// This outputs a diagonal 2-D tensor (or simply a matrix) by taking a 1-D tensor (or vector).
std::vector<std::vector<double>> diag(double start, double end, double step){
    std::vector<double> values = arange(start, end, step);  // get arange values
    int n = values.size();  // Get size
    std::vector<std::vector<double>> matrix(n, std::vector<double>(n, 0.0));  // Make matrix using arange n
    
    for(int i = 0; i < n; i++){
        matrix[i][i] = values[i];
    }

    return matrix;
}
// Matrix inversion using LAPACK
std::vector<std::vector<double>> inverse(std::vector<std::vector<double>>& matrix){
    int n = matrix.size();
    
    // Converting the 2-D tensor into 1-D column major tensor
    // LAPACK requirement
    std::vector<double> flat_matrix(n * n);
    
    for(int i = 0; i < n; i++){
        for(int j = 0; j < n; j++){
            flat_matrix[j * n + i] = matrix[i][j];  // In column major order
        }
    }

    std::vector<int> ipiv(n);  // Pivot indices
    
    // LAPACKE operations
    LAPACKE_dgetrf(LAPACK_COL_MAJOR, n, n, flat_matrix.data(), n, ipiv.data());
    LAPACKE_dgetri(LAPACK_COL_MAJOR, n, flat_matrix.data(), n, ipiv.data());
    
    // Converting back to 2-D tensor (matrix)
    std::vector<std::vector<double>> inverse_matrix(n, std::vector<double>(n));
    
    for(int i = 0; i < n; i++){
        for(int j = 0; j < n; j++){
            inverse_matrix[i][j] = flat_matrix[j * n + i];
        }
    }

    return inverse_matrix;
}


// Matrix multiplication using BLAS
std::vector<std::vector<double>> matmul(std::vector<std::vector<double>>& matrix1, std::vector<std::vector<double>>& matrix2){
    int n = matrix1.size();  // We assume that both matrices have the same size, so no bother checking

    // Converting the 2-D tensor into 1-D column major tensor (required by BLAS)
    std::vector<double> flat_matrix1(n * n);
    std::vector<double> flat_matrix2(n * n);

    for(int i = 0; i < n; i++){
        for(int j = 0; j < n; j++){
            flat_matrix1[j * n + i] = matrix1[i][j];
            flat_matrix2[j * n + i] = matrix2[i][j];
        }
    }

    // BLAS operations
    std::vector<double> result(n * n);  // Keeping the results

    cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, n, n, n, 1.0, flat_matrix1.data(), n, flat_matrix2.data(), n, 0.0, result.data(), n);

    // Converting the result to 2-D tensor (matrix)
    std::vector<std::vector<double>> matmul_result(n, std::vector<double>(n));
    
    for(int i = 0; i < n; i++){
        for(int j = 0; j < n; j++){
            matmul_result[i][j] = result[j * n + i];
        }
    }

    return matmul_result;
}


// Finding the eigenvalues
std::vector<std::complex<double>> getEigenvalues(std::vector<std::vector<double>>& matrix){
    int n = matrix.size();  // As usual we assume the matrix is square
    
    // Converting the 2-D tensor into 1-D because of LAPACK
    std::vector<double> flat_matrix(n * n);
    
    for(int i = 0; i < n; i++){
        for(int j = 0; j < n; j++){
            flat_matrix[j * n + i] = matrix[i][j];
        }
    }
    
    // LAPACK for getting the eigenvalues
    std::vector<double> wr(n);  // Real
    std::vector<double> wi(n);  // Imaginary

    LAPACKE_dgeev(LAPACK_COL_MAJOR, 'N', 'N', n, flat_matrix.data(), n, wr.data(), wi.data(), nullptr, n, nullptr, n);

    // Converting to std::complex and returning values
    std::vector<std::complex<double>> eigenvalues(n);
    for(int i = 0; i < n; i++){
        eigenvalues[i] = std::complex<double>(wr[i], wi[i]);
    }

    // Optional: sorting it like how the assignment wants it
    std::sort(eigenvalues.begin(), eigenvalues.end(), [](const std::complex<double> &a, const std::complex<double> &b){
    return a.real() < b.real();
    });
    
    return eigenvalues;

}    
     
    
// Main code
int main(){
    
    int n = 20;  // Matrix size

    // Make matrices
    auto matrix = gaussian_matrix(n);
    auto diag_arange_matrix = diag(1, n + 1, 1);
    auto matrix_inversed = inverse(matrix);
    
    // Matmul 
    auto inverse_diag_matmul = matmul(matrix_inversed, diag_arange_matrix);    
    auto final_matmul = matmul(inverse_diag_matmul, matrix);    
    
    // Find eigenvalues
    // And also printing it
    std::vector<std::complex<double>> eigenvalues = getEigenvalues(final_matmul);
    
    for (const auto &ev : eigenvalues){
        std::cout << ev << std::endl;
    }
    
    
    return 0;
}    
