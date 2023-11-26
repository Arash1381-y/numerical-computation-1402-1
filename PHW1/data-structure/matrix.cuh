#ifndef C_PHW1_MATRIX_H
#define C_PHW1_MATRIX_H

#include <iostream>

#define CHECK_CUDA_ERROR(val) check((val), #val, __FILE__, __LINE__)

template<typename T>
void check(T err, const char *const func, const char *const file,
           const int line);


typedef struct matrix {
    size_t n_rows;
    size_t n_cols;
    double *flatt_val;
} matrix;


__host__ // host function
bool init_matrix_host(
        matrix *m,
        size_t row,
        size_t col,
        bool random_init,
        double sparsity
);


__host__ // host function
void print_matrix_host(matrix *m);

__host__ // host function
void free_matrix_host(matrix *m);

__host__ // host function
void init_matrix_cuda(matrix **m_device, matrix *m_host, double **d_device);


#define PRINT_MATRIX_DEVICE(m) print_matrix_device<<<1, 1>>>(m); cudaDeviceSynchronize();

__global__ // kernel function
void print_matrix_device(matrix *m);

#endif
