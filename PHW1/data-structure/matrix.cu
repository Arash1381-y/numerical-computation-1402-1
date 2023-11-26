#include "matrix.cuh"
#include "./../utils/rand_gen.h"


template<typename T>
void check(T err, const char *const func, const char *const file,
           const int line) {
    if (err != cudaSuccess) {
        std::cerr << "CUDA Runtime Error at: " << file << ":" << line
                  << std::endl;
        std::cerr << cudaGetErrorString(err) << " " << func << std::endl;
    }
}

__host__
bool init_matrix_host(
        matrix *m,
        size_t row,
        size_t col,
        bool random_init,
        double sparsity
) {
    m->n_rows = row;
    m->n_cols = col;
    m->flatt_val = static_cast<double *>(malloc(row * col * sizeof(double)));

    if (!m->flatt_val || !random_init) return false;

    for (int i = 0; i < row; i++) {
        for (int j = 0; j < col; j++) {
            double rand_value = simple_rand();
            if (rand_value < sparsity) {
                m->flatt_val[i * col + j] = 0;
            } else {
                m->flatt_val[i * col + j] = rand_gen_m_entry();
            }
        }
    }
    return true;
}


__host__
void print_matrix_host(
        matrix *m
) {
    for (int i = 0; i < m->n_rows; i++) {
        for (int j = 0; j < m->n_cols; j++) {
            printf("%f, ", m->flatt_val[i * m->n_cols + j]);
        }
        printf("\n");
    }
}

__host__
void free_matrix_host(matrix *m) {
    free(m->flatt_val);
    free(m);
}


__global__
static void swap_space_kernel(matrix *m, double *data) {
    m->flatt_val = data;
}


__host__
void init_matrix_cuda(matrix **m_device, matrix *m_host, double **d_device) {
    auto matrix_size_in_byt = m_host->n_cols * m_host->n_rows * sizeof(double);
    CHECK_CUDA_ERROR(cudaMalloc(m_device, sizeof(matrix)));
    CHECK_CUDA_ERROR(cudaMalloc(d_device, matrix_size_in_byt));
    CHECK_CUDA_ERROR(cudaMemcpy(*m_device, m_host, sizeof(matrix), cudaMemcpyHostToDevice));
    CHECK_CUDA_ERROR(cudaMemcpy(*d_device, m_host->flatt_val, matrix_size_in_byt, cudaMemcpyHostToDevice));
    swap_space_kernel<<<1, 1>>>(*m_device, *d_device);
}


__global__
void print_matrix_device(matrix *m) {
    if (blockIdx.x == 0 && blockIdx.y == 0) {
        for (int i = 0; i < m->n_rows; i++) {
            for (int j = 0; j < m->n_cols; j++) {
                printf("%f ", m->flatt_val[i * m->n_cols + j]);
            }
            printf("\n");
        }
    }
    printf("\n");
    cudaDeviceSynchronize();
}



