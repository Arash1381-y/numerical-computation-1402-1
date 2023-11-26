#include <cuda_runtime_api.h>
#include "matrix_mult_gpu_shared.cuh"



/**
 * CUDA kernel for matrix multiplication using shared memory optimization on GPU.
 * Computes the product of two matrices (m1 and m2) and stores the result in matrix r.
 *
 * @param m1 Pointer to the first input matrix (matrix struct)
 * @param m2 Pointer to the second input matrix (matrix struct)
 * @param r Pointer to the resulting matrix (matrix struct)
 */
__global__
void mult_gpu_shared_kernel(matrix *m1, matrix *m2, matrix *r) {

    __shared__
    double shared_memory[BLOCK_SIZE_X][BLOCK_SIZE_Y][BLOCK_SIZE_Z];

    uint row = blockIdx.x * blockDim.x + threadIdx.x;
    uint col = blockIdx.y * blockDim.y + threadIdx.y;

    // check if row and col are in range
    if (row < m1->n_rows && col < m2->n_cols) {
        if (m1->n_cols < BLOCK_SIZE_Z) {
            if (threadIdx.z == 0) {
                // compute the value of the element
                double val = 0;
                for (uint i = 0; i < m1->n_cols; i++) {
                    val += m1->flatt_val[row * m1->n_cols + i] * m2->flatt_val[i * m2->n_cols + col];
                }

                // write the value to the result matrix
                r->flatt_val[row * r->n_cols + col] = val;
            }
            return;
        }

        // compute the index of the element need to be iterated over
        uint start_index = threadIdx.z * (m1->n_cols / BLOCK_SIZE_Z);
        uint end_index = min((threadIdx.z + 1) * (m1->n_cols / BLOCK_SIZE_Z), m1->n_cols);

        // compute the value of the element
        double val = 0;
        for (uint i = start_index; i < end_index; i++) {
            val += m1->flatt_val[row * m1->n_cols + i] * m2->flatt_val[i * m2->n_cols + col];
        }

        // save the value in the shared memory
        shared_memory[threadIdx.x][threadIdx.y][threadIdx.z] = val;

        // wait for all threads to finish
        cudaDeviceSynchronize();


        // parallel reduction
        for (int i = BLOCK_SIZE_Z / 2; i > 0; i /= 2) {
            if (threadIdx.z < i) {
                shared_memory[threadIdx.x][threadIdx.y][threadIdx.z] += shared_memory[threadIdx.x][threadIdx.y][
                        threadIdx.z + i];
            }
            cudaDeviceSynchronize();
        }

        // write the value to the result matrix
        if (threadIdx.z == 0) {
            r->flatt_val[row * r->n_cols + col] = shared_memory[threadIdx.x][threadIdx.y][0];
        }
    }

}


__host__
void mult_gpu_shared(matrix *m1, matrix *m2, matrix *r) {
    // define cuda allocated matrix struct
    matrix *device_m1, *device_m2, *device_r;
    double *device_m1_data, *device_m2_data, *device_r_data;

    init_matrix_cuda(&device_m1, m1, &device_m1_data);
    init_matrix_cuda(&device_m2, m2, &device_m2_data);
    init_matrix_cuda(&device_r, r, &device_r_data);
    cudaDeviceSynchronize();

    // define block size
    dim3 block_size(BLOCK_SIZE_X, BLOCK_SIZE_Y, BLOCK_SIZE_Z);

    // define grid size
    uint row_num = m1->n_rows / block_size.x + 1;
    uint col_num = m2->n_cols / block_size.y + 1;
    dim3 grid_size(row_num, col_num);

    // call the kernel
    mult_gpu_shared_kernel<<<grid_size, block_size>>>(device_m1, device_m2, device_r);
    cudaDeviceSynchronize();

    // copy result back to host
    CHECK_CUDA_ERROR(
            cudaMemcpy(r->flatt_val, device_r_data, r->n_cols * r->n_rows * sizeof(double),
                       cudaMemcpyDeviceToHost)
    );


    // free cuda memory
    CHECK_CUDA_ERROR(cudaFree(device_m1));
    CHECK_CUDA_ERROR(cudaFree(device_m2));
    CHECK_CUDA_ERROR(cudaFree(device_r));
    CHECK_CUDA_ERROR(cudaFree(device_m1_data));
    CHECK_CUDA_ERROR(cudaFree(device_m2_data));
    CHECK_CUDA_ERROR(cudaFree(device_r_data));

}