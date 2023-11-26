#include <ctime>
#include "data-structure/matrix.cuh"
#include "baseline/matrix_mult_baseline.h"
#include "gpu_no_shared/matrix_mult_gpu_no_shared.cuh"
#include "gpu_with_shared_mem/matrix_mult_gpu_shared.cuh"

#define LOG_EXE_TIME(time)(printf("Execution time: %f\n", time))

/**
 * @brief Checks if two matrices are equal.
 *
 * This function checks if two matrix objects are equal, i.e., they have the same dimensions
 * and all corresponding elements are equal.
 *
 * @param m1 Pointer to the first matrix object.
 * @param m2 Pointer to the second matrix object.
 *
 * @return True if the matrices are equal, false otherwise.
 */
bool is_matrix_equal(matrix *m1, matrix *m2) {
    if (m1->n_rows != m2->n_rows || m1->n_cols != m2->n_cols) {
        return false;
    }

    for (size_t i = 0; i < m1->n_rows; i++) {
        for (size_t j = 0; j < m1->n_cols; j++) {
            // compare 3 digits after the decimal point
            if ((int) (m1->flatt_val[i * m1->n_cols + j] * 1000) !=
                (int) (m2->flatt_val[i * m2->n_cols + j] * 1000)) {
                return false;
            }
        }
    }

    return true;
}


/**
 * \brief Calculate the baseline test for matrix multiplication.
 *
 * This function calculates the baseline test for matrix multiplication by performing
 * the multiplication of two matrices, `m1` and `m2`, and storing the result in `r`.
 *
 * \param m1 Pointer to the first matrix.
 * \param m2 Pointer to the second matrix.
 * \param r Pointer to the matrix where the result will be stored.
 */
double test_baseline(matrix *m1, matrix *m2, matrix *r) {

    double time;
    clock_t start, end;

    start = clock();
    mult_baseline(m1, m2, r);
    end = clock();


    return (double) (end - start) / CLOCKS_PER_SEC;
}

/**
 * @brief Multiply two matrices without using shared memory.
 *
 * This function performs matrix multiplication on two matrices `m1` and `m2`,
 * and stores the result in the matrix `r`. The function assumes that the
 * dimensions of matrices `m1` and `m2` are compatible for multiplication, i.e.,
 * the number of columns in `m1` is equal to the number of rows in `m2`.
 *
 * @param m1 Pointer to the first matrix.
 * @param m2 Pointer to the second matrix.
 * @param r Pointer to the resulting matrix.
 *
 * @return void
 */
double test_gpu_no_shared_memory(matrix *m1, matrix *m2, matrix *r) {

    clock_t start, end;

    start = clock();
    mult_gpu_no_shared(m1, m2, r);
    end = clock();

    return (double) (end - start) / CLOCKS_PER_SEC;
}


/**
 * @brief Calculates matrix multiplication using GPU shared memory.
 *
 * This function calculates the matrix multiplication of two given matrices
 * (m1 and m2) using GPU shared memory and stores the result in the provided
 * matrix (r).
 *
 * @param m1 Pointer to the first matrix.
 * @param m2 Pointer to the second matrix.
 * @param r Pointer to the result matrix.
 *
 * @note The function assumes that the row size of matrix m1 is equal to the
 * column size of matrix m2. The result matrix r should already be allocated
 * with the appropriate size for the multiplication result.
 *
 * @note This function is designed for use with CUDA-enabled GPUs.
 * It should not be called directly on CPUs.
 *
 * @warning The matrices m1, m2, and r should be allocated and properly
 * initialized before calling this function, otherwise the behavior is undefined.
 */

double test_gpu_shared_memory(matrix *m1, matrix *m2, matrix *r) {

    clock_t start, end;

    start = clock();
    mult_gpu_shared(m1, m2, r);
    end = clock();

    return (double) (end - start) / CLOCKS_PER_SEC;
}


int main() {
    srand(time(NULL));  // Seed the random number generator with the current time
    size_t row, col;
    row = 1024;
    col = 1024;
    double sparsity = 0;


    auto *m1 = static_cast<matrix *>(malloc(sizeof(matrix)));
    auto *m2 = static_cast<matrix *>(malloc(sizeof(matrix)));
    auto *r1 = static_cast<matrix *>(malloc(sizeof(matrix)));


    init_matrix_host(m1, row, col, true, sparsity);
    init_matrix_host(m2, row, col, true, sparsity);
    init_matrix_host(r1, row, col, true, 1);

#ifdef DEBUG
    printf("matrix 1: \n");
    print_matrix_host(m1);
    printf("matrix 2: \n");
    print_matrix_host(m2);
#endif


    LOG_EXE_TIME(test_baseline(m1, m2, r1));
#ifdef DEBUG
    printf("baseline result: \n");
    print_matrix_host(r1);
#endif

    auto *r2 = static_cast<matrix *>(malloc(sizeof(matrix)));
    init_matrix_host(r2, row, col, true, 1);
    LOG_EXE_TIME(test_gpu_no_shared_memory(m1, m2, r2));

#ifdef DEBUG
    printf("no shared result: \n");
    print_matrix_host(r2);
#endif

    auto *r3 = static_cast<matrix *>(malloc(sizeof(matrix)));
    init_matrix_host(r3, row, col, true, 1);
    LOG_EXE_TIME(test_gpu_shared_memory(m1, m2, r3));

#ifdef DEBUG
    printf("shared result: \n");
    print_matrix_host(r3);
#endif

    if (!is_matrix_equal(r1, r2)) {
        printf("WRONG COMPUTATION!!!\n");
    }

    if (!is_matrix_equal(r1, r3)) {
        printf("WRONG COMPUTATION!!!\n");
    }


    free_matrix_host(m1);
    free_matrix_host(m2);
    free_matrix_host(r1);
    free_matrix_host(r2);
    free_matrix_host(r3);
}