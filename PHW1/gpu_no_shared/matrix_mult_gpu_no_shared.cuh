#ifndef PHW1_MATRIX_MULT_GPU_NO_SHARED_CUH
#define PHW1_MATRIX_MULT_GPU_NO_SHARED_CUH


#include "../data-structure/matrix.cuh"

#define BLOCK_SIZE_X 32
#define BLOCK_SIZE_Y 32

__host__
void mult_gpu_no_shared(matrix *m1, matrix *m2, matrix *r);


#endif //PHW1_MATRIX_MULT_GPU_NO_SHARED_CUH
