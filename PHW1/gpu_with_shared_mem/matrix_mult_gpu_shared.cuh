#ifndef PHW1_MATRIX_MULT_GPU_SHARED_CUH
#define PHW1_MATRIX_MULT_GPU_SHARED_CUH

#include "../data-structure/matrix.cuh"


#define BLOCK_SIZE_X 16
#define BLOCK_SIZE_Y 16
#define BLOCK_SIZE_Z 4

__host__
void mult_gpu_shared(matrix *m1, matrix *m2, matrix *r);

#endif //PHW1_MATRIX_MULT_GPU_SHARED_CUH
