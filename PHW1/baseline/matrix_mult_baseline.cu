#include "../data-structure/matrix.cuh"

void mult_baseline(matrix *m1, matrix *m2, matrix *r) {
    if (m1->n_cols != m2->n_rows) {
        printf("INVALID OPERATION!");
        return;
    }

    size_t row_num = m1->n_rows;
    size_t col_num = m2->n_cols;

    for (size_t i = 0; i < row_num; i++) {
        for (size_t j = 0; j < col_num; j++) {
            r->flatt_val[i * col_num + j] = 0;
            for (size_t k = 0; k < m1->n_cols; k++) {
                r->flatt_val[i * col_num + j] += m1->flatt_val[i * m1->n_cols + k] * m2->flatt_val[k * m2->n_cols + j];
            }
        }
    }
}