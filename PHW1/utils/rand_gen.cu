#include <cstdlib>
#include "rand_gen.h"


int UPPER_BOUND_R = 3;
int LOWER_BOUND_R = -3;


double rand_gen_m_entry() {
    double rand_value = (((double) rand()) / RAND_MAX) * 2 * UPPER_BOUND_R + LOWER_BOUND_R;
    return rand_value;
}

double simple_rand() {
    return (double) rand() / RAND_MAX;
}