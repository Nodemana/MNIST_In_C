#include "matrixmath.h"
#include <math.h>

double squared_error(double y_hat, double y);

Matrix compute_cost_matrix(Matrix *predicted, Matrix *actual);
