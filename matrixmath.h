#ifndef DEF_MATRIXMATH
#define DEF_MATRIXMATH

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <unistd.h>
#include "sigmoid.h"

// Definition of Vector struct
typedef struct {
    int length;
    double *data;
} Vector;

// Definition of Matrix struct
typedef struct {
    int rows;
    int cols;
    double **data;
} Matrix;

Vector vector(int length);

Matrix matrix(int rows, int cols);

void free_matrix(Matrix *m);

void print_vector(Vector *v);

void print_matrix(Matrix *m);

Matrix matrix_multiply(Matrix *m, Matrix *n);

Matrix matrix_add(Matrix *m, Matrix *n);

Matrix transpose_matrix(Matrix *a);

double vector_dot_product(Vector *v1, Vector *v2);

double random_double();

Matrix init_matrix(int rows, int cols);

Matrix activation(Matrix *m);

#endif
