#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <unistd.h>
#include "sigmoid.h"
#include "matrixmath.h"

// Function to create a vector
Vector vector(int length) {
    Vector v;
    v.length = length;
    v.data = (double *)malloc(length * sizeof(double));
    return v;
}

// Function to create a matrix
Matrix matrix(int rows, int cols) {
    Matrix m;
    m.rows = rows;
    m.cols = cols;
    m.data = (double **)malloc(m.rows * sizeof(double *));
    for (int i = 0; i < m.rows; i++) {
        m.data[i] = (double *)malloc(m.cols * sizeof(double));
    }
    return m;
}

// Function to free memory allocated for a matrix
void free_matrix(Matrix *m) {
    for (int i = 0; i < m->rows; i++) {
        free(m->data[i]);
    }
    free(m->data);
    m->data = NULL;
    m->rows = 0;
    m->cols = 0;
}

// Function to print a vector
void print_vector(Vector *v) {
    for (int i = 0; i < v->length; i++) {
        printf("%lf ", v->data[i]);
    }
    printf("\n");
}

// Function to print a matrix of doubles
void print_matrix(Matrix *m) {
    for (int i = 0; i < m->rows; i++) {
        for (int j = 0; j < m->cols; j++) {
            printf("%lf ", m->data[i][j]);
        }
        printf("\n");
    }
}

// Function to multiply two matrices
Matrix matrix_multiply(Matrix *m, Matrix *n) {
    if (m->cols != n->rows) {
        printf("ERROR: Row and Column Mismatch!\n");
        exit(EXIT_FAILURE);
    }

    Matrix result = matrix(m->rows, n->cols);
    for (int i = 0; i < m->rows; i++) {
        for (int j = 0; j < n->cols; j++) {
            result.data[i][j] = 0.0;
            for (int k = 0; k < m->cols; k++) {
                result.data[i][j] += m->data[i][k] * n->data[k][j];
            }
        }
    }

    return result;
}

Matrix matrix_add(Matrix *m, Matrix *n) {
    if(m->rows != n->rows || m->cols != n->cols){
        printf("ERROR: Matrices Dimensions Don't Match!\n");
        exit(EXIT_FAILURE);
    }
    Matrix result = matrix(m->rows, n->cols);
    for(int i = 0; i < m->rows; i++) {
        for(int j = 0; j < m->cols; j++) {
            result.data[i][j] = m->data[i][j] + n->data[i][j];
        }
    }
    return result;
}

Matrix transpose_matrix(Matrix *a) {
    Matrix result = matrix(a->cols, a->rows);
    for (int i = 0; i < a->cols; i++) {
        for (int j = 0; j < a->rows; j++) {
            result.data[i][j] = a->data[j][i];
        }
    }
    return result;
}

// Function to calculate the dot product of two vectors
double vector_dot_product(Vector *v1, Vector *v2) {
    if (v1->length != v2->length) {
        printf("ERROR: Vector Length Mismatch!\n");
        exit(EXIT_FAILURE);
    }
    double result = 0.0;
    for (int i = 0; i < v1->length; i++) {
        result += v1->data[i] * v2->data[i];
    }
    return result;
}

double random_double() {
    return (double)rand()/RAND_MAX*2.0-1.0;
}

Matrix init_matrix(int rows, int cols) {
    Matrix result = matrix(rows, cols);

    for(int i = 0; i<rows; i++) {
        for(int j = 0; j<cols; j++) {
            result.data[i][j] = random_double();
        }
    }
    return result;
}

Matrix activation(Matrix *m) {
    Matrix result = matrix(m->rows, m->cols);

    for (int i = 0; i < m->rows; i++) {
        for (int j = 0; j < m->cols; j++) {
            result.data[i][j] = sigmoid(m->data[i][j]);        
        }
    }
    return result;
}


