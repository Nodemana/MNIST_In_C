#include <math.h>
#include <stdio.h>
#include <stdlib.h>


typedef struct {
    int length;
    double *data;
} Vector;

typedef struct {
    int rows;
    int cols;
    Vector **data;
} Matrix;

Vector vector(int length) {
    Vector v;
    v.length = length;
    v.data = (double*)malloc(length * sizeof(double *));
    return v;
}

Matrix matrix(int x, int y) {
    Matrix m;
    m.rows = x;
    m.cols = y;
    m.data = (Vector **)malloc(m.rows * sizeof(Vector *));
    for (int i = 0; i < m.rows; i++) {
        m.data[i] = (Vector *)malloc(m.cols * sizeof(Vector));
        for (int j = 0; j < m.cols; j++) {
            m.data[i][j] = *vector(vector_length);
        }
    }
    return m;
}

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

// Function to print a matrix of vectors
void print_matrix(Matrix *m) {
    for (int i = 0; i < m->rows; i++) {
        for (int j = 0; j < m->cols; j++) {
            print_vector(&(m->data[i][j]));
        }
        printf("\n");
    }
}

// Function to multiply two matrices
Matrix dot_product(Matrix *m, Matrix *n) {
    if (m->cols != n->rows) {
        printf("ERROR: Row and Column Mismatch!\n");
        exit(EXIT_FAILURE);
    }

    int vector_length = m->data[0][0].length;
    Matrix result = create_matrix(m->rows, n->cols, vector_length);
    for (int i = 0; i < m->rows; i++) {
        for (int j = 0; j < n->cols; j++) {
            double dot_product_value = 0.0;
            for (int k = 0; k < m->cols; k++) {
                dot_product_value += vector_dot_product(&(m->data[i][k]), &(n->data[k][j]));
            }
            for (int l = 0; l < vector_length; l++) {
                result.data[i][j].data[l] = dot_product_value;
            }
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

int main() {
    // Define matrices
    Matrix n = create_matrix(3, 1);
    n.data[0][0] = 1;
    n.data[1][0] = 2;
    n.data[2][0] = 3;
    
    Matrix m = create_matrix(1, 3);
    m.data[0][0] = 2;
    m.data[0][1] = 2;
    m.data[0][2] = 3;
    // Multiply matrices
    //Matrix result = multiply_matrices(&m, &n);
    double c = vector_dot_product(&m, &n);
    // Print matrices
    printf("Matrix M\n");
    print_matrix(&m);
    printf("\nMatrix N\n");
    print_matrix(&n);
    //printf("\nResult Matrix\n");
    //print_matrix(&result);
    printf("Result: %lf\n", c);
    // Free matrices
    free_matrix(&m);
    free_matrix(&n);
    //free_matrix(&result);

    return 0;
}
