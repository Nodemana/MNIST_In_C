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

Matrix element_wise(Matrix *m, Matrix *n) { 
    if(m->rows != n->rows) {
        printf("ERROR: Dimensions innappropriate for element wise operation.\n");
        printf("Matrix A Rows: %d, Matrix B Rows: %d\n", m->rows, n->rows);
        exit(EXIT_FAILURE);
    }
    Matrix result = matrix(m->rows, m->cols);
    for(int i = 0; i < m->rows; i++){
       result.data[i][0] =  m->data[i][0] * n->data[i][0];
    }
    return result;
}

// Function to multiply two matrices
// This function is used to compute the dot product between the last activation
// and the next layers weights.
Matrix matrix_multiply(Matrix *m, Matrix *n) {
    if (m->cols != n->rows) {
        //printf("ERROR: Row and Column Mismatch!\n");
        //exit(EXIT_FAILURE);
        printf("ERROR: Multiply Dimension Mismatch!\n");
        printf("Matrix A rows and cols:\n");
        printf("Rows %d\n", m->rows);
        printf("Cols %d\n", m->cols);
        //print_matrix(predicted);
        printf("Matrix B:\n");
        printf("Rows %d\n", n->rows);
        printf("Cols %d\n", n->cols);
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

// Function to add to matrices togethor
// This function is used to add the bias vector to the resultant vector of the
// dot product between the weights and last activation.
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

// Function to add to matrices togethor
// This function is used to add the bias vector to the resultant vector of the
// dot product between the weights and last activation.
void matrix_add_inplace(Matrix *m, Matrix *n) {
    if(m->rows != n->rows || m->cols != n->cols){
        printf("ERROR: Matrices Dimensions Don't Match!\n");
        exit(EXIT_FAILURE);
    }
    for(int i = 0; i < m->rows; i++) {
        for(int j = 0; j < m->cols; j++) {
            m->data[i][j] = m->data[i][j] + n->data[i][j];
        }
    }
}

Matrix matrix_scalar_divide(Matrix *m, double x) {
    Matrix result = matrix(m->rows, m->cols);
    for(int i = 0; i< m->rows; i++) {
        for(int j = 0; j< m->cols; j++){
            result.data[i][j] = m->data[i][j] / x;
        }
    }
    return result;
}

Matrix matrix_scalar_multiply(Matrix *m, double x) {
    Matrix result = matrix(m->rows, m->cols);
    for(int i = 0; i< m->rows; i++) {
        for(int j = 0; j< m->cols; j++){
            result.data[i][j] = m->data[i][j] * x;
        }
    }
    return result;
}

void matrix_scalar_divide_inplace(Matrix *m, double x) {
    for (int i = 0; i < m->rows; i++) {
        for (int j = 0; j < m->cols; j++) {
            m->data[i][j] /= x;
        }
    }
}

void matrix_scalar_multiply_inplace(Matrix *m, double x) {
    for (int i = 0; i < m->rows; i++) {
        for (int j = 0; j < m->cols; j++) {
            m->data[i][j] *= x;
        }
    }
}

// Function transposes a given matrix.
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

// Used to initialise matrices with random doubles.
double random_double() {
    return (double)rand()/RAND_MAX*2.0-1.0;
}

// This function generates a matrice of a certain size of random doubles.
Matrix init_matrix(int rows, int cols) {
    Matrix result = matrix(rows, cols);

    for(int i = 0; i<rows; i++) {
        for(int j = 0; j<cols; j++) {
            result.data[i][j] = random_double();
        }
    }
    return result;
}

// This function generates a matrice of a certain size with a given double value.
Matrix init_matrix_value(int rows, int cols, double value) {
    Matrix result = matrix(rows, cols);

    for(int i = 0; i<rows; i++) {
        for(int j = 0; j<cols; j++) {
            result.data[i][j] = value;
        }
    }
    return result;
}

Matrix sum_horizontally(Matrix *m) {
    Matrix result = init_matrix_value(m->rows, 1, 0);
    for(int i = 0; i < m->rows; i++) {
        for(int j = 0; j < m->cols; j++) {
            result.data[i][0] += m->data[i][j];
        }
    }
    return result;
}

Matrix sum_vertically(Matrix *m) {
    Matrix result = init_matrix_value(1, m->cols, 0);
    for(int j = 0; j < m->cols; j++) {
        for(int i = 0; i < m->rows; i++) {
            result.data[0][j] += m->data[i][j];
        }
    }
    return result;
}

// This function applys the activation function (sigmoid) on a given matrice.
Matrix activation(Matrix *m) {
    Matrix result = matrix(m->rows, m->cols);

    for (int i = 0; i < m->rows; i++) {
        for (int j = 0; j < m->cols; j++) {
            result.data[i][j] = sigmoid(m->data[i][j]);        
        }
    }
    return result;
}

Matrix derivative_activation(Matrix *m) {
    Matrix result = matrix(m->rows, m->cols);

    for (int i = 0; i < m->rows; i++) {
        for (int j = 0; j < m->cols; j++) {
            result.data[i][j] = derivative_sigmoid(m->data[i][j]);        
        }
    }
    return result;
}

void activation_inplace(Matrix *m) {
    for (int i = 0; i < m->rows; i++) {
        for (int j = 0; j < m->cols; j++) {
            m->data[i][j] = sigmoid(m->data[i][j]);        
        }
    }
}

// Helper function to copy a matrix
Matrix copy_matrix(Matrix *m) {
    Matrix copy = matrix(m->rows, m->cols);
    for (int i = 0; i < m->rows; i++) {
        for (int j = 0; j < m->cols; j++) {
            copy.data[i][j] = m->data[i][j];
        }
    }
    return copy;
}

// Function to find the index of the maximum value in a matrix row
int argmax(Matrix *matrix) {
    int max_index = 0;
    double max_value = matrix->data[0][0];
    for (int i = 1; i < matrix->rows * matrix->cols; i++) {
        if (matrix->data[i][0] > max_value) {
            max_value = matrix->data[i][0];
            max_index = i;
        }
    }
    return max_index;
}
