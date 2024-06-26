#include "sigmoid.h"
#include "matrixmath.h"

int main() {
    srand(time(0) + getpid());
    // Define matrices
    Matrix n = matrix(3, 1);
    n.data[0][0] = 1;
    n.data[1][0] = 2;
    n.data[2][0] = 3;
    
    Matrix m = matrix(1, 3);
    m.data[0][0] = 2;
    m.data[0][1] = 2;
    m.data[0][2] = 3;

    // Multiply matrices
    Matrix result = matrix_multiply(&m, &n);

    Matrix transpose = transpose_matrix(&m);
    //double c = vector_dot_product(&m, &n);
    // Print matrices
    printf("Matrix M\n");
    print_matrix(&m);
    printf("\nMatrix N\n");
    print_matrix(&n);
    printf("\nMultiplied Matrix\n");
    print_matrix(&result);

    printf("\nTransposed Matrix M\n");
    print_matrix(&transpose);

    Matrix addition = matrix_add(&transpose, &n);
    printf("\nAdded Matrix M\n");
    print_matrix(&addition);
    //printf("Result: %lf\n", c);
    // Free matrices
    free_matrix(&m);
    free_matrix(&n);
    free_matrix(&result);
    free_matrix(&transpose);
    free_matrix(&addition);
    printf("\n");
   // for(int i = 0; i<10 ;i++) {
    //    printf("%lf\n", random_double());
   // }
  
    Matrix k = init_matrix(5,5);
 
    printf("Random Matrix:\n");
    print_matrix(&k);

    printf("Sigmoid Matrix:\n");
    Matrix a = activation(&k);
    print_matrix(&a);
    return 0;
}
