#include "sigmoid.h"
#include "matrixmath.h"
#include "neural.h"
#include "mnist.h"

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
    printf("\n");

    Matrix actual = init_matrix_value(10,1,0);

    actual.data[5][0] = 1;

    Matrix predicted = init_matrix(10,1);
    
    printf("Predicted Matrix:\n");
    print_matrix(&predicted);
    printf("Actual Matrix:\n");
    print_matrix(&actual);

    Matrix cost_result = compute_cost_matrix(&predicted, &actual);
    printf("Cost Matrix:\n");
    print_matrix(&cost_result);

    compute_cost_matrix_sum(&cost_result, &predicted, &actual);
    printf("Cost Matrix Sum:\n");
    print_matrix(&cost_result);

    Matrix cost_flatten = sum_vertically(&k);
    printf("Cost Flatten:\n");
    print_matrix(&cost_flatten);

    free_matrix(&cost_flatten);

    free_matrix(&predicted);
    free_matrix(&actual);
    free_matrix(&cost_result);

    //load_mnist();
    //print_mnist_pixel(train_image, 1);
    int num_layers = 3;
    int layer_sizes[] = {2, 2, 1};
    int input_layer_size = 3;

    Network network = init_network(num_layers, layer_sizes, input_layer_size);

    print_network(&network);

    free_network(&network);


    return 0;
}
