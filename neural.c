
#include "neural.h"

double squared_error(double y_hat, double y) {
   return pow(y_hat - y, 2);
}


// For one training sample this can calculate the cost function between
// the true matrix and the final activation output of the last layer.
Matrix compute_cost_matrix(Matrix *predicted, Matrix *actual) {
    if(predicted->rows != actual->rows){
        printf("ERROR: Dimension Mismatch!\n");
        exit(EXIT_FAILURE);
    }

    Matrix cost = matrix(predicted->rows, 1);

    for(int i = 0; i < predicted->rows; i++) {
        cost.data[i][0] = squared_error(predicted->data[i][0], actual->data[i][0]);
    }

    return cost;
}

Matrix init_weights_layer(int neurons, int weights) {
    Matrix layer = init_matrix(weights, neurons);





}
