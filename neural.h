#include "matrixmath.h"
#include <math.h>

typedef struct {
    Matrix weights;
    Matrix biases;
} Layer;

typedef struct {
    int num_layers;
    Layer *layers;
} Network;

typedef struct {
    Matrix *activations;
    int num_activations;
} ForwardPassResult;

double squared_error(double y_hat, double y);

Matrix compute_cost_matrix(Matrix *predicted, Matrix *actual);

void compute_cost_matrix_sum(Matrix *sum, Matrix *predicted, Matrix *actual);

Matrix init_truth_matrix(int *train_label, int *current_index);

Layer init_layer(int neurons, int connections);

Network init_network(int num_layers, int *layer_sizes, int input_layer_size);

void free_layer(Layer *layer);

void free_network(Network *network);

void print_layer(Layer *layer);

void print_network(Network *network);

Matrix forward_pass_layer(Layer *layer, Matrix *input);

ForwardPassResult forward_pass(Network *network, Matrix *input_layer);