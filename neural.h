#ifndef NEURAL_H
#define NEURAL_H

#include "matrixmath.h"
#include "mnist.h"
#include <math.h>

typedef struct {
    Matrix weights;
    Matrix biases;
} Layer;

typedef struct {
    int num_layers;
    Layer *layers;
    Matrix *layer_error;
} Network;

typedef struct {
    Matrix *z_values;
    Matrix *activations;
    Matrix *errors;
    int num_activations;
} SampleResult;

typedef struct {
    SampleResult *sample_results;
    Matrix *costs;
    Matrix *truths;
    int batch_size;
} Batch;

double squared_error(double y_hat, double y);

Matrix compute_cost_matrix(Matrix *predicted, Matrix *actual);

void compute_cost_matrix_sum(Matrix *sum, Matrix *predicted, Matrix *actual);

Matrix compute_batch_cost_matrix(Batch *batch);

Matrix init_truth_matrix(int *train_label, int *current_index);

Layer init_layer(int neurons, int connections);

Network init_network(int num_layers, int *layer_sizes, int input_layer_size);

void free_layer(Layer *layer);

void free_network(Network *network);

void print_layer(Layer *layer);

void print_network(Network *network);

Matrix forward_pass_layer(Layer *layer, Matrix *input);

SampleResult forward_pass(Network *network, Matrix *input_layer);

Batch forward_pass_batch(Network *network, double data_image[][IMAGE_SIZE], int labels[NUM_TRAIN],  int batch_size, int num_samples);


#endif

