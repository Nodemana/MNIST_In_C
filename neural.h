#ifndef NEURAL_H
#define NEURAL_H

#include "matrixmath.h"
#include "mnist.h"
#include <math.h>
#include <stdbool.h>

typedef struct {
    Matrix weights;
    Matrix biases;
} Layer;

typedef struct {
    int num_layers;
    int *layer_sizes;
    Layer *layers;
    Matrix *layer_error;
} Network;

typedef struct {
    Matrix input_layer;
    Matrix *z_values;
    Matrix *activations;
    Matrix *errors;
    int num_activations;
} SampleResult;

typedef struct {
    SampleResult *sample_results;
    Matrix *costs; // Per sample
    Matrix *truths; // Per sample
    Matrix *batch_layer_error; // Per layer
    Matrix *weights_layer_error;
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

void normalise_input(Matrix *input);

Matrix forward_pass_layer(Layer *layer, Matrix *input);

SampleResult forward_pass(Network *network, Matrix *input_layer);

Batch init_batch(Network *network, int batch_size); 

void forward_pass_batch(Network *network, Batch *batch, double data_image[][IMAGE_SIZE], int labels[NUM_TRAIN], int num_samples, int batch_number);

Matrix calculate_output_layer_error(Matrix *cost_matrix, Matrix *current_z_value);

Matrix calculate_hidden_layer_error(Matrix *current_z_value, Matrix *next_cost, Matrix *next_layer_weights);

void backwards_pass_network(Network *network, SampleResult *sample_result, Matrix *cost_matrix);

void backwards_pass_batch(Network *network, Batch *batch);

void adjust_weights_and_biases(Network *network, Batch *batch, double learning_rate);

void clip_gradients(Matrix *m, double threshold);

bool evaluate_sample_performance(SampleResult *sample, Matrix *truth_matrix);

double evaluate_batch_performance(Batch *batch); 

void train_network(Network *network, int batch_size, int epochs, double learning_rate, double data_image[][IMAGE_SIZE], int labels[NUM_TRAIN]);

#endif

