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

void compute_cost_matrix_sum(Matrix *sum, Matrix *predicted, Matrix *actual) {
    Matrix temp_cost_matrix = compute_cost_matrix(predicted, actual);
    matrix_add_inplace(sum, &temp_cost_matrix);
    free_matrix(&temp_cost_matrix);
}

Matrix init_weights_layer(int neurons, int weights) {
    Matrix layer = init_matrix(neurons, weights); 
    return layer;
}

Matrix init_truth_matrix(int *train_label, int *current_index) {
    Matrix truth = init_matrix_value(10, 1, 0);

    truth.data[train_label[*current_index]][0] = 1; 
    
    return truth;
}

Matrix init_bias_layer(int neurons, int biases) {
    Matrix layer = init_matrix(neurons, biases); 
    return layer;
}

Layer init_layer(int neurons, int connections) {
    Layer layer;
    layer.weights = init_weights_layer(neurons, connections);
    layer.biases = init_bias_layer(neurons, 1);
    return layer;
}

Network init_network(int num_layers, int *layer_sizes, int input_layer_size) {
    Network network;
    network.num_layers = num_layers;
    network.layers = (Layer *)malloc(num_layers * sizeof(Layer));

    for (int i = 0; i < num_layers; i++) {
        int neurons = layer_sizes[i];
        int connections = (i == 0) ? input_layer_size : layer_sizes[i - 1];  // Connections i.e wieghts and bias matrices are neurons[i-1] * neurons[i]
        network.layers[i] = init_layer(neurons, connections);
    }

    return network;
}

void free_layer(Layer *layer) {
    free_matrix(&layer->weights);
    free_matrix(&layer->biases);
}

void free_network(Network *network) {
    for (int i = 0; i < network->num_layers; i++) {
        free_layer(&network->layers[i]);
    }
    free(network->layers);
}

// Function to print a layer
void print_layer(Layer *layer) {
    printf("Weights:\n");
    printf("Weights rows: %d\n", layer->weights.rows);
    printf("Weights cols: %d\n", layer->weights.cols);
    print_matrix(&layer->weights);
    printf("Biases:\n");
    printf("Biases rows: %d\n", layer->biases.rows);
    printf("Biases cols: %d\n", layer->biases.cols);
    print_matrix(&layer->biases);
}

// Function to print the entire network
void print_network(Network *network) {
    for (int i = 0; i < network->num_layers; i++) {
        printf("Layer %d:\n", i + 1);
        print_layer(&network->layers[i]);
    }
}

Matrix forward_pass_layer(Layer *layer, Matrix *input) {
    Matrix result = matrix_multiply(&layer->weights, input);
    matrix_add_inplace(&result, &layer->biases);
    activation_inplace(&result);
    return result;
}

ForwardPassResult forward_pass(Network *network, Matrix *input_layer) {
    ForwardPassResult result;
    result.num_activations = network->num_layers + 1;  // +1 for input layer
    result.activations = malloc(result.num_activations * sizeof(Matrix));
    
    // Store input as first activation
    result.activations[0] = copy_matrix(input_layer);

    Matrix current_input = copy_matrix(input_layer);

    for (int i = 0; i < network->num_layers; i++) {
        Matrix layer_output = forward_pass_layer(&network->layers[i], &current_input);
        
        // Store the activation
        result.activations[i + 1] = copy_matrix(&layer_output);

        // Free the previous input and update for next iteration
        free_matrix(&current_input);
        current_input = layer_output;
    }

    // The last activation is also the final output
    return result;
}

// Don't forget to add a function to free the ForwardPassResult
void free_forward_pass_result(ForwardPassResult *result) {
    for (int i = 0; i < result->num_activations; i++) {
        free_matrix(&result->activations[i]);
    }
    free(result->activations);
}