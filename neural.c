#include "neural.h"

double squared_error(double y_hat, double y) {
   return pow(y_hat - y, 2);
}


// For one training sample this can calculate the cost function between
// the true matrix and the final activation output of the last layer.
Matrix compute_cost_matrix(Matrix *predicted, Matrix *actual) {
    if(predicted->rows != actual->rows){
        printf("ERROR: Line 12 Dimension Mismatch!\n");
        printf("Matrix A rows and cols:\n");
        printf("Rows %d\n", predicted->rows);
        printf("Cols %d\n", predicted->cols);
        //print_matrix(predicted);
        printf("Matrix B:\n");
        printf("Rows %d\n", actual->rows);
        printf("Cols %d\n", actual->cols);
        //print_matrix(actual);
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
    printf("Cost:\n");
    print_matrix(&temp_cost_matrix); 
    matrix_add_inplace(sum, &temp_cost_matrix);
    printf("Running Cost Sum\n");
    print_matrix(sum);
    free_matrix(&temp_cost_matrix);
}

Matrix compute_batch_cost_matrix(Batch *batch){
    Matrix cost_sum = init_matrix_value(batch->truths[0].rows, 1, 0);
    int output_layer_number = batch->forwardpasses[0].num_activations - 1;
    printf("output layer number %d\n", output_layer_number);
    for(int i = 0; i < batch->batch_size; i++) {
        printf("Iteration of cost sum: %d\n", i);
        compute_cost_matrix_sum(&cost_sum, &batch->forwardpasses[i].activations[output_layer_number], &batch->truths[i]);
    }
    return cost_sum;
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
    network.layer_error = malloc(num_layers * sizeof(Matrix));

    for (int i = 0; i < num_layers; i++) {
        int neurons = layer_sizes[i];
        int connections = (i == 0) ? input_layer_size : layer_sizes[i - 1];  // Connections i.e wieghts and bias matrices are neurons[i-1]
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

Matrix calculate_z(Layer *layer, Matrix *input) {
    Matrix result = matrix_multiply(&layer->weights, input);
    matrix_add_inplace(&result, &layer->biases);
    return result;
}



SampleResult forward_pass(Network *network, Matrix *input_layer) {
    SampleResult result;
    result.num_activations = network->num_layers;   
    result.activations = malloc(result.num_activations * sizeof(Matrix));
    result.z_values = malloc(result.num_activations * sizeof(Matrix));
    result.errors = malloc(result.num_activations * sizeof(Matrix));


    Matrix current_input = copy_matrix(input_layer);

    for (int i = 0; i < network->num_layers; i++) {
        Matrix z = calculate_z(&network->layers[i], &current_input);
        Matrix a = activation(&z); 

        // Store the z value and activation
        result.activations[i + 1] = copy_matrix(&a);
        result.z_values[i + 1] = copy_matrix(&z);
        // Free the previous input and update for next iteration
        free_matrix(&current_input);
        current_input = a;
    }

    // The last activation is also the final output
    return result;
}

Batch forward_pass_batch(Network *network, double data_image[][IMAGE_SIZE], int labels[NUM_TRAIN], int batch_size, int num_samples) {
    Batch batch;
    batch.batch_size = batch_size;
    batch.forwardpasses = malloc(batch.batch_size * sizeof(ForwardPassResult));
    batch.truths = malloc(batch.batch_size * sizeof(Matrix));
    batch.costs = malloc(batch.batch_size * sizeof(Matrix));
    int output_layer_number = network->num_layers;
    
    int current_index = 0;
    for(int i = 0; i < batch.batch_size; i++) {
        printf("Batch %d\n", i);
        Matrix input = extract_next_image(data_image, &current_index, num_samples);
        printf("Input Layer Generated\n");
        batch.forwardpasses[i] = forward_pass(network, &input);
        printf("Forward Pass Calculated\n");
        batch.truths[i] = init_truth_matrix(labels, &current_index);
        printf("Truth Matrix Generated\n");
        batch.costs[i] = compute_cost_matrix(&batch.forwardpasses[i].activations[output_layer_number], &batch.truths[i]);
        printf("Cost Matrix Calculated\n");
        free_matrix(&input);
    }

    return batch;
}

// Don't forget to add a function to free the ForwardPassResult
void free_forward_pass_result(ForwardPassResult *result) {
    for (int i = 0; i < result->num_activations; i++) {
        free_matrix(&result->activations[i]);
    }
    free(result->activations);
}

Matrix calculate_output_layer_error(Matrix *cost_matrix, Matrix *current_z_value) {
    Matrix layer_error = matrix_multiply(cost_matrix, current_z_value);
    return layer_error;
}


Matrix calculate_hidden_layer_error(Matrix *current_z_value, Matrix *next_cost, Matrix *next_layer_weights) {
    Matrix backwards_weights = matrix_multiply(next_layer_weights, next_cost);
    Matrix sigmoid_prime_z_value = sigmoid_derivative(current_z_value);
    Matrix layer_error = matrix_multiply(&backwards_weights, &sigmoid_prime_z_value);

    free_matrix(&backwards_weights);
    free_matrix(&sigmoid_prime_z_value);
    return layer_error;
}   

void backwards_pass_network(Matrix *network, SampleResult *sample_result, Matrix *cost_matrix) {
    int num_layers = network->num_layers;
    Matrix output_layer_error = calculate_output_layer_error(cost_matrix, sample_result->z_values[num_layers-1]);
    sample_result->errors[num_layers-1] = copy_matrix(&output_layer_error);
    free_matrix(output_layer_error);
    
    for(int i = num_layers-2; i >= 0; i--){
        Matrix hidden_layer_error = calculate_hidden_layer_error(sample_result->z_values[i], sample_result->activations[i+1], network->layers[i+1].weights);
        sample_result->errors[i] = copy_matrix(&hidden_layer_error);
        free_matrix(&hidden_layer_error);
    } 
}
