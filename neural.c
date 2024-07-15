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
    int output_layer_number = batch->sample_results[0].num_activations - 1;
    printf("output layer number %d\n", output_layer_number);
    for(int i = 0; i < batch->batch_size; i++) {
        printf("Iteration of cost sum: %d\n", i);
        compute_cost_matrix_sum(&cost_sum, &batch->sample_results[i].activations[output_layer_number], &batch->truths[i]);
    }
    return cost_sum;
}

Matrix init_weights_layer(int neurons, int weights) {
    Matrix layer = init_matrix(neurons, weights);
    double scale = sqrt(2.0 / (neurons + weights)); // Xavier/Glorot initialization
    for (int i = 0; i < neurons; i++) {
        for (int j = 0; j < weights; j++) {
            layer.data[i][j] = ((double)rand() / RAND_MAX) * 2 * scale - scale;
        }
    }
    return layer;
}

Matrix init_truth_matrix(int *train_label, int *current_index) {
    Matrix truth = init_matrix_value(10, 1, 0);

    truth.data[train_label[*current_index]][0] = 1; 
    
    return truth;
}

Matrix init_bias_layer(int neurons, int biases) {
    Matrix layer = init_matrix_value(neurons, biases, 0.1); 
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
    network.layer_sizes = layer_sizes;
    network.layers = (Layer *)malloc(num_layers * sizeof(Layer));
    network.layer_error = malloc(num_layers * sizeof(Matrix));

    for (int i = 0; i < num_layers; i++) {
        int neurons = layer_sizes[i];
        int connections = (i == 0) ? input_layer_size : layer_sizes[i - 1];
        network.layers[i] = init_layer(neurons, connections);
        
        // Print the dimensions of each layer for debugging
        printf("Layer %d: %d neurons, %d connections\n", i+1, neurons, connections);
        printf("Weight matrix dimensions: %d x %d\n", network.layers[i].weights.rows, network.layers[i].weights.cols);
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

void normalise_input(Matrix *input) {
    for (int i = 0; i < input->rows; i++) {
        for (int j = 0; j < input->cols; j++) {
            input->data[i][j] = 0.1 + (input->data[i][j] * 0.8);
        }
    }
}

SampleResult forward_pass(Network *network, Matrix *input_layer) {
    SampleResult result;
    result.input_layer = copy_matrix(input_layer);
    result.num_activations = network->num_layers;
    result.activations = malloc(result.num_activations * sizeof(Matrix));
    result.z_values = malloc(result.num_activations * sizeof(Matrix));
    result.errors = malloc(result.num_activations * sizeof(Matrix));

    Matrix current_input = copy_matrix(input_layer);

    for (int i = 0; i < network->num_layers; i++) {
        Matrix z = calculate_z(&network->layers[i], &current_input);
        Matrix a = activation(&z); 

        // Store the z value and activation
        result.activations[i] = copy_matrix(&a);
        result.z_values[i] = copy_matrix(&z);
        //printf("Activation %d (rows: %d, cols: %d)\n", i, result.activations[i].rows, result.activations[i].cols);
        // Free the previous input and update for next iteration
        if (i > 0) {
            free_matrix(&current_input);
        }
        free_matrix(&z);
        
        current_input = a;  // Transfer ownership
    }

    // Free the last input, which was the output of the last layer
    free_matrix(&current_input);

    return result;
}

Batch init_batch(Network *network, int batch_size) {
    Batch batch;
    batch.batch_size = batch_size;
    batch.sample_results = malloc(batch.batch_size * sizeof(SampleResult));
    batch.truths = malloc(batch.batch_size * sizeof(Matrix));
    batch.costs = malloc(batch.batch_size * sizeof(Matrix));
    batch.batch_layer_error = malloc(network->num_layers * sizeof(Matrix));
    batch.weights_layer_error = malloc(network->num_layers * sizeof(Matrix));

    for(int i = 0; i < network->num_layers; i++) {
        batch.batch_layer_error[i] = init_matrix_value(network->layer_sizes[i], 1, 0);
        batch.weights_layer_error[i] = init_matrix_value(network->layers[i].weights.rows, network->layers[i].weights.cols, 0);
    }

    return batch;
}

void forward_pass_batch(Network *network, Batch *batch, double data_image[][IMAGE_SIZE], int labels[NUM_TRAIN], int num_samples, int batch_number) {
    int output_layer_number = network->num_layers - 1;  // Adjust this if necessary
    
    int current_index = 0;
    for(int i = 0; i < batch->batch_size; i++) {
        current_index = batch_number*batch->batch_size + i;
        //printf("Sample %d (current_index: %d)\n", i, current_index);
        if (batch_number*batch->batch_size >= num_samples) {
            fprintf(stderr, "Error: Exceeded number of samples\n");
            exit(1);
        }
        Matrix input = extract_next_image(data_image, &current_index, num_samples);
        normalise_input(&input);
        //print_mnist_image(data_image, current_index);
        //printf("Input Layer Generated (rows: %d, cols: %d)\n", input.rows, input.cols);
        batch->sample_results[i] = forward_pass(network, &input);
        //printf("Forward Pass Calculated\n");
        batch->truths[i] = init_truth_matrix(labels, &current_index);
        //printf("Truth Matrix Generated (rows: %d, cols: %d)\n", batch->truths[i].rows, batch->truths[i].cols);
       // if (output_layer_number >= batch.sample_results[i].num_activations) {
        //    fprintf(stderr, "Error: Invalid output layer number\n");
        //    exit(1);
       // }
        batch->costs[i] = compute_cost_matrix(&batch->sample_results[i].activations[output_layer_number], &batch->truths[i]);
        //printf("Cost Matrix Calculated (rows: %d, cols: %d)\n", batch->costs[i].rows, batch->costs[i].cols);
        free_matrix(&input);
    }
}

// Don't forget to add a function to free the ForwardPassResult
void free_forward_pass_result(SampleResult *result) {
    for (int i = 0; i < result->num_activations; i++) {
        free_matrix(&result->activations[i]);
    }
    free(result->activations);
}

void free_sample_result(SampleResult *sample_result) {
    if (sample_result == NULL) return;

    // Free input layer
    free_matrix(&sample_result->input_layer);

    // Free z_values
    if (sample_result->z_values != NULL) {
        for (int i = 0; i < sample_result->num_activations; i++) {
            free_matrix(&sample_result->z_values[i]);
        }
        free(sample_result->z_values);
    }

    // Free activations
    if (sample_result->activations != NULL) {
        for (int i = 0; i < sample_result->num_activations; i++) {
            free_matrix(&sample_result->activations[i]);
        }
        free(sample_result->activations);
    }

    // Free errors
    if (sample_result->errors != NULL) {
        for (int i = 0; i < sample_result->num_activations; i++) {
            free_matrix(&sample_result->errors[i]);
        }
        free(sample_result->errors);
    }
}

void free_batch(Batch *batch, int num_layers) {
    if (batch == NULL) return;

    // Free sample results
    if (batch->sample_results != NULL) {
        for (int i = 0; i < batch->batch_size; i++) {
            free_sample_result(&batch->sample_results[i]);
        }
        free(batch->sample_results);
    }

    // Free costs
    if (batch->costs != NULL) {
        for (int i = 0; i < batch->batch_size; i++) {
            free_matrix(&batch->costs[i]);
        }
        free(batch->costs);
    }

    // Free truths
    if (batch->truths != NULL) {
        for (int i = 0; i < batch->batch_size; i++) {
            free_matrix(&batch->truths[i]);
        }
        free(batch->truths);
    }

    // Free batch layer error
    if (batch->batch_layer_error != NULL) {
        for (int i = 0; i < num_layers; i++) {
            free_matrix(&batch->batch_layer_error[i]);
        }
        free(batch->batch_layer_error);
    }

    // Free weights layer error
    if (batch->weights_layer_error != NULL) {
        for (int i = 0; i < num_layers; i++) {
            free_matrix(&batch->weights_layer_error[i]);
        }
        free(batch->weights_layer_error);
    }
}

Matrix calculate_output_layer_error(Matrix *cost_matrix, Matrix *current_z_value) {
    printf("Z(L)0 Neuron 1: %f\n", current_z_value->data[0][0]);
    Matrix sigmoid_prime_z_value = derivative_activation(current_z_value);
    printf("sigmoid prime(Z(L)0) Neuron 1: %f\n", sigmoid_prime_z_value.data[0][0]);
    printf("cost(L)0 Neuron 1: %f\n", cost_matrix->data[0][0]);
    Matrix layer_error = element_wise(cost_matrix, &sigmoid_prime_z_value);
    printf("Output Layer Error Neuron 1: %f\n", layer_error.data[0][0]);
    free_matrix(&sigmoid_prime_z_value);
    return layer_error;
}

Matrix calculate_hidden_layer_error(Matrix *current_z_value, Matrix *next_layer_error, Matrix *next_layer_weights) {
    
    Matrix transposed_next_layer_weights = transpose_matrix(next_layer_weights);
    printf("W(L+1)0.T Neuron 1: %f\n", transposed_next_layer_weights.data[0][0]);
    printf("Error(L+1)0 Neuron 1: %f\n", next_layer_error->data[0][0]);

    Matrix backwards_error = matrix_multiply(&transposed_next_layer_weights, next_layer_error);
    printf("W(L+1)0.T * Error(L+1)0 Neuron 1: %f\n", backwards_error.data[0][0]);
    printf("Z(L)0 Neuron 1: %f\n", current_z_value->data[0][0]);
    Matrix sigmoid_prime_z_value = derivative_activation(current_z_value);
    printf("sigmoid prime(Z(L)0) Neuron 1: %f\n", sigmoid_prime_z_value.data[0][0]);
    Matrix layer_error = element_wise(&backwards_error, &sigmoid_prime_z_value);
    printf("Hidden Layer Error Neuron 1: %f\n", layer_error.data[0][0]);

    free_matrix(&backwards_error);
    free_matrix(&sigmoid_prime_z_value);
    free_matrix(&transposed_next_layer_weights);
    
    return layer_error;
}

void backwards_pass_network(Network *network, SampleResult *sample_result, Matrix *cost_matrix) {
    int num_layers = network->num_layers;
    printf("\nOutput Layer:\n");
    Matrix output_layer_error = calculate_output_layer_error(cost_matrix, &sample_result->z_values[num_layers-1]);
    sample_result->errors[num_layers-1] = copy_matrix(&output_layer_error);
    free_matrix(&output_layer_error);
    
    for(int i = num_layers-2; i >= 0; i--) {
        printf("\nHidden Layer %d\n", i);
        Matrix hidden_layer_error = calculate_hidden_layer_error(&sample_result->z_values[i], &sample_result->errors[i+1], &network->layers[i+1].weights);
        sample_result->errors[i] = copy_matrix(&hidden_layer_error);
        free_matrix(&hidden_layer_error);
    } 
}

void backwards_pass_batch(Network *network, Batch *batch) {
    if(network->num_layers != batch->sample_results[0].num_activations) {
        printf("ERROR: Num Activations to Num Layers don't match.\n");
        exit(1);
    }

    for(int i = 0; i < batch->batch_size; i++) {
        // Compute errors for this sample
        backwards_pass_network(network, &batch->sample_results[i], &batch->costs[i]);

        // Accumulate errors and weight gradients
        for(int j = 0; j < network->num_layers; j++) {
            printf("Layer: %d\n", j);
            printf("Error Total Neuron 1: %f\n", batch->batch_layer_error[j].data[0][0]);
            printf("Error Batch %d Neuron 1: %f\n", i, batch->sample_results[i].errors[j].data[0][0]);
            // Accumulate layer errors
            matrix_add_inplace(&batch->batch_layer_error[j], &batch->sample_results[i].errors[j]);
            printf("Error Total Neuron 1: %f\n", batch->batch_layer_error[j].data[0][0]); 
            // Compute and accumulate weight gradients
            Matrix transposed_activation;
            if (j == 0) {
                transposed_activation = transpose_matrix(&batch->sample_results[i].input_layer);
            } else {
                transposed_activation = transpose_matrix(&batch->sample_results[i].activations[j-1]);
            }
            printf("\na(L-1).T Neuron 1: %f\n", transposed_activation.data[0][0]);
            int zero_count = 0;
            int total_elements = transposed_activation.rows * transposed_activation.cols;

            for (int i = 0; i < transposed_activation.rows; i++) {
                for (int j = 0; j < transposed_activation.cols; j++) {
                    if (transposed_activation.data[i][j] == 0.0) {
                        zero_count++;
                    }
                 }
            }

            printf("Number of zero elements in a(L-1).T: %d out of %d\n", zero_count, total_elements);
            printf("Percentage of zero elements: %.2f%%\n", (float)zero_count / total_elements * 100);
            
            printf("Error Neuron 1: %f\n", batch->sample_results[i].errors[j].data[0][0]);
            Matrix weights_error = matrix_multiply(&batch->sample_results[i].errors[j], &transposed_activation);
            printf("dW(L) Neuron 1: %f\n", weights_error.data[0][0]);
            printf("dW Total(L): %f\n", batch->weights_layer_error[j].data[0][0]);
            matrix_add_inplace(&batch->weights_layer_error[j], &weights_error);
            printf("dW Total(L) + dW(L): %f\n", batch->weights_layer_error[j].data[0][0]);
            free_matrix(&weights_error);
            free_matrix(&transposed_activation);
        }
    }

    // Average the accumulated errors and gradients
    for(int k = 0; k < network->num_layers; k++) {
        matrix_scalar_divide_inplace(&batch->batch_layer_error[k], batch->batch_size);
        printf("\ndW Total(%d) Before Average Neuron 1: %f\n", k, batch->weights_layer_error[k].data[0][0]); 
        matrix_scalar_divide_inplace(&batch->weights_layer_error[k], batch->batch_size);
        printf("dW Total(%d)/%d: %f\n", k, batch->batch_size, batch->weights_layer_error[k].data[0][0]);
    }
}

void adjust_weights_and_biases(Network *network, Batch *batch, double learning_rate) {
    for(int i = 0; i < network->num_layers; i++) {
       // printf("Layer %d:\n", i+1);
        
        Matrix weights_error_step = matrix_scalar_multiply(&batch->weights_layer_error[i], learning_rate);
        matrix_scalar_multiply_inplace(&weights_error_step, -1);
        
        // Print the same sample of weight updates
       // printf("Sample weight updates:\n");
        //for (int j = 0; j < 5 && j < weights_error_step.rows; j++) {
         //   for (int k = 0; k < 5 && k < weights_error_step.cols; k++) {
          //      printf("%lf ", weights_error_step.data[j][k]);
          //  }
         //   printf("\n");
       // }

        // Adjusting Weights
        matrix_add_inplace(&network->layers[i].weights, &weights_error_step);
        free_matrix(&weights_error_step);

        Matrix bias_error_step = matrix_scalar_multiply(&batch->batch_layer_error[i], learning_rate);
        matrix_scalar_multiply_inplace(&bias_error_step, -1);
        matrix_add_inplace(&network->layers[i].biases, &bias_error_step);

        free_matrix(&bias_error_step);
    }
}

bool evaluate_sample_performance(SampleResult *sample, Matrix *truth_matrix) {
    // Assuming the last activation is the output layer
    Matrix *output = &sample->activations[sample->num_activations - 1];
    
    // Find the predicted class (index of the maximum value in the output)
    int predicted_class = argmax(output);
    
    // Find the true class (index of the maximum value in the truth matrix)
    int true_class = argmax(truth_matrix);
    
    // Check if the predicted class matches the true class
    return predicted_class == true_class;
}

double evaluate_batch_performance(Batch *batch) {
    double num_correct = 0;
    for (int i = 0; i < batch->batch_size; i++) {
        if (evaluate_sample_performance(&batch->sample_results[i], &batch->truths[i])) {
            num_correct += 1;
        }
    }
    return (num_correct / batch->batch_size) * 100.0;
}

void train_network(Network *network, int batch_size, int epochs, double learning_rate,  double data_image[][IMAGE_SIZE], int labels[NUM_TRAIN]) {
    if(NUM_TRAIN % batch_size != 0){
        printf("Dynamic mini-batching is not supported yet\n");
        printf("Please select a batch size that is a multiple of NUM_TRAIN\n");
    }

    int num_batchs = NUM_TRAIN/batch_size;

    for(int i = 0; i < epochs; i++) {
        printf("Epoch: %d\n", i);

        for(int j = 0; j < num_batchs; j++) {
            printf("Batch %d\n", j);
            Batch batch = init_batch(network, batch_size);
            forward_pass_batch(network, &batch, data_image, labels, NUM_TRAIN, j);
            backwards_pass_batch(network, &batch);

            adjust_weights_and_biases(network, &batch, learning_rate);
            double batch_accuracy = evaluate_batch_performance(&batch);
            printf("Batch %d Accuracy: %lf\n", j, batch_accuracy);
            free_batch(&batch, network->num_layers);
        }
    }
}


