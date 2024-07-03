#include "sigmoid.h"
#include "matrixmath.h"
#include "neural.h"

int main() {
    srand(time(0) + getpid());
    printf("Starting main function\n");
    load_mnist();
    printf("MNIST data loaded\n");

    int current_index = 1;
    Matrix input = extract_next_image(train_image, &current_index, 2);
    Matrix truth = init_truth_matrix(train_label, &current_index);
    printf("Label: %d \n", train_label[current_index]);
    printf("Image extracted\n");

    int num_layers = 3;
    int layer_sizes[] = {16, 16, 10};
    int input_layer_size = 784;

    Network network = init_network(num_layers, layer_sizes, input_layer_size);
    Batch batch = forward_pass_batch(&network, train_image, train_label, 10, NUM_TRAIN);
    printf("Final activation dimensions: rows = %d, cols = %d\n", 
        batch.forwardpasses[0].activations[batch.forwardpasses[0].num_activations - 1].rows,
        batch.forwardpasses[0].activations[batch.forwardpasses[0].num_activations - 1].cols);
    printf("Truth matrix dimensions: rows = %d, cols = %d\n", 
        batch.truths[0].rows, batch.truths[0].cols);
    
    Matrix batch_cost = compute_batch_cost_matrix(&batch);
    printf("Batch Cost Matrix:\n");
    print_matrix(&batch_cost);

    free_network(&network);
    free_matrix(&batch_cost);

    return 0;
}

