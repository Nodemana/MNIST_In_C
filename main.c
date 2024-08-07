#include "sigmoid.h"
#include "matrixmath.h"
#include "neural.h"

int main() {
    srand(time(0) + getpid());
    printf("Starting main function\n");
    load_mnist();
    printf("MNIST data loaded\n");

    //int current_index = 1;
    //Matrix input = extract_next_image(train_image, &current_index, 2);
    //Matrix truth = init_truth_matrix(train_label, &current_index);
    //printf("Label: %d \n", train_label[current_index]);
    //printf("Image extracted\n");

    int num_layers = 2;
    int layer_sizes[] = {100, 10};
    int input_layer_size = 784;

    Network network = init_network(num_layers, layer_sizes, input_layer_size);
    //for(int i = 0; i < network.num_layers; i++) {
    //    print_layer(&network.layers[i]);
    //}
    //train_network(Network *network, int batch_size, int epochs, double learning_rate,  double data_image[][IMAGE_SIZE], int labels[NUM_TRAIN])
    train_network(&network, 1, 100, 0.01, train_image, train_label);
    
    free_network(&network);

    return 0;
}

