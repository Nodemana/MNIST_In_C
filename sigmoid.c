#include <math.h>
#include "sigmoid.h"

double sigmoid(double n) {
    return (1 / (1 + pow(EULER_NUMBER, -n)));
}

double derivative_sigmoid(double n){
    return sigmoid(n) * (1 - sigmoid(n));
}

float sigmoidf(float n) {
    return (1 / (1 + powf(EULER_NUMBER_F, -n)));
}

long double sigmoidl(long double n) {
    return (1 / (1 + powl(EULER_NUMBER_L, -n)));
}
