#include <iostream>
using namespace std;

#include "NetworkLayer.h"
#include "NeuralNetwork.h"

float f (float x) {
	return x;
}

float f_prime (float x) {
	return 1;
}

int main(){
	int *test_layers = new int[2];
	test_layers[0] = 2;
	test_layers[1] = 2;

	NeuralNetwork test_network(test_layers, 2, f, f_prime, 1);

	float *test_inputs = new float[2];
	test_inputs[0] = 2;
	test_inputs[1] = 1;

	float *test_correct_outputs = new float[2];
	test_correct_outputs[0] = 2;
	test_correct_outputs[1] = 2;

	float *test_outputs = test_network.propagate(test_inputs);

	for (int i = 0; i < 2; i++) {
			cout << test_outputs[i] << endl;
		}

	test_network.back_propagation(test_inputs, test_correct_outputs);

	float *test_inputs2 = new float[2];
	test_inputs2[0] = 2;
	test_inputs2[1] = 1;

	float* new_test_outputs = test_network.propagate(test_inputs2);

	for (int i = 0; i < 2; i++) {
		cout << new_test_outputs[i] << endl;
	}

	delete[] test_inputs2;
	delete[] test_outputs;
	return 0;
}
