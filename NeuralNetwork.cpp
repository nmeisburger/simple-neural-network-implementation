#include "NeuralNetwork.h"
#include "NetworkLayer.h"
#include <algorithm>

namespace std {

NeuralNetwork::NeuralNetwork(int *layer_sizes, int num_layers, float (*func)(float), float (*f_prime)(float), float N) {
	_func = func;
	_f_prime = f_prime;
	_n = N;
	_num_layers = num_layers;
	_layer_sizes = layer_sizes;
	_layers = new NetworkLayer*[num_layers];
	for (int k = 1; k < _num_layers; k++){
		_layers[k] = new NetworkLayer(_layer_sizes[k], _layer_sizes[k - 1]);
	}

}

float* NeuralNetwork::propagate (float *inputs) {
	float *current = (_layers[1])->propagate(inputs, _func);
	for (int layer = 2; layer < _num_layers; layer++) {
		float *new_current = (_layers[layer])->propagate(current, _func);
		delete[] current;
		current = new_current;
	}
	return current;
}

void NeuralNetwork::update_errors (float *outputs, float *correct_outputs) {
	(_layers[_num_layers - 1])->output_layer_error(outputs, correct_outputs, _f_prime);
	for (int layer = (_num_layers - 2); layer > 1; layer--) {
		float *error_sums = (_layers[layer + 1])->prev_layer_error_sums();
		(_layers[layer])->update_error(error_sums, _f_prime);
		delete[] error_sums;
	}
}

void NeuralNetwork::update_weights (float *inputs) {
	float* prev_layer_outputs = inputs;
	for (int layer = 1; layer < _num_layers; layer++) {
		(_layers[layer])->update_weights(prev_layer_outputs, _n);
		float* new_prev_layer_outputs = (_layers[layer])->get_outputs(_func);
		delete[] prev_layer_outputs;
		prev_layer_outputs = new_prev_layer_outputs;
	}
}

void NeuralNetwork::back_propagation (float *inputs, float *correct_outputs) {
	float *outputs = this->propagate(inputs);
	this->update_errors(outputs, correct_outputs);
	this->update_weights(inputs);
}

NeuralNetwork::~NeuralNetwork() {
	delete[] _layer_sizes;
	for (int k = 1; k < _num_layers; k++) {
		delete _layers[k];
	}
	delete[] _layers;
}

}
