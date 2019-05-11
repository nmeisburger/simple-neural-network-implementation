#include "NetworkLayer.h"
#include <iostream>

namespace std {

NetworkLayer::NetworkLayer(int num_nodes_layer, int num_nodes_prev) {
	_num_nodes = num_nodes_layer;
	_num_prev = num_nodes_prev;
	_weights = new float[num_nodes_layer * num_nodes_prev];
	for (int i = 0; i < (num_nodes_layer * num_nodes_prev); i++) {
		_weights[i] = 0.5;
	}
	_error = new float[num_nodes_layer];
	_activation = new float[num_nodes_layer];
}

float* NetworkLayer::propagate (float* input, float (*func)(float)) {
	float* output = new float[_num_nodes];
	for (int node_current = 0; node_current < _num_nodes; node_current++) {
		float node_activation = 0;
		for (int node_prev = 0; node_prev < _num_prev; node_prev++) {
			node_activation += _weights[node_current * _num_prev + node_prev] * input[node_prev];
		}
		output[node_current] = (*func)(node_activation);
		_activation[node_current] = node_activation;
	}
	return output;
}

void NetworkLayer::output_layer_error (float* outputs, float* correct_outputs, float (*f_prime)(float)) {
	for (int i = 0; i < _num_nodes; i++) {
		float temp_error = (*f_prime)(_activation[i]) * (correct_outputs[i] - outputs[i]);
		_error[i] = temp_error;
	}
}

float* NetworkLayer::prev_layer_error_sums () {
	float *prev_error_sums = new float[_num_prev];
	for (int prev_node = 0; prev_node < _num_prev; prev_node++) {
		float error_sum = 0;
		for (int next_node = 0; next_node < _num_nodes; next_node++) {
			error_sum += _error[next_node] * _weights[next_node * _num_prev + prev_node];
		}
		prev_error_sums[prev_node] = error_sum;
	}
	return prev_error_sums;
}

void NetworkLayer::update_error (float* error_sums, float (*f_prime)(float)) {
	for (int node = 0; node < _num_nodes; node++) {
		_error[node] = (*f_prime)(_activation[node]) * (error_sums[node]);
	}
}

float* NetworkLayer::get_outputs(float (*func)(float)) {
	float* outputs = new float[_num_nodes];
	for (int node = 0; node < _num_nodes; node++) {
		outputs[node] = (*func)(_activation[node]);
	}
	return outputs;
}

void NetworkLayer::update_weights (float* prev_outputs, float N) {
	for (int current_node = 0; current_node < _num_nodes; current_node++) {
		for (int prev_node = 0; prev_node < _num_prev; prev_node++) {
			_weights[current_node * _num_prev + prev_node] += N * _error[current_node] * prev_outputs[prev_node];
		}
	}
}

NetworkLayer::~NetworkLayer() {
	delete[] _weights;
	delete[] _error;
	delete[] _activation;
}

}
