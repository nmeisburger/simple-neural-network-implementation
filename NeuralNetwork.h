#include "NetworkLayer.h"
#include <iostream>

#ifndef NEURALNETWORK_H_
#define NEURALNETWORK_H_

namespace std {

class NeuralNetwork {
private:
	float (*_func)(float);
	float (*_f_prime)(float);
	int _num_layers;
	int* _layer_sizes;
	NetworkLayer **_layers;
	float _n;

public:
	NeuralNetwork(int *layer_sizes, int num_layers, float (*func)(float), float (*f_prime)(float), float N);

	float* propagate (float *inputs);

	void update_errors (float *outputs, float *correct_outputs);

	void update_weights (float *inputs);

	void back_propagation (float *inputs, float *correct_outputs);

	~NeuralNetwork();
};

}

#endif
