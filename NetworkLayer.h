#ifndef NETWORKLAYER_H_
#define NETWORKLAYER_H_

namespace std {

class NetworkLayer {
private:
	int _num_nodes, _num_prev;
	float *_weights;
	float *_error;
	float *_activation;
public:
	NetworkLayer(int num_nodes_layer, int num_nodes_prev);

	float* propagate (float* input, float (*func)(float));

	void output_layer_error (float* outputs, float* correct_outputs, float (*f_prime)(float));

	float* prev_layer_error_sums ();

	void update_error (float* error_sums, float (*f_prime)(float));

	float* get_outputs(float (*func)(float));

	void update_weights (float* prev_outputs, float N);

	~NetworkLayer();
};

}

#endif
