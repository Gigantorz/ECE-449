  - introduces non-linearity by replacing all negative values in the output with zero.
    This helps the network learn complex patterns in the data.

Most commonly used activation function in deep learning models. More computationally efficient, making it a popular choice. 
Mitigates the vanishing gradient problem because ReLU only sets negative values to 0, while leaving positive values unchanged, allowing for better propagation of gradients.

Drawback
- can suffer from the "dead neuron" problem, 
	- where neurons can become inactive and produce 0 for every input, resulting in a loss of learning.
		- To address this issue, variants of ReLU, such as Leaky ReLU and Parametric ReLU, have been developed to provide a small slope for negative inputs, preventing neurons from becoming completely inactive.\