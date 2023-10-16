![[Pasted image 20231010180402.png]]
Sensor-level fusion: 
- combine information from multiple sensors, usually of the same type, to get a more accurate, noise-free measurement. That measurement is passed on, in our case to an AI.
	• E.g. average several thermometer readings in an area to eliminate biases from shadowing and reflected sunlight.
- all recording the same thing with the same attributes from different environments

Feature-level fusion: 
- first, the feature vectors from multiple sensors (which might be different) are each extracted, and then concatenated into a single vector representing the observed object. The concatenated vector is passed to the AI.
	• E.g. in a recent project of mine, we take the sampled voltage and currents from all three power phases in an electric motor and concatenate them into a six-attribute feature vector. We then look for fluctuations that indicate the motor is damaged; it’s called condition monitoring, and it’s a major application of AI in industry.
- all recording various things from different environments and collecting many variables then all added together to form a dataset for the AI.
	
Decision-level fusion:
- The feature vector from each sensor is passed to a first-stage AI. The outputs of each individual AI are then passed to a second-level AI that makes a final decision.
	• E.g. 1st-stage facial recognition is useful in itself, but if you have a suspicious mind you could pass individual recognitions on to another classifier that looks for known groups of associates. This would not be favorably received in North America, but I would bet money that the Chinese security agencies already have it.
- sensors are acting like feature-level where they are recording different things, but then instead of being concatenated they are being passed into an AI then passed to an output AI.