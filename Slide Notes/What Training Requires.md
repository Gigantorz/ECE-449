There are three approaches to learning in a neural network
![[Pasted image 20231010173135.png]]
**Supervised Learning**
- we have an exact "ground truth" output for every object
- In some fashion, the desired output is determined for every feature vector 
	- in this diagram, that's the job of the teacher.
- That desired response is compared to the actual response learned so far in the AI system, producing an error signal.
	- We feed the error signal back to the learner so that it can improve its performance (reduce the error)
Finding an effective "teacher" or source of the ground truth, is often very difficult. 
- key part of data engineering.

**Reinforcement Learning**
![[Pasted image 20231010173420.png]]
In reinforcement learning, the learning system is active on the environment observing the results and receiving positive or negative feedback from a "critic."
- The critic is some form of assessment of the AI system's performance.
- The critic cannot give an explicit error signal, only a "good" or "bad" response. The goal here is minimize "bad" feedback, and maximize "good" feedback. 
	- This is commonly used in robotics.

Reinforcement is complicated. The state of the AI system's environment is an input to the AI system, but the AI is directly modifying it through the actions it takes
- there will be a limited set of actions that the AI is capable of, but their impact on the environment will change as the environment changes. The AI has to explore the combinations of action-state pairs to learn how best to manipulate the environment.
- Something in the environment, some critical variables, will be observed by the critic. 
	- That's the primary reinforcement signal. 
		- this is turned into a reward or penalty by the critic, and passed to the AI
			- remember, that means that there's delay between taking an action and seeing the reward or penalty.


**Unsupervised Learning**
![[Pasted image 20231010174141.png]]
The learning system receives no feedback at all. 
- essentially clustering:
	- The learning system can do no more than identify the spatial distribution of feature vectors.

Finding groups of feature vectors is very often useful; remember, we are assuming that these are objects that are relatively similar to one another.

Some deep learning algorithms train neural networks using unsupervised learning paired with information theory;
- graduate class on neural networks.

### But also
**Preprocessing**
Feature vectors don't just happen, they have to be computed.
	If this data is coming from a database, then an entire Extract, Transform, Load process might need to be invoked.
	- or the outputs of multiple sensors may need to be captured, formatted and ingested into the AI system

Data in real world is noisy. 
- might also be corrupted by processing errors or hardware malfunctions in your system.
	- These have to be removed, and any missing values must be replaced. 
		- "Data Cleaning"
			- AI algorithms don't do well when confronted with the dirt and mess of real life
				- so AI system must scrub the data clean first.

You may want to compute some useful values from your measured features, and add them to simplify things for your AI system. 

Redundancy from
- Multicollinearity 
	- when multiple features are linear functions of one another.
	- is the worst case, but even lower levels of redundancy are unhelpful to an AI.
Feature Selection or reduction 
- are the techniques for removing this redundancy, without losing useful information that your AI needs to learn from the data.
- Also helps speed up the AI
	- because most algorithms have poor time and space complexity growth relative to the size of their feature vectors, 
		- so having fewer features makes for much faster learning
