![[Pasted image 20231010162022.png]]

**Data Ingestion**
- Data may be coming in from many sources:
	- sensors, point-of-sale systems, historical database,
- Many different data types possible:
	- numeric, text, images, videos
- Sample instants might not be coordinated
How do you synchronize these observations? 
	The sampling rates may be different; and even if they're the same, they might be offset from one another.
		Maybe to a significant degree, maybe not; but you have to figure that out.

**Data Transformation**
What formats can your ML model accept?
- most just takes columns of data.
	Deep Learning models 
	- can accept raw images, text and video if designed correctly.
You can combine all the different data types together using [[Data Fusion]]
Normalization
- means that each feature needs to have an equal chance to contribute to the learning process.
	- Any time you directly use prediction errors in learning (like in all NN models), if the amplitude (across the dataset) of some features is much smaller, then they will have a smaller chance to affect the learning outcome.

Derived (computed) features
- you can sometimes compute useful artificial features based on the measured ones.

**Feature Engineering**
Missing Data Imputation
- some feature vectors will be missing some features
- it happens
- for most algorithms, those missing feature values need to be replaced
- An algo is required
	- simply copy the last observed value
	- take an average 
	- use an ML algorithm to predict the missing value
		- these are design decisions that have to be made by the ML pipeline engineer.

**Model Training**
Experimental Design and Execution
- Production pipelines also never consider a model to be "final"; 
	- as new data comes in, the model will need to be updated.
		- That means more experiments to be run, and again we can't wait for a human to setup a new experiment. 
			- This must be automated as well, but there's a lot of complexity in experiment design

Cross-Validation
How are you partitioning your data into train and test sets?
- are you running a cross-validation design or single splits?

Hyper-parameter Exploration
What hyper-parameters for your model are you exploring? 
- This is the trial-and-error part of training, and it can be automated.

Evaluation metrics
The Decision about how "good" an experimental result can be automated, but it is more than measuring accuracy on a test set. -
- Your evaluation metrics have to be carefully chosen, and (importantly) they have to be congruent to the purposes of your training.

Metadata retention for replication
All the meta-data about the model you are training, and the data you're training it on, needs to be recorded and saved so that you can replicate this specific experiment when needed.

**Model Validation**
- Automatically determining whether the learned model meets the engineering requirements for the AI system
- Usually we test the model on a test set wholly separate from the training data.
	- In a running pipeline, this would mean runnign a new version in parallel with the deployed version, and comparing their results
		- cut in the new version when it clearly outperforms the old one.
	- on initial standup, cut in the first model when the performance reaches a defined threshold

Covariate Shift!
- means that the probability distribution of feature vectors in your training data is not the same as in your test data.
	- This always means that your out-of-sample performance is poorer than you expected
For an AI pipeline, this means your training set can age and become irrelevant.
	Thus, you will need to constantly refresh it. 

**Model Deployment**
![[Pasted image 20231010201842.png]]
On the left 
- traditional approach to running a program
	- an app interfaces with the operating system to access the filesystem, inter-process communcation, and hardware systems like external disk drives or fingerprint readers
		- however, this means that the app can only run on one operating system, because each different OS has a different set of commands and libraries to acecss.

On the middle
- Java, Python, Ruby and many other modern languages are portable; 
	- you write a program once and runs on many different operating systems. 
		- The way to do this is to add another layer:
			- a "virtual machine"
				- that presents the app with the illusion of a computer that runs Java (specifically Java bytecode) as its basic language.
					- There's different virtual machine for each different operating system, but they all look exactly the same to the apps
	- Works very well;
		- Python is the most widely used programming language, 
			- but the virtual machine is a huge program, pretty much the size of an actual operating system. And you need one for each language you might want to run on that computer.

On the right
Docker popularized an alternative:
- package an app with everything (libraries, configuration files) it will need in order to run, and then have a lightweight virtual machine (called a runtime) that can run any language.
	- The packaged app is called a container
	- The container runtime is a very simple virtual machine with a pretty limited set of capabilities. Everything you would have called on the VM or the OS for is packed in the container itself
	- Modern AI pipelines deploy models as containers.

