**Dataset requirements**
- what will the data represent?
- what exactly are the input and output domains supposed to be?
- How are the data distributed through these domains?
- What do outliers look like, and how should they be treated?
- What fairness and ethical concerns must be considered in this domain

---
- [ ] Read and make notes

- the requirements for our datasets will be related to the requirements for the AI system being created, but dataset requirements refine the AI system requirements in a very particular way.

- The ML algorithm must be fed samples of input-output relationships that are created by completing instances of the task, and the ML algorithms will induce a faithful model of the task from these sample. So we need data.
	- Thus we need a first clear statement of what task is to be accomplished. (GOAL)
		- However, AI requirements are normally incomplete and/or vague, meaning that even this first step needs refinement
			- working with your client you will have to decide what, exactly, the scope of the ai system is
				- and how that scope is going to be realized.
					- is it a single ML algorithm?
					- is it an ensemble
					- is it some combination of parallel and sequential AI models
					- or some other possibility

- The scope for each ML algorithm defines a phenomenon to model.
- we need to ask
	- what exactly are the input and
	- output domains for this phenomenon, and 
	- how data points are distributed in these domains. 
- What rare cases may look like and how they should be treated

We also have to ask what fairness and ethical concerns must be considered in this domain, and how they affect the data distribution above.
- black men are very disproportionately represented in USA prisons.
	- it is therefore reasonable to include race in an AI that predicts recidvism?
		Recedivism - tendency of a prisoner to commit another crime after they finish their prison term
		- Most people would say no, but you won’t find that from looking at the data distribution.

**Dataset Analysis**
- What constitutes "normal" behaviour for the phenomenon being studied?
- what are the edge cases? what is an outlier?
- How should outliers be dealt with in the model?
- What constitutes high-quality or low-quality data?
- how should low-quality data be handled?
- What features of the data are essential to the phenomenon, and which are tangential?
- Which data manipulations would be acceptable, and which would not?

- most problems that are complex enough to require ML models have more than a single behaviour.
	- much of the behaviour would be considered "normal"
		- then there will be edge cases and outliers
			- then this leads into the complicated world of defining what an "outlier" is.
A defintion of **outliers** from the 19th century
> an observation so different from the ordinary for your phenomena (scenario/problem), that it seems to come from a different phenomenon altogether.
- This is very difficult to quantify.
- How are we going to handle outliers? 
	- are they just going to get ignored?
	- processed like any other data point?
	- do they require special handling?
		- every possible choice affects how you are going to design and develop your dataset.

Anomaly detection
- entire field that is about the struggle to find a quantifiable, testable way of deciding what is an outlier, and what is not.

**Quality is not simple to quantify either**
There's a whole branch of engineering called quality engineering, 
- first thing it does is define the word "quality"
	- "degree of excellence of a product when considering all relevant quality attributes"
		- treats quality as a multidimensional quality, composed of many different aspects.
saying "high-quality data" it not a simple thing to say

Data that has little measurement noise might be higher-quality than data with more measurement noise ...
- but if the data is not reliable, or does not really capture the essence of the phenomena you need to model, 
	- then it is actually low-quality no matter how exact it is.
Example:
> how do you quantify "fun" in an MMO video game? 
> 	It's plainly related to player engagement but are you only going to measure "fun" by how many players are logged in at a given time?
> 		shouldn't we measure the feeling of the player by asking them a survey of how they felt while playing the game?
> 			this is a limitation of the AI because they won't be able to tell how a player is internally feeling. 
> 				Question of consciousness and how we can't answer why people have their own personality even though theoretically we are all built the same.

- not every correlation is meaningful. Some even offend, even though they are statistically valid.

•    The textbook I got this example from suggests adding additional photos of male nurses, even if you have to create them artificially. I agree.
-  This is a data manipulation, and it’s a decision an intelligent systems engineer must often make. We cannot ignore the data we collect, it’s the only real thing we have to base our AI system on. However, the raw data from the real world must often be cleaned and augmented (noise removed, biases mitigated) because our models will be STUCK with what we feed them. 
	- Noise must usually be removed because the AI might wrongly conclude that just the wrong pattern of random noise in a finite dataset is actually meaningful. 
		-  Undesirable correlations (often coming from persistent historical patterns) need to be dealt with for our AI to be accurate, fair, and ethical. Merely repeating the world as it exists is often not enough to satisfy the engineering requirement of fairness for an AI system.

We have to decide what manipulations are allowable and necessary for this domain and this ML algorithm
- **you can't modularize the AI part of your system**, 
	- every bit of it is coupled to every other bit
		- you have to understand how your data manipulations will impact the learning algorithm, because the whole point of these manipulations
			- IS TO INFLUENCE YOUR LEARNING.

Startification is a commonly used dataset manipulation

**Oversample**:
- randomly select, without replacement, samples to duplicate

**Undersample**: 
- randomly select, without replacement, samples to delete.
---
---- Read here -----

•    Let’s talk a little more about manipulating class distributions.

•    We previously discussed how normalization keeps one feature from overly biasing a model. But classes can bias a model too if they are imbalanced.

•    Satellite image of an oil spill: skewed 100,000 : 1 in favor of normal ocean

•    For every pixel showing oil, there are 100,000 showing clean water.

•    If you always guess that your input shows clean water, you will be correct 99.999% of the time. And your AI is useless, because the oil is what you’re trying to find.

•    A well-known problem: the majority-class examples have more opportunities to influence your weight updates. And thus the AI will be biased towards modeling the majority class.

•    Stratification is one major way of fixing this. You can oversample your minority class, duplicating examples in the dataset to give them more influence.

•    You can also thin out the majority class by undersampling it.

•    The effects of under- and over-sampling are unpredictable, although they tend to improve minority-class accuracy at the expense of majority-class accuracy. It’s much the same as you saw with k-NN in the first lab; changing your k changes the predictions in complicated ways.

![[Pasted image 20231010215637.png]]
•    This is a synthetic dataset, generated in Matlab (code on eClass). I just took all of the output data points and threw away the labels (which normally separate this dataset into 2 crescent shapes).

•    2000 datapoints, making a pretty dense population in the classification region.

•    I will undersample this dataset by 90%, to thin it out.

![[Pasted image 20231010215707.png]]•    The same dataset, undersampled. Each datapoint had a 10% chance of staying in the dataset. Not that this does not mean exactly 200 datapoints are left.

•    The shape of the classification region does seem less clear.
![[Pasted image 20231010215732.png]]
•    The undersampled dataset again, but this time with the removed points as magenta dots. Retained points are still blue crosses. We do clearly see to have lost a number of points on the boundary of the dataset, which will change our model.

•    Again, code for doing this in MATLAB is on the eClass page.

•    More advanced stratification routines try to determine which points are “boundary” points, and avoid deleting them.

### SMOTE
Synthetic Minority Oversampling Technique (SMOTE)
- Oversample by creating new data points rather than duplicating existing ones.
	- add a new example to the minority class at a random point on the line between two existing ones.
	![[Pasted image 20231010215856.png]]

•    Now, what about oversampling to increase the number of minority-class samples?

•    Duplicating examples doesn’t work too terribly well. The class regions tend to become very focused around the duplicated points.

•    SMOTE was developed as a response. A very highly regarded technique, with a number of refined variations. Borderline-SMOTE, for instance, oversamples just near the boundary of a class. I’m using that one in some of my current research.

•    Disadvantage: oversampling makes datasets bigger. Undersampling, on the other hand, makes them smaller. For this reason, in the Big Data world, undersampling tends to be favored.

Unknown whether undersampling, oversampling, or both is best for a dataset
- unknown what sampling rates are best for a dataset
	- note that the answers likely change for each ML algorithm

•    So, the implication is that stratification, while it has been demonstrated to often help, adds new dimensions to your hyperparameter grid search. And compute time grows exponentially as a function of dimensions to be searched.

Dataset Engineering is *still a very active area of research*
- means that alot of the questions i posed don't have answers yet.
- your generation of engineers will have to find the answers... or else the AI revolution may just be a passing fad.

•    As things stand right now, I cannot tell you how to build a dataset that reliably leads to an AI satisfying all of its functional and non-functional requirements, being reliable, safe, fair and trustworthy, all at a reasonable price.

•    If we really are going to have AI embedded throughout our lives, then that needs to change.

•    The uptake of new technologies has been studied, and the lesson is, not every promising technology will take root and grow. If too much is promised, and not enough delivered, people (and their money) lose interest.

•    In fact, AI has over-promised and under-delivered before; the AI Winter in the late 1980s lasted more than a decade even for academics, and AI did not come back as a mainstream technology in the public eye until 2015.

•    There’s nothing to stop it from happening again except the talent of AI engineers.