Why does SE4AI matter? 
- what's the likely economic impact of AI?
	- Total impact is estimated around $3.5 to $5.8 trillion dollars per year.
- Economics aren't everything, but when you start talking about impacts of trillions per year, then whatever is making that impact is a big deal
	- equally, when trillions of dollars ride on a class of engineered systems, then those systems BETTER WORK!.

From the very popular citation to a 2022 systematic review of SE4AI ACM TOSEM, v. 31 no. 2, art. 37e: 21.
The authors suggest a 3-dimensional classification
1. Scope
	- is AI one algorithm or more in the system? Is it one or more individual components of the overall system, or is it the "point" of of the whole system?
	- Is it an infrastructure or a pipeline rather than an end-user product?
2. Application Domain
	- Lots of AI research is just theory and operationalization
		- dream up a new deep neural network, realize it in PyTorch and try it out on some datasets.
	- SE4AI does not need to focus on the application domain, because engineers BUILDS THINGS to accomplish some tasks.
	- **Whats the domain, and what constraints does it impose on the system?**
		- E.g. self-driving cars are different than chatbots.
3. AI Technology
	- AI is a generic term as "food".
		- There is one "AI"
			- There are thousands of algorithms. 
	- What technology are we dealing with here, what are the characteristics limitations?

Why do we want to use AI?

- If your rationale for using AI is to sound exciting, or it makes the system sound modern and high-tech, you are using it as marketing. 
	- everyone in the tech sector is sprinkling AI fairy dust over all of their products. But that’s not engineering, and it won’t be our focus in this class.
- Are you instead using AI because it’s the best way you can see to accomplish a system goal? 
	- Congratulations, that’s engineering. 
    
- The basic design triad in engineering is the tradeoff between cost, performance, and reliability. **The goal of SE4AI, ultimately, is to achieve high performance and high reliability** – whatever those mean in a specific context – **at an affordable cost.** 
    
- One of the vital questions engineers must answer is, 
	- **how does our proposed solution (AI) compare to other alternatives for accomplishing this use case?** 
		- If you are looking at AI to control (for example) a chemical plant, what is its cost performance and reliability compared to one of the variants of classical optimal control? 
			- In particular, can you analyze stability, controllability, and observability to the same extent in the neurocontroller as for the classical controller?

## Advantages of Artificial Intelligence

- The great advantage of AI comes from the fact that computers are idiots. But really smart when programmed well.
	- The programmer does not have to think of everything. 
		- Considering every possibility as a programmer may be impossible.
- Now imagine the elif ladder you’re going to need for a self-driving car. You have to swiftly recognize many different objects, some moving in different directions at different speeds while others are stationary. You must determine which ones are relevant to your driving task, and how to respond to them. You must then execute the appropriate action in real-time.
	- E.g. DO NOT HIT THE IDIOT CYCLIST WHO JUST RODE ACROSS THE STREET IN FRONT OF YOU. But also, you cannot swerve onto the sidewalk where the nice old lady is taking her dog for a walk, and the road is icy so nailing your brakes just means you’ll lose control.   
		- Make decisions in complex environments
		- perform complex behaviours.
    
- This is what complex means in this context: there are many interacting factors to consider, leading to many possible outcomes of your decision/behavior, and some of those interactions are not favorable to you.

AI in general tries to build programs that can use well-defined approaches to make decisions to execute behaviours that are not wholly predefined


2 Great approaches that AI has studied
1. **The agent might have some stored knowledge about the world around it, and some predefined rules relating objects and actions to each other**.
	- based on these, a logical reasoning process allows the agent to deduce actions to be taken in a novel situation. Expert systems are the classic example, case-based reasoning is another.
		> "Reasoning about the observed world state, based on stored knowledge"

1. **The agent may be provided with stimulus-response pairs representing a desired response to some phenomena.**
	- based on these, the agent inductively learns a functional mapping from inputs to outputs. Neural networks, decision trees, etc.
		- alternatively, if only a reward function is provided, the agent explores its input-action combined space, to inductively determine what actions on what inputs yield the highest reward. This is reinforcement learning
			> "Learning appropriate responses by observation and/or exploration"
			
### State-of-the-art in many tasks
AI algorithms and especially deep learning are the best for many useful tasks
- computer vision includes image classification, object / person recognition, video analysis, and anomaly detection.
- speech recognition enables voices interfaces, such as google home and amazon echo
- Natural Language Processing, (Chat GPT)

### AI can find unexpected solutions
- In 2016 AlphaGo used AI to beat Lee Sedol. This was the moment that deep learning and the "new AI" first burst into the mainstream news.
	- AlphaGo's 19th move was unusual, prompting one the expert commentators to say he hadn't seen a human take that approach before
One of the key use cases for AI is to uncover associations by mining every large datasets associations that humans can't find because we can't wade through that much data.
- AI is used to discover new solutions in many areas

Rational Drug Discovery predicts the 3d Shape of new proteins, based on how many other molecules fold up.

# Challenges of AI Systems
### AI Bias
Refers to unjust or unacceptable outcomes due to how the AI was trained

2019
- a hospital used AI to predict which patients would benefit from more care. One of the inputs was people's prior health care spending, which in the US is heavily dependent on income.
	- poorer people spend less on health care because it comes out of their own pockets 
	- black people are disproportionately poor, and thus there's an implicit correlation - that the AI found - between race and prior health spending.
		- but it's not helpful for predicting who would benefit from more care

2015
- Google pictures was returning disproportionately few pictures of women CEOs, and google's ad network was serving high-paying job adverts disproportionately to men. 
	- Ad networks are dependent on click-through rate predictors - an AI predicts how likely a specific user is to click a specific ad on a specific website at a specific time. 
		- Before the rise of GPT, these were the largest AI systems on the planet, trained on trillions of user-ad interactions. They were biased against showing those ads to women.

2023
- Det Police used facial recognition to **falsely** arrest a black woman for "carjacking".
	- The woman was 8 months pregnant.
	- Facial recognition is known to have a much higher error rate for minorities compared to whites,
		- largely because the databases they are trained on are predominantly of white people's faces.

### Brittleness
Change the input to an AI just a little, and it can break
- Changes may be too small for humans to see
- can cause unexpected, maybe dangerous behaviour

Possibly accidental, but possibly not
- Disruption in normal processing (glitches) also cause unpredictable behaviours.

AI working furiously hard to find anything that statistically associates with the target class. 
	- we built algorithms that squeeze pictures for every drop of information, by doing nonlinear transforms on groups of pixels
		- the final prediction model, while seeming to work great, commonly has almost nothing to do with how a human would actually recognize an image.
	- By changing the right pixels the wrong way, you can bias those nonlinear transforms to give you something wildly different.
		- even if the changes made are so slight that a human literally can't see them
			- these altered images are called adversarial examples

It can also happen than an image that is just slightly outside what the Neural network is trained on gets a wildly different predicted label, just accidentally.

![[Pasted image 20231010142226.png]]
glasses with an adversarial patch added. 
This patch was successful in causing trained facial recognition systems to confidently identify this person as some other random person from the training dataset.
	- Thus, they dodged the recognizer.

### Changing the environments and user behaviour
Microsoft Tay chatbot
- designed to learn from its interactions on the internet.
	- The result:
		- Twitter users bombarded it with enough profanity and hate speech that the bot learned this was acceptable and started mimicking it. 
			- Tay was shut down less than 24 hours after going live, and was never re-released.

Soccer match with "ball tracking" using neural network.
- The ai spotted a bald referee and decided that his head was the ball, all game long. Home audience was not entertained.

Zillow offers was a unit of Zillow, an online real-estate firm. The unit specialized in flipping houses using Zillow's own funds.
- Ai was used to predict future house prices. 
- However,
	- covid knocked the real estate market around bad, first depressing it then turbocharging it.
		- This udermined the accuracy of the AI's predictions, and ultimately the profit and loss for the whole unit

Cruise self-driving vehicles all came to a stop in San Francisco, causing a major traffic jam.
- music festival that day caused a big spike in wireless traffic - interrupting connectivity from central stations to the vehicles.
	- Presumably "stop and put on hazard lights" is a designated safe state that the cars went into.
		- Change in the environment that the system could not adopt to 

## AI systems Life-Cycle
![[Pasted image 20231010161328.png]]
AI requirements
- fleshes out the use cases you've decided need AI

Data Collection
- these are the inputs your AI will learn from. 
- Dataset Engineering is now a vital topic we 're going to talk about in this module

Data Cleaning
- cleaning up the mess that is data in the real world. No nice clean distributions, "rare" events happening more than they should, noise everywhere, data might be missing or corrupted in transmission

Data Labeling
- attaching the correct output to each input. 
- Often extremely difficult and expensive; guestimates won't do, because your learning algorithm presumes it is given the THE TRUTH

Feature Engineering
- Feature selection/reduction, or derived features.
	- This is a core step in improving the training time and performance of the AI.

Model Training
- you feed data to the AI algorithm. This step is the actual machine learning; 
	- everything up to now was getting data ready to be fed in, and everything after is about assessing and utilizing the AI to fulfill its use cases.

Model Evaluation
- is the AI accurate?
- unbiased?
- behaves correctly in new situations?

Model Deployment
- putting the AI to work. Actually use the AI outputs to do some useful work.

Model Monitoring
- you have to constantly keep your eye on the ai, and make sure it continues to behave correctly. (until you push out the next version.)

### Pipeline
![[Pasted image 20231010162022.png]]
The steps that are inside the red line is part of the AI Pipeline. This is everything from ingestion of data right through the monitoring of a deployed model.
Specifically the AI pipeline consists of:
1. Data Cleaning
2. Data Labelling
3. Feature Engineering
4. Model Training
5. Model Evaluation
6. Model Deployment
7. Model Monitoring

This automation lets us train up a new version of the AI on fresh incoming data, even while the last deployed version is still running and processing that same data.
	- when a new version starts outperforming the older one (usually because it is a better model for the newest data), you would cut over and deploy the new one while pulling the old one offline.

- Dozens of new models might be generated each day. This is a way to deal with changes in the environment and user behaviours.
	- remember, everything revolves around the AI model. 
	- The whole pipeline exists to support it, and deploy new versions when the old one starts to lose relevance.

- to understand the pipeline, we first have to understand the AI; what do we need to feed it, and what will come out the other end?

### Feature Vector
A dictionary is a data object consisting of name-value pairs.
- a real-world object is represented in a record by certain observations about that object ("attributes"). 

Feature
- Each kind of observation is a "feature" rather than an attribute
- a specific object will have specific values for each feature; these are the feature values for the object
- A feature has a legal set of values.
	- this is usually called a dimension in machine learning

Dimension "name" is all possible names (these are a special kind of string)
Dimension "age" all possible ages (positive integers, less than 200)
- We usually order the features representing objects in a fixed way, so we can treat them as vectors.


### Feature Space
Extending the idea of feature dimensions to feature space;
- virtual space where each axis is one feature dimension, each at right angles to every other one. (vector space from linear algebra)
	- one very powerful tool this gives us is looking at how feature vectors (objects) are distributed in the feature space.
		- Are there natural groups they fall into? 
		- Do these groups mean something? 
		- Is there any overlap. or are they cleanly  separated?

![[Pasted image 20231010163535.png]]
•    Here, car engines are represented as their displacement, and fuel efficiency. There’s a third variable, the number of gears in the transmission. That’s represented by the color and symbol used to plot each feature vector. Normally in machine learning, this color/shape for a point would be used to represent a different label (i.e. the output variable)

•    Notice that each object is a feature vector, which also means it is one point in the feature space, and it will be assigned one label value (just a scalar).

### Classification
Idea of classification in feature space. 
The core idea is that feature vectors close to each other should share the same label, while points far away are likely to have different labels.
![[Pasted image 20231010163808.png]]
**The K-NN Classifier**
1. memorize all the training data points you're given
2. take the k points that are closest to a new test data point it has not seen, and predict that the test point has the most common label amongst those k neighbours.
	This example: for a 1-NN classifier, you would predict "blue"
				For a 3-NN classifier, you would predict "red"
	K is a hyper-parameter. It's one of the values that has to be provided to a learning algorithm as an input before learning even starts
		- So what is the right value of k? 
			- we don't know.

#### We don't know???
- There is no scientific theory of learning or pattern recognition for machines, because we don't have one for humans either.
- At the practical engineering level: that means we cannot tell in advance how well any algorithm, with a chosen set of hyper-parameters, will perform on a dataset; what the prediction accuracy will be on the training set, and especially how that will change on the test data points. 
- All we can do is try it out and see what happens. (Parameter Exploration)

### Parameter Exploration
![[Pasted image 20231010164411.png]]
The basic approach for machine learning is:
- take the data you have
- split it into a training set and separate test set

This is called out-of-sample testing in stats, and used as an unbiased estimate of how well the model generalizes to new data.

**Hyper-parameter exploration**
You will train the model many times, changing its hyper-parameters to explore how the different values affect the final accuracy of the predictions. 

For K-nn classifier
Would normally expect the accuracy to increase along with k, until it hits a plateau, and then starts to drop.
- what happens is that you have initially  have too few neighbours to make an accurate prediction. Then as the number of neighbours rises, you'll have increasingly accurate classification,
	- until you reach the natural limit of the classifier on this dataset
- As you keep increasing the number of neighbours, you're bringing in more and more distant examples, which are probably less reliable guides to the labeling of this test point. 
	- so you would see the accuracy start to drop off again.
	- overfitting

### Machine Learning: [[What Training Requires]]
Summary:
**Supervised Learning**:
- explicitly map inputs to outputs
	- requires feature vectors and labels for every object

**Reinforcement Learning**:
- optimize a reward function 
	- requires state vectors and a reward function for every possible action-state pair.

**Unsupervised Learning**:
- clustering objects into groups
	- requires only feature vectors

[[What Training Requires#But also]]

### [[Machine Learning Pipelines]]

### Software Engineering for AI: Software engineering Body of Knowledge
![[Pasted image 20231010211903.png]]
The body of knowledge is a codification, maintained by the IEEE, of what a fully trained and licensed software engineer should know.

•        [[elicitation]], analysis, specification and validation of requirements.

•        Software architecture, software components and modules. User interface design. Design paradigms: procedure, data and object oriented design, others.

•        Conversion of a design to working code. Languages, toolchains, programing techniques.

•        Testing approaches and plans. Coverage criteria. Test harnesses, scripts. Unit, integration, system and regression testing. This is much more complicated because software is digital, and therefore can jump between wildly different states instantaneously, where physical systems cannot.

•        Maintenance is also complicated, as maintainers often do not have a full understanding of the conceptual models underlying a software system (they’re too complex). Maintenance (to fix bugs, adapt the system to a new environment, or add new functions) thus slowly breaks down the conceptual integrity of the program. In fact, some writers consider this breakdown to be the equivalent of aging and wear-out for software.

•        Common to SW and highly complex physical systems. Ultimately about controlling changes to reqs or design, so that the final delivered product is still what was desired. Change control, change impact analysis, configuration audit (to PROVE what was delivered matches updated specs).

•        Software contract formation, project planning & execution. Typical of all engineering management.

•        Software life cycles, product and process measurement. Process improvement, process maturity growth.

•        Models used in SE: information, human behaviour, structural modeling. Syntax and semantics of models. Model analysis. Methods including formal and agile approaches to software development.

•        SQ measurement and analysis. Quality engineering. V&V, reviews, audits. Quality improvement processes.

•        Professionalism, group dynamics and psychology, communication skills. Always necessary, but more vital in software: the great majority of software failures seem to come down to inadequate communication between possibly thousands of developers working on a large system.

•        Basic economics and bookkeeping. Project cost estimation and tracking. Investment portfolio management (multiple projects competing for resources), risk management.

•        Computer science, much as physics is the foundation of mechanical or electrical engineering

•        Discrete mathematics, automata theory. Completely different from all other engineering disciplines which rely on continuous mathematics (calculus, differential equations).

•        Common across engineering disicplines. Experimental methods, statistical analysis, engineering design and problem solving. Standards and root-cause analysis.

**So how do these change in SE4AI?** 
	No textbooks that have this material yet

- Partial specifications, 
	- e.g. self-driving cars must recognize and yield to pedestrians – but what, EXACTLY are the rules for recognizing a pedestrian in a digital image? 
		- *Often AI is chosen for a use case because exact specification of the behaviour and/or environment is not feasible.* 
			- How do you validate a system with partial specifications? 
				- There is also much more emphasis on non-functional requirements (bias, trustworthiness, robustness, etc.)

- Most AI design focuses on improving one specific quality attribute (e.g. transparency or fairness). 
	- Literature on design patterns is still in an early stage, few patterns have been proposed and evaluated.
	- There is no design pattern in Software Engineering 4 AI.

- Some guidelines for specific design challenges exist, but there is very little literature on engineering AI to conform precisely to stated requirements.

- *One of the most heavily studied areas*. Most focus is on designing and implementing test cases for AI systems, measuring test case coverage, and testing for specific quality attributes.

- *Maintenance has barely been studied at all*. 
	- It is known that trained AIs are black boxes, and highly brittle; 
		- changing them carries significant risks for major failure as everything is entangled with everything else.

- Only a single known paper on configuration management, a case study at Amazon.

- Software has always been interdisciplinary in that is touches many application domains, but the software construct itself was not. 
	- *No longer true*, Software Engineers must work with data scientists / AI experts to create the AI construct. Significantly complicates management.

- ML process research has largely focused on the “pipeline” architecture pattern we are discussing in this course. Frameworks and toolchains are emerging that support this pattern.

- The main thrust of the research in this area is proving safety of AI systems in safety-critical applications; 
	- much less modeling of information and user behaviour. Cobots, however, are an exception, but that is really a different field.

- Most studies in this area focus on quality attributes such as safety, robustness, explainability, etc. Definition, verification, assurance, frameworks for creating systems with these quality attributes. 
	- There’s not much apparent about usability of AI systems, which is a major concern in SE.

- One paper only, about producing fact sheets about the AI.

- No papers at all for (12)-(15). It seems obvious, however, that the mathematics and computing foundations areas would have to additionally include optimization, linear algebra, and integer programming.
	- specifically no papers for:
		- software engineering economics
		- computing foundations
		- mathematical foundations 
		- engineering foundations

### SE4AI: Requirements
![[Pasted image 20231010212028.png]]
In every engineering discipline, deciding on requirements is a joint effort between engineers and clients (whoever that is)
- For software, we distinguish [[elicitation]] and definition of requirements from their specification
	1. we engage in knowledge discovery, to understand the application domain AND the client's needs
		- e.g. students from the aeronautical engineering program can be qualified to design and implement fly-by-wire systems. There’s quite a bit of effort spent to teach the software engineers about the application domain, so that their design decisions are sensible.
			- the software engineers has to know the domain that the aeronautical engineers are talking about to implement properly their desired application, 
				- but the aeronautical engineers also need to educate them selves of the software engineering domain to know the limitations of software engineering. Such as bugs and deadlines, and version control.
	2. These learnings are turned into detailed functional and non-functional requirements, which is the reference against which the software is designed.

Software is directly created. AI is not; we must therefore also specify how to train an AI in such a way that it meets the specifications laid out in the SRS. 
- These are vital decisions that the client should be educated on and participate in.

### Use Case Model
![[Pasted image 20231010212844.png]]
•    Use case models are a standard way of documenting requirements in software engineering. 4 main elements are actors, use cases, system boundary, and relationships.

•    Actors consume the services provided by software (or some other system). Could be users, could also be other computer systems. In either case, they are who uses the software.
	- Actors are the waiters, clients, cashiers, chefs.

•    Use cases are system functions the produce observable results for the user. They must always be related (directly or indirectly) to an actor outside of the system.

•    The system boundary represents what is “inside” the software, and divides it from “outside.”

•    Relations mostly define how actors will interact with use cases. The “extend” relation indicate a specialization (so ordering wine is a special case of ordering food, different enough that it needs separate tracking).

- **In SE4AI, you would start with a use case diagram for the system, and then determine which use cases you will build an AI to accomplish.**

### Machine Learning Canvas
![[Pasted image 20231010214447.png]]
The ML Canvas is a template for converting use cases into the requirements for an AI system, one at a time. 
- Start with one use case in the middle, and analyze the value being delivered. ("Value Proposition")
	- not a standard yet.

On the left.
- How do we turn the use case into an AI model?
1st. What AI task will the use case be translated into, and how would this actually by employed for the user's needs?
- what are the inputs and the ouputs?
- when will we make predictions, and what are the time constraints and compute needs?
	- how do we validate the model against the user's needs?

On the right
- how do we select and train the AI model?
- What raw data sources are available to us, and how do we want to collect a training set from them?
	- how do we then turn raw data into features (including data cleaning steps required). 
	- what models do we choose for training, and how do we update them? How long does training take, and what are the compute needs?

At the bottom
- how do we monitor the AI system? 
	- What metrics do we collect to determine impact on users and return on investment? 
	- How do these relate when we need to update the model?

### [[Dataset Engineering]]