Open source machine learning library that supports supervised and unsupervised learning. It also provides various tools for model fitting, data preprocessing, model selection, model evaluation, and many other utilities

main features that scikit-learn provides.
Assumes very basic working knowledge of machine learning practices (model fitting, predicting, cross-validation)

---
### Fitting and predicting: estimator basics
Scikit-learn provides dozens of built-in machine learning algorithms and models, called estimators
each estimator can be fitted to some data using its fit method.

```
from sklearn.ensemble import RandomForestClassifier
clf = RandomForestClassifier(random_state=0)
x = [[1,2,3], # 2 samples, 3 features
	[11,12,13 ]]
y = [0,1] # classes of each sample
clf.fit(x,y)
RandomForestClassifier(random_state=0)
```

The fit method accepts 2 inputs.
	- the samples matrix (or design matrix) `x`. The size of x is typically (n_samples, n_features), 
		- samples are represented as rows and features are represented as columns.
	- target values `y`which are real numbers for regression tasks, or integers for classification (or any other discrete set of values).
		- For unsupervised learning tasks, y does not need to be specified, y is usually a 1d array where `i` in the try corresponds to the target of the ith sample (row) of `x`.

Both x and y are usually expected to be numpy arrays or equivalent array-like data types, though some estimators work with other formats such as sparse matrices.
Once the estimator is fitted, it can be used to predicting target values of new data. you don't need to re-train the estimator:

#### Estimator meaning 
- an object which manages the estimation and decoding of a model. The model is estimated as a deterministic function of:
	- parameters provided in object construction or with set_params:
	- the global numpy.random random state if the estimator's random_state parameter is set to None; and
	- any data or sample properties passed ot hte most recent call to fit, fit_transform or fit_predict, or data similarly passed in a sequence of calls to partial_fit.
The estimated model is stored in public and private attributes on the estimator instance, facilitating decoding through prediction and transformation methods.
Estimators must provide a fit method, and should provide set_params and get_params, although these are usually provided by inheritance from base.BaseEstimator
The core functionality of some estimators may also be available as a function
```
clf.predict(x) # predict classes of the training data
array([0,1])
clf.predict([[4,5,6], [14,15,16]]) # predict classes of new data
array([0,1])
```


### Transformers and pre-processors
Machine learning workflows are often composed of different parts. 
A typical pipeline consists of 
a pre-processing step that transforms or imputes the data 
and 
a final predictor that predicts target values.

In scikit-learn, pre-processors and transformers follow the same API as the estimator objects (they actually all inherit from the same BaseEstimator class). The transformer objects don't have a predict method but rather a transform method that outputs a newly transformed sample matrix x:

```
from sklearn.preprocessing import StandardScaler
x = [[0, 15], 
	[1, -10]]
# scale data according to computed scaling values
StandardScaler().fit(x).transform(x)
```
```
array([[-1.,  1.],
       [ 1., -1.]])
```

Sometimes, you want to apply different transformations to different features: the [ColumnTransformer](https://scikit-learn.org/stable/modules/compose.html#column-transformer) is designed for these use-cases.

StandardScaler()
_class_ sklearn.preprocessing.StandardScaler(_*_, _copy=True_, _with_mean=True_, _with_std=True_)
- standardize features by removing the mean and scaling to unit variance
- The standard score of a sample x is calculate as: 
	$z = \frac{x - u}{s}$
	u - mean of the training samples or zero if with_mean=false,
	s - standard deviation of the training samples or one if with_std=False.

Standardization of a dataset is a common requirement for many machine learning estimators: they might behave badly if the individual features do not more or less look like standard normally distributed data
	standard normally distributed data
		- Gaussian with 0 mean 

### Pipelines: chaining pre-processors and estimators
Pipeline
- Transformers and estimators (predictors) can be combined together into a single unifying object
The pipeline offers the same API as a regular estimator
- It can be fitted and used for prediction with fit and predict.
	- Using a pipeline will also prevent you from data leakage, i.e. disclosing some testing data in your training data.

In this example, we load the iris dataset, split it into train and tests sets, and comput the accuracy score of a pipeline on the test data.
```
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# create a pipeline object
pipe = make_pipeline(
		StandardSCaler(),
		LogisticRegression()
		)

# load the iris dataset and split it into train and test sets
X, y = load_iris(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)

# fit the whole pipeline
pipe.fit(X_train, y_train)

# we can now use it like any other estimator
accuracy_score(pipe.predict(X_test), y_test)
```
```
0.97...
```
### Model Evaluation
Fitting a model to some data does not entail that it will predict well on unseen data. This needs to be directly evaluated. 
The train_test_split helper that splits a dataset into train and test sets, but scikit-learn provides many other tools for model evaluation in particular [cross-validation](https://scikit-learn.org/stable/modules/cross_validation.html#cross-validation).

Example of briefly showing how to perform a 5-fold cross-validation procedure, using the cross_validate helper. 
	Note that it is also possible to manually iterate over the folds, use different data splitting strategies, and use custom scoring functions.

```
from sklearn.datasets import make_regression
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import cross_validate

X, y = make_regression(n_samples=1000, random_state=0)
lr = LinearRegression()

result = cross_validate(lr, X, y)  # defaults to 5-fold CV
result['test_score']  # r_squared score is high because dataset is easy
```
```
array([1., 1., 1., 1., 1.])
```

### Automatic Parameter Searches
All estimators have parameters (often called hyper-parameters in the literature) that can be tuned. The generalization power of an estimator often critically depends on a few parameters.
For example a [`RandomForestRegressor`](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestRegressor.html#sklearn.ensemble.RandomForestRegressor "sklearn.ensemble.RandomForestRegressor") has a `n_estimators` parameter that determines the number of trees in the forest, and a max_depth parameter that determines the maximum depth of each tree. 
Quite often, it is not clear what the exact values of these parameters should be since they depend on the data at hand.

scikit-learn provides tools to automatically find the best parameter combinations (via cross-validation).
In the following example, we randomly search over the parameter space of a random forest with a [`RandomizedSearchCV`](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.RandomizedSearchCV.html#sklearn.model_selection.RandomizedSearchCV "sklearn.model_selection.RandomizedSearchCV") object. When the search is over, the [`RandomizedSearchCV`](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.RandomizedSearchCV.html#sklearn.model_selection.RandomizedSearchCV "sklearn.model_selection.RandomizedSearchCV") behaves as a [`RandomForestRegressor`](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestRegressor.html#sklearn.ensemble.RandomForestRegressor "sklearn.ensemble.RandomForestRegressor") that has been fitted with the best set of parameters.

```
from sklearn.datasets import fetch_california_housing
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import RandomizedSearchCV
from sklearn.model_selection import train_test_split
from scipy.stats import randint

X, y = fetch_california_housing(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)

# define the parameter space that will be searched over
param_distributions = {'n_estimators': randint(1, 5),
						'max_depth': randint(5,10)}

# now create a searchCV object and fit it to the data
search = RandomizedSearchCV(estimator=RandomForestRegressor(random_state=0),
							n_iter=5,
							param_distributions = param_distributions,
							random_state = 0)


search.fit(X_train, y_train)
RandomizedSearchCV(estimator=RandomForestRegressor(random_state=0), n_iter=5,
                   param_distributions={'max_depth': ...,
                                        'n_estimators': ...},
                   random_state=0)
                   
search.best_params_
{'max_depth': 9, 'n_estimators': 4}

# the search object now acts like a normal random forest estimator
# with max_depth=9 and n_estimators=4
search.score(X_test, y_test)
```
```
0.73
```

In practice, you almost always want to search over a pipeline, instead of a single estimator. 
One of the main reasons is that if you apply a pre-processing step to the whole dataset without using a pipeline, and then perform any kind of cross-validation, you would be breaking the fundamental assumption of independence between training and testing data. Indeed, since you pre-processed the data using the whole dataset, some information about the test sets are available to the train sets. This will lead to over-estimating the generalization power of the estimator.
Using a pipeline for cross-validation and searching will largely keep you from this common pitfall.

---
In general a learning problem considers a set of n samples of data and then tries to predict properties of unknown data.
If each sample is more than a single number and, for instance, a multi-dimensional entry (aka multivariate data), it is said to have several attributes or features

Learning problems fall into a few categories:

Supervised learning
- in which the data comes with additional attributes that we want to predict. 
This problem can be either:
	Classification
	- samples belong to two or more classes and we want to learn from already labeled data how to predict the class of unlabeled data.
		- example of a classification problem would be handwritten digit recognition, in which the aim is to assign each input vector to one of a finite number of discrete categories. 
		- Another way to think of classification is as a discrete (as opposed to continuous) form of supervised learning where one has a limited number of categories and for each of the n samples provided, one is to try to label them with the correct category or class.

Unsupervised learning
- in which the training data consists of a set of input vectors x without any corresponding target values. The goal in such problems may be to discover groups of similar examples within the data, where it is called clustering, or to determine the distribution of data within the input space, known as density estimation, or to project the data from a high-dimensional space down to two or three dimension for the purpose of visualization

Training set and testing set
	Machine learning is about learning some properties of a data set and then testing those properties against another data set.
	A common practice in machine learning is to evaluate an algorithm by splitting a data set into two.
	We call one of those sets the training set
		- we learn some properties
	We call the other set the testing set,
		- we test the learned properties.

Some helpful notes could help from the data mining workshop from UAIS [[Data mining Workshop]]