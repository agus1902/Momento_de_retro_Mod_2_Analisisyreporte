#!/usr/bin/env python
# coding: utf-8

# In[2475]:


import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression


df = pd.read_csv('diabetes.csv')
len(df)


# ## Train and split
# Function from sk learn, test_size means ..., random_state means
# ! [Imagen] (~ / Usuarios / agustintapia / Downloads / 1_train-test-split_0.jpeg)
# 
# 

# In[2476]:


from sklearn.model_selection import train_test_split
X = df.drop(['Outcome'], axis=1)
y = df['Outcome']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)


# In[2477]:


X_train


# In[2478]:


X_test


# In[2479]:


y_train


# In[2480]:


y_test


# # Scale and transform data
# Standard Scaler
# fit_transforms means
# transform means
# 
# 

# In[2481]:


from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)


# ## Logistic Regression Function sci kit learn
# 

# In[2482]:


log_reg = LogisticRegression(random_state = 0)


# # Argument explanation ('solver')
# 'liblinear': This is the default solver for logistic regression in scikit-learn. It is suitable for small to medium-sized datasets and works well for both binary and multi-class classification problems. It uses a coordinate descent algorithm with L1 and L2 regularization.
# 
# 'newton-cg': This solver is appropriate for logistic regression with L2 regularization. It uses the Newton-CG optimization method, which is an iterative method for finding the minimum of a function.
# 
# 'lbfgs': LBFGS (Limited-memory Broyden-Fletcher-Goldfarb-Shanno) is another optimization method used for logistic regression with L2 regularization. It is efficient and suitable for medium to large datasets.
# 
# 'sag': This solver is suitable for large datasets and works well with L2 regularization. SAG stands for Stochastic Average Gradient, and it uses a stochastic gradient descent variant.
# 
# 'saga': SAGA is an improved version of the 'sag' solver, and it supports both L1 and L2 regularization. It is suitable for large datasets and can handle both binary and multi-class classification problems.
# 
# L1 refers to 'Lasso Regularization' which adds the absolute values of the coefficients as a penalty term to the loss function.
# The regularization term added to the loss function is typically represented as λ * ∑|θi|, where θi represents the model's coefficients, and λ (lambda) is the regularization strength, which is a hyperparameter you can tune.
# L1 regularization encourages sparsity in the model because it tends to force some of the coefficients to become exactly zero. This means that L1 regularization can be used for feature selection, as it effectively eliminates less important features from the model.
# L1 regularization is useful when you suspect that only a subset of features is relevant for making predictions.
# 
# L2 refers to 'Ridge Regularization' which adds the squared values of the coefficients as a penalty term to the loss function.
# The regularization term added to the loss function is typically represented as λ * ∑θi^2, where θi represents the model's coefficients, and λ (lambda) is the regularization strength, a hyperparameter.
# L2 regularization penalizes large coefficients and encourages them to be small but does not force them to become exactly zero. It smooths the coefficient values.
# L2 regularization is effective at preventing multicollinearity (correlation between predictor variables) by spreading the impact of correlated features across all of them.
# L2 regularization is generally a good default choice when you want to add regularization to a linear model.

# In[2483]:


log_reg.fit(X_train, y_train)


# In[2484]:


#LogisticRegression(C=1.0, class_weight=None, dual=False, fit_intercept=True,
          #intercept_scaling=1, max_iter=100, multi_class='warn',
          #n_jobs=None, penalty='l2', random_state=0, solver='liblinear',
          #tol=0.0001, verbose=0, warm_start=False)


# # fit function
# makes the model considered "trained" or "fitted" to the training data. It means that the model's internal parameters have been adjusted to approximate the underlying patterns in the data, making it capable of making predictions or classifications on new, unseen data.

# In[2485]:


y_pred_test = log_reg.predict(X_test)

y_pred_test


# # predict function
# A function method to make predictions using a defined user classification model. The predict method takes an input feature matrix and returns the predicted class labels.
# 
# 

# # predicted_proba function
# 
# 

# In[2486]:


log_reg.predict_proba(X_test)[:,0] #probability of getting 0 in the outcome of diabetes


# In[2487]:


log_reg.predict_proba(X_test)[:,1] #probability of getting 1 in the outcome of diabetes


# # Accuracy

# In[2488]:


from sklearn.metrics import accuracy_score

print('Model accuracy score: {0:0.4f}'. format(accuracy_score(y_test, y_pred_test)))


# # Compare the train-set and test-set accuracy

# In[2489]:


y_pred_train = log_reg.predict(X_train)

y_pred_train


# In[2490]:


print('Training-set accuracy score: {0:0.4f}'.format(accuracy_score(y_train, y_pred_train)))


# In[2491]:


log_reg.intercept_


# ## Check null accuracy

# In[2492]:


y_test.value_counts()


# In[2493]:


null_accuracy = (107/(107+47))

print('Null accuracy score: {0:0.4f}'. format(null_accuracy))


# ## Validation (Cross Validation)

# In[2494]:


from sklearn import model_selection
from sklearn.model_selection import cross_val_score

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

log_reg = LogisticRegression(random_state = 0)
log_reg.fit(X_train, y_train)

kfold = model_selection.KFold(n_splits=15)
modelCV = log_reg
scoring = 'accuracy'
results = model_selection.cross_val_score(modelCV, X_train, y_train, cv=kfold, scoring=scoring)
print("15-fold cross validation average accuracy: %.3f" % (results.mean()))


# # Overfitting and underfitting

# In[2495]:


print('Training set score: {:.4f}'.format(log_reg.score(X_train, y_train)))
print('Test set score: {:.4f}'.format(log_reg.score(X_test, y_test)))


# In[2496]:


import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import log_loss


# Create a range of sample sizes for training subsets
sample_sizes = np.arange(10, len(X_train), 10)

# Initialize arrays to store bias and variance values
bias_values = []
variance_values = []
error_values =[]

# Train logistic regression models on different training subset sizes
for sample_size in sample_sizes:
    X_subset = X_train[:sample_size]
    y_subset = y_train[:sample_size]

    model = LogisticRegression()
    model.fit(X_subset, y_subset)

    y_pred = model.predict_proba(X_test)[:, 1]
    
    
    # Calculate log loss (cross-entropy) as a measure of error
    error = log_loss(y_test, y_pred)
    
    # Calculate bias and variance using the squared error decomposition
    #bias_sq = (1 - error) 
    #variance = error

    #bias_values.append(bias_sq)
    #variance_values.append(variance)
    error_values.append(error)

# Plot bias and variance as a function of sample size
plt.figure(figsize=(10, 6))
#plt.plot(sample_sizes, bias_values, label='Bias^2', marker='o')
#plt.plot(sample_sizes, variance_values, label='Variance', marker='o')
#plt.xlabel('Training Sample Size')
#plt.ylabel('Bias^2 and Variance')
#plt.title('Bias-Variance in Logistic Regression')

plt.plot(sample_sizes, error_values, label='Log loss', marker='o')
plt.xlabel('Training Sample Size')
plt.ylabel('Log loss')
plt.title('Log loss in Logistic Regression')


plt.legend()
plt.grid()
plt.show()


# ## bias variance tradeoff

# In[2497]:


import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss

# Define a range of regularization strengths (C values)
C_values = np.logspace(-4, 4, 9)
bias = []
variance = []

# Calculate bias and variance for different C values
for C in C_values:
    # Create a logistic regression model with the current C value
    model = LogisticRegression(C=C, random_state=42)
    
    # Train the model on the training data
    model.fit(X_train, y_train)
    
    # Make predictions on the test data
    y_pred_probs = model.predict_proba(X_test)
    
    # Calculate the log loss (a measure of error) - this is the bias
    logloss = (log_loss(y_test, y_pred_probs))
    var = np.var(y_pred_probs)
    
    # Calculate variance as zero because logistic regression is a linear model
    variance.append(var)
    
    # Bias is log loss
    bias.append(logloss)

# Plot the bias and variance
plt.figure(figsize=(10, 6))
plt.plot(C_values, bias, label='Bias (Log Loss)', marker='o')
plt.plot(C_values, variance, label='Variance', marker='o')
plt.xscale('log')
plt.xlabel('Regularization Strength (C) - Log Scale')
plt.ylabel('Bias (Log Loss) and Variance')
plt.title('Bias (Log Loss) and Variance vs. Regularization Strength in Logistic Regression')
plt.legend()
plt.grid(True)
plt.show()


# # Confussion Matrix

# In[2498]:


from sklearn.metrics import confusion_matrix
import seaborn as sbs
confusion_matrix = confusion_matrix(y_test, y_pred_test)
print(confusion_matrix)
sbs.heatmap(confusion_matrix, annot=True, fmt='d', cmap='YlGnBu')

print('\nTrue Positives(TP) = ', confusion_matrix[0,0])

print('\nTrue Negatives(TN) = ', confusion_matrix[1,1])

print('\nFalse Positives(FP) = ', confusion_matrix[0,1])

print('\nFalse Negatives(FN) = ', confusion_matrix[1,0])


# ## Clasification report

# In[2499]:


from sklearn.metrics import classification_report

print(classification_report(y_test, y_pred_test))


# ## Validation Cruve

# In[2500]:


from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import learning_curve
from sklearn.model_selection import validation_curve
param_range = np.logspace(-3, 3, 7)

# Calculate validation scores for different values of C
train_scores, test_scores = validation_curve(
    LogisticRegression(random_state=0), 
    X_train, y_train, param_name='C', param_range=param_range, cv=5, scoring='accuracy')

# Calculate the mean and standard deviation of validation scores
train_mean = np.mean(train_scores, axis=1)
train_std = np.std(train_scores, axis=1)
test_mean = np.mean(test_scores, axis=1)
test_std = np.std(test_scores, axis=1)

# Plot the validation curve
plt.figure(figsize=(10, 6))
plt.semilogx(param_range, train_mean, label='Training score', color='b', marker='o')
plt.fill_between(param_range, train_mean - train_std, train_mean + train_std, alpha=0.15, color='b')
plt.semilogx(param_range, test_mean, label='Validation score', color='g', marker='o')
plt.fill_between(param_range, test_mean - test_std, test_mean + test_std, alpha=0.15, color='g')
plt.title('Validation Curve for Logistic Regression')
plt.xlabel('C (Regularization Parameter)')
plt.ylabel('Accuracy')
plt.legend(loc='best')
plt.grid(True)
plt.show()

#explanation 


# Un valor alto de C (por ejemplo, C=1.0 o mayor) da como resultado un modelo menos regularizado y, por lo tanto, más complejo.
# Un valor bajo de C (por ejemplo, C=0.01 o menor) da como resultado un modelo más regularizado y, por lo tanto, más simple.
# Un valor de C menor aumentará la regularización, lo que puede dar lugar a coeficientes menores (incluido el término de sesgo)

# ## Learning curve

# In[2501]:


import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_digits
from sklearn.model_selection import learning_curve
from sklearn.linear_model import LogisticRegression

# Load a dataset (example: the digits dataset)

# Create a logistic regression model
model = LogisticRegression( random_state=0)

# Create a function to plot the learning curve
def plot_learning_curve(estimator, title, X, y, ylim=None, cv=None,
                        n_jobs=None, train_sizes=np.linspace(.1, 1.0, 5)):
    plt.figure()
    plt.title(title)
    if ylim is not None:
        plt.ylim(*ylim)
    plt.xlabel("Training examples")
    plt.ylabel("Score")

    train_sizes, train_scores, test_scores = learning_curve(
        estimator, X, y, cv=cv, n_jobs=n_jobs, train_sizes=train_sizes)
    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)

    plt.grid()

    plt.fill_between(train_sizes, train_scores_mean - train_scores_std,
                     train_scores_mean + train_scores_std, alpha=0.1,
                     color="r")
    plt.fill_between(train_sizes, test_scores_mean - test_scores_std,
                     test_scores_mean + test_scores_std, alpha=0.1, color="g")
    plt.plot(train_sizes, train_scores_mean, 'o-', color="r",
             label="Training score")
    plt.plot(train_sizes, test_scores_mean, 'o-', color="g",
             label="Cross-validation score")

    plt.legend(loc="best")
    return plt

# Create the learning curve plot
plot_learning_curve(model, "Learning Curve (Logistic Regression)", X, y, cv=6)
plt.savefig("Learning Curve (Logistic Regression).jpg")

plt.show()


#Learning curves are helpful for understanding whether your model is suffering 
#from overfitting (high variance) or underfitting (high bias).

#The learning curve will help you assess whether your logistic regression model is overfitting or underfitting. 
#If the training and validation scores are both low and close to each other, it may indicate underfitting (high bias). 
#If there's a large gap between the training and validation scores, it may indicate overfitting (high variance).


# # Hyperparameter Optimization using GridSearch CV

# In[2502]:


from sklearn.model_selection import GridSearchCV
df = pd.read_csv('diabetes.csv')
X = df.drop(['Outcome'], axis=1)
y = df['Outcome']


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)
log_reg = LogisticRegression(random_state=0)
log_reg.fit(X_train, y_train)

y_pred_test = log_reg.predict(X_test)

print('Model accuracy score: {0:0.4f}'.format(accuracy_score(y_test, y_pred_test)))
print('Training set score: {:.4f}'.format(log_reg.score(X_train, y_train)))
print('Test set score: {:.4f}'.format(log_reg.score(X_test, y_test)))

parameters = [{'C':[0.001, 0.01, 0.1, 10, 100, 1000]}]



grid_search = GridSearchCV(estimator = log_reg,  
                           param_grid = parameters,
                           #scoring = 'accuracy',
                           cv = 5,)
                           #verbose=0)


grid_search.fit(X_train, y_train)


# In[2503]:


# examine the best model

# best score achieved during the GridSearchCV
print('GridSearch CV best score : {:.4f}\n\n'.format(grid_search.best_score_))

# print parameters that give the best results
print('Parameters that give the best results :','\n\n', (grid_search.best_params_))

# print estimator that was chosen by the GridSearch
print('\n\nEstimator that was chosen by the search :','\n\n', (grid_search.best_estimator_))


# In[2504]:


print('GridSearch CV score on test set: {0:0.4f}'.format(grid_search.score(X_test, y_test)))
print('GridSearch CV score on train set: {0:0.4f}'.format(grid_search.score(X_train, y_train)))


# ## Learning curve

# In[2505]:


import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_digits
from sklearn.model_selection import learning_curve
from sklearn.linear_model import LogisticRegression

# Load a dataset (example: the digits dataset)

# Create a logistic regression model
model = LogisticRegression(C = 0.1, random_state=0, max_iter=100)

# Create a function to plot the learning curve
def plot_learning_curve(estimator, title, X, y, ylim=None, cv=None,
                        n_jobs=None, train_sizes=np.linspace(.1, 1.0, 5)):
    plt.figure()
    plt.title(title)
    if ylim is not None:
        plt.ylim(*ylim)
    plt.xlabel("Training examples")
    plt.ylabel("Score")

    train_sizes, train_scores, test_scores = learning_curve(
        estimator, X, y, cv=cv, n_jobs=n_jobs, train_sizes=train_sizes)
    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)

    plt.grid()

    plt.fill_between(train_sizes, train_scores_mean - train_scores_std,
                     train_scores_mean + train_scores_std, alpha=0.1,
                     color="r")
    plt.fill_between(train_sizes, test_scores_mean - test_scores_std,
                     test_scores_mean + test_scores_std, alpha=0.1, color="g")
    plt.plot(train_sizes, train_scores_mean, 'o-', color="r",
             label="Training score")
    plt.plot(train_sizes, test_scores_mean, 'o-', color="g",
             label="Cross-validation score")

    plt.legend(loc="best")
    return plt

# Create the learning curve plot
plot_learning_curve(model, "Learning Curve (Logistic Regression)", X, y, cv=50)
plt.savefig("Learning Curve (Logistic Regression)gd.jpg")
plt.show()

#Learning curves are helpful for understanding whether your model is suffering 
#from overfitting (high variance) or underfitting (high bias).

#The learning curve will help you assess whether your logistic regression model is overfitting or underfitting. 
#If the training and validation scores are both low and close to each other, it may indicate underfitting (high bias). 
#If there's a large gap between the training and validation scores, it may indicate overfitting (high variance).




# In[2506]:


from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import learning_curve
from sklearn.model_selection import validation_curve
param_range = np.logspace(-3, 3, 7)

# Calculate validation scores for different values of C
train_scores, test_scores = validation_curve(
    LogisticRegression(C = 0.1, random_state=0), 
    X_train, y_train, param_name='C', param_range=param_range, cv=5, scoring='accuracy')

# Calculate the mean and standard deviation of validation scores
train_mean = np.mean(train_scores, axis=1)
train_std = np.std(train_scores, axis=1)
test_mean = np.mean(test_scores, axis=1)
test_std = np.std(test_scores, axis=1)

# Plot the validation curve
plt.figure(figsize=(10, 6))
plt.semilogx(param_range, train_mean, label='Training score', color='b', marker='o')
plt.fill_between(param_range, train_mean - train_std, train_mean + train_std, alpha=0.15, color='b')
plt.semilogx(param_range, test_mean, label='Validation score', color='g', marker='o')
plt.fill_between(param_range, test_mean - test_std, test_mean + test_std, alpha=0.15, color='g')
plt.title('Validation Curve for Logistic Regression')
plt.xlabel('C (Regularization Parameter)')
plt.ylabel('Accuracy')
plt.legend(loc='best')
plt.grid(True)
plt.savefig("Validation Curve (Logistic Regression)2gd.jpg")
plt.show()

#explanation 


# # Decision Tree

# In[2507]:


from sklearn.metrics import accuracy_score
from sklearn.tree import DecisionTreeClassifier


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)
clf_gini = DecisionTreeClassifier(criterion='gini',random_state=0)

clf_gini.fit(X_train, y_train)
y_pred_gini = clf_gini.predict(X_test)


print('Model accuracy score with criterion gini index: {0:0.4f}'. format(accuracy_score(y_test, y_pred_gini)))
print('Training set score: {:.4f}'.format(log_reg.score(X_train, y_train)))
print('Test set score: {:.4f}'.format(log_reg.score(X_test, y_test)))




# ## bias variance tradeoff

# In[2508]:


import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score


# Split the data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Define a range of maximum depths for the decision tree
max_depths = np.arange(1, 21)
bias = []
variance = []

# Calculate bias and variance for different max_depth values
for max_depth in max_depths:
    # Create a decision tree classifier
    model = DecisionTreeClassifier(max_depth=max_depth, random_state=42)
    
    # Train the model on the training data
    model.fit(X_train, y_train)
    
    # Make predictions on the test data
    y_pred = model.predict(X_test)
    
    # Calculate the accuracy (you can replace this with an appropriate metric)
    accuracy = accuracy_score(y_test, y_pred)
    
    # Calculate bias (error) and variance
    bias.append(1 - accuracy)  # Bias is the error
    variance.append(np.var(y_pred) * (1 - np.var(y_pred)))  # Variance

# Plot the bias and variance
plt.figure(figsize=(10, 6))
plt.plot(max_depths, bias, label='Bias (Error)', marker='o')
plt.plot(max_depths, variance, label='Variance', marker='o')
plt.xlabel('Maximum Depth of Decision Tree')
plt.ylabel('Bias and Variance')
plt.title('Bias and Variance vs. Maximum Depth of Decision Tree')
plt.legend()
plt.grid(True)
plt.show()


# In[2509]:


param_range = np.arange(1, 11)

# Calculate validation scores for different values of C
train_scores, test_scores = validation_curve(
    DecisionTreeClassifier(criterion='gini', random_state=0), 
    X_train, y_train, param_name='max_depth', param_range=param_range, cv=5, scoring='accuracy')

# Calculate the mean and standard deviation of validation scores
train_mean = np.mean(train_scores, axis=1)
train_std = np.std(train_scores, axis=1)
test_mean = np.mean(test_scores, axis=1)
test_std = np.std(test_scores, axis=1)

# Plot the validation curve
plt.figure(figsize=(10, 6))
plt.plot(param_range, train_mean, label='Training score', color='b', marker='o')
plt.fill_between(param_range, train_mean - train_std, train_mean + train_std, alpha=0.15, color='b')
plt.plot(param_range, test_mean, label='Validation score', color='g', marker='o')
plt.fill_between(param_range, test_mean - test_std, test_mean + test_std, alpha=0.15, color='g')
plt.title('Validation Curve for Decision Tree Classifier')
plt.xlabel('Max depth')
plt.ylabel('Accuracy')
plt.legend(loc='best')
plt.grid(True)
plt.savefig("Validation Curve Decision Tree Classifier.jpg")
plt.show()


# ## Learning curve

# In[2510]:


import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_digits
from sklearn.model_selection import learning_curve
from sklearn.linear_model import LogisticRegression

# Load a dataset (example: the digits dataset)

# Create a logistic regression model
model = DecisionTreeClassifier(criterion='gini',random_state=0)

# Create a function to plot the learning curve
def plot_learning_curve(estimator, title, X, y, ylim=None, cv=None,
                        n_jobs=None, train_sizes=np.linspace(.1, 1.0, 5)):
    plt.figure()
    plt.title(title)
    if ylim is not None:
        plt.ylim(*ylim)
    plt.xlabel("Training examples")
    plt.ylabel("Score")

    train_sizes, train_scores, test_scores = learning_curve(
        estimator, X, y, cv=cv, n_jobs=n_jobs, train_sizes=train_sizes)
    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)

    plt.grid()

    plt.fill_between(train_sizes, train_scores_mean - train_scores_std,
                     train_scores_mean + train_scores_std, alpha=0.1,
                     color="r")
    plt.fill_between(train_sizes, test_scores_mean - test_scores_std,
                     test_scores_mean + test_scores_std, alpha=0.1, color="g")
    plt.plot(train_sizes, train_scores_mean, 'o-', color="r",
             label="Training score")
    plt.plot(train_sizes, test_scores_mean, 'o-', color="g",
             label="Cross-validation score")

    plt.legend(loc="best")
    return plt

# Create the learning curve plot
plot_learning_curve(model, "Learning Curve (Decision Tree Classifier)", X, y, cv=5)
plt.savefig("Learning Curve Decision Tree Classifier2.jpg")

plt.show()

#Learning curves are helpful for understanding whether your model is suffering 
#from overfitting (high variance) or underfitting (high bias).

#The learning curve will help you assess whether your logistic regression model is overfitting or underfitting. 
#If the training and validation scores are both low and close to each other, it may indicate underfitting (high bias). 
#If there's a large gap between the training and validation scores, it may indicate overfitting (high variance).



# # Using another dataset

# ## Logistic regresion

# In[2511]:


from sklearn.metrics import confusion_matrix
import seaborn as sbs
import pandas as pd
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt

df = pd.read_csv('data1.csv')
df = df.drop(['Unnamed: 32'], axis=1)
df['diagnosis'] = df['diagnosis'].replace({'M': 0, 'B': 1})

X = df.drop(['diagnosis'], axis=1)
y = df['diagnosis']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

log_reg = LogisticRegression()
log_reg.fit(X_train, y_train)
y_pred_test = log_reg.predict(X_test)
print('Model accuracy score: {0:0.4f}'.format(accuracy_score(y_test, y_pred_test)))
print('Training set score: {:.4f}'.format(log_reg.score(X_train, y_train)))
print('Test set score: {:.4f}'.format(log_reg.score(X_test, y_test)))

cm = confusion_matrix(y_test, y_pred_test)
sbs.heatmap(cm, annot=True, fmt="d", cmap="Blues")
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix')
plt.show()


# In[2512]:


from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import learning_curve
from sklearn.model_selection import validation_curve
param_range = np.logspace(-3, 3, 7)

# Calculate validation scores for different values of C
train_scores, test_scores = validation_curve(
    LogisticRegression(), 
    X_train, y_train, param_name='C', param_range=param_range, cv=5, scoring='accuracy')

# Calculate the mean and standard deviation of validation scores
train_mean = np.mean(train_scores, axis=1)
train_std = np.std(train_scores, axis=1)
test_mean = np.mean(test_scores, axis=1)
test_std = np.std(test_scores, axis=1)

# Plot the validation curve
plt.figure(figsize=(10, 6))
plt.semilogx(param_range, train_mean, label='Training score', color='b', marker='o')
plt.fill_between(param_range, train_mean - train_std, train_mean + train_std, alpha=0.15, color='b')
plt.semilogx(param_range, test_mean, label='Validation score', color='g', marker='o')
plt.fill_between(param_range, test_mean - test_std, test_mean + test_std, alpha=0.15, color='g')
plt.title('Validation Curve for Logistic Regression')
plt.xlabel('C (Regularization Parameter)')
plt.ylabel('Accuracy')
plt.legend(loc='best')
plt.grid(True)
plt.savefig("Validation Curve (Logistic Regression)ad.jpg")
plt.show()

#explanation 



# In[2513]:


import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_digits
from sklearn.model_selection import learning_curve
from sklearn.linear_model import LogisticRegression

# Load a dataset (example: the digits dataset)

# Create a logistic regression model
model = LogisticRegression()

# Create a function to plot the learning curve
def plot_learning_curve(estimator, title, X, y, ylim=None, cv=None,
                        n_jobs=None, train_sizes=np.linspace(.1, 1.0, 5)):
    plt.figure()
    plt.title(title)
    if ylim is not None:
        plt.ylim(*ylim)
    plt.xlabel("Training examples")
    plt.ylabel("Score")

    train_sizes, train_scores, test_scores = learning_curve(
        estimator, X, y, cv=cv, n_jobs=n_jobs, train_sizes=train_sizes)
    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)

    plt.grid()

    plt.fill_between(train_sizes, train_scores_mean - train_scores_std,
                     train_scores_mean + train_scores_std, alpha=0.1,
                     color="r")
    plt.fill_between(train_sizes, test_scores_mean - test_scores_std,
                     test_scores_mean + test_scores_std, alpha=0.1, color="g")
    plt.plot(train_sizes, train_scores_mean, 'o-', color="r",
             label="Training score")
    plt.plot(train_sizes, test_scores_mean, 'o-', color="g",
             label="Cross-validation score")

    plt.legend(loc="best")
    return plt

# Create the learning curve plot
plot_learning_curve(model, "Learning Curve (Logistic Regression)", X, y, cv=5)
plt.savefig("Learning Curve (Logistic Regression)ad.jpg")
plt.show()

#Learning curves are helpful for understanding whether your model is suffering 
#from overfitting (high variance) or underfitting (high bias).

#The learning curve will help you assess whether your logistic regression model is overfitting or underfitting. 
#If the training and validation scores are both low and close to each other, it may indicate underfitting (high bias). 
#If there's a large gap between the training and validation scores, it may indicate overfitting (high variance).




# In[ ]:





# ## Decision tree

# In[2514]:


from sklearn.metrics import accuracy_score
from sklearn.tree import DecisionTreeClassifier


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)
clf_gini = DecisionTreeClassifier(criterion='gini',random_state=0)

clf_gini.fit(X_train, y_train)
y_pred_gini = clf_gini.predict(X_test)


print('Model accuracy score with criterion gini index: {0:0.4f}'. format(accuracy_score(y_test, y_pred_gini)))
print('Training set score: {:.4f}'.format(log_reg.score(X_train, y_train)))
print('Test set score: {:.4f}'.format(log_reg.score(X_test, y_test)))

cm = confusion_matrix(y_test, y_pred_gini)
sbs.heatmap(cm, annot=True, fmt="d", cmap="Blues")
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix')
plt.show()


# In[2515]:


y_test.value_counts()


# In[2516]:


y_test.value_counts()

null_accuracy = (108/(108+63))

print('Null accuracy score: {0:0.4f}'. format(null_accuracy))


# In[2517]:


param_range = np.arange(1, 11)

# Calculate validation scores for different values of C
train_scores, test_scores = validation_curve(
    DecisionTreeClassifier(criterion='gini', random_state=0), 
    X_train, y_train, param_name='max_depth', param_range=param_range, cv=5, scoring='accuracy')

# Calculate the mean and standard deviation of validation scores
train_mean = np.mean(train_scores, axis=1)
train_std = np.std(train_scores, axis=1)
test_mean = np.mean(test_scores, axis=1)
test_std = np.std(test_scores, axis=1)

# Plot the validation curve
plt.figure(figsize=(10, 6))
plt.plot(param_range, train_mean, label='Training score', color='b', marker='o')
plt.fill_between(param_range, train_mean - train_std, train_mean + train_std, alpha=0.15, color='b')
plt.plot(param_range, test_mean, label='Validation score', color='g', marker='o')
plt.fill_between(param_range, test_mean - test_std, test_mean + test_std, alpha=0.15, color='g')
plt.title('Validation Curve for Decision Tree')
plt.xlabel('Max depth')
plt.ylabel('Accuracy')
plt.legend(loc='best')
plt.grid(True)
plt.show()


# In[2518]:


import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_digits
from sklearn.model_selection import learning_curve
from sklearn.linear_model import LogisticRegression

# Load a dataset (example: the digits dataset)

# Create a logistic regression model
model = DecisionTreeClassifier(criterion='gini', random_state=0)

# Create a function to plot the learning curve
def plot_learning_curve(estimator, title, X, y, ylim=None, cv=None,
                        n_jobs=None, train_sizes=np.linspace(.1, 1.0, 5)):
    plt.figure()
    plt.title(title)
    if ylim is not None:
        plt.ylim(*ylim)
    plt.xlabel("Training examples")
    plt.ylabel("Score")

    train_sizes, train_scores, test_scores = learning_curve(
        estimator, X, y, cv=cv, n_jobs=n_jobs, train_sizes=train_sizes)
    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)

    plt.grid()

    plt.fill_between(train_sizes, train_scores_mean - train_scores_std,
                     train_scores_mean + train_scores_std, alpha=0.1,
                     color="r")
    plt.fill_between(train_sizes, test_scores_mean - test_scores_std,
                     test_scores_mean + test_scores_std, alpha=0.1, color="g")
    plt.plot(train_sizes, train_scores_mean, 'o-', color="r",
             label="Training score")
    plt.plot(train_sizes, test_scores_mean, 'o-', color="g",
             label="Cross-validation score")

    plt.legend(loc="best")
    return plt

# Create the learning curve plot
plot_learning_curve(model, "Learning Curve (Decision Tree)", X, y, cv=3)

plt.show()

#Learning curves are helpful for understanding whether your model is suffering 
#from overfitting (high variance) or underfitting (high bias).

#The learning curve will help you assess whether your logistic regression model is overfitting or underfitting. 
#If the training and validation scores are both low and close to each other, it may indicate underfitting (high bias). 
#If there's a large gap between the training and validation scores, it may indicate overfitting (high variance).


# # Decision Tree using gridsearch

# In[2519]:


from sklearn.model_selection import GridSearchCV

parameters = [{'max_depth':[1,2,3,4,5,6,7,8,9,10]}]



grid_search = GridSearchCV(estimator = DecisionTreeClassifier(criterion='gini', random_state=0),  
                           param_grid = parameters,
                           #scoring = 'accuracy',
                           cv = 5,)
                           #verbose=0)


grid_search.fit(X_train, y_train)
# examine the best model

# best score achieved during the GridSearchCV
print('GridSearch CV best score : {:.4f}\n\n'.format(grid_search.best_score_))

# print parameters that give the best results
print('Parameters that give the best results :','\n\n', (grid_search.best_params_))

# print estimator that was chosen by the GridSearch
print('\n\nEstimator that was chosen by the search :','\n\n', (grid_search.best_estimator_))

print('GridSearch CV score on train set: {0:0.4f}'.format(grid_search.score(X_train, y_train)))
print('GridSearch CV score on test set: {0:0.4f}'.format(grid_search.score(X_test, y_test)))


# In[2520]:


param_range = np.arange(1, 11)

# Calculate validation scores for different values of C
train_scores, test_scores = validation_curve(
    DecisionTreeClassifier(criterion='gini',max_depth=2, random_state=0), 
    X_train, y_train, param_name='max_depth', param_range=param_range, cv=5, scoring='accuracy')

# Calculate the mean and standard deviation of validation scores
train_mean = np.mean(train_scores, axis=1)
train_std = np.std(train_scores, axis=1)
test_mean = np.mean(test_scores, axis=1)
test_std = np.std(test_scores, axis=1)

# Plot the validation curve
plt.figure(figsize=(10, 6))
plt.plot(param_range, train_mean, label='Training score', color='b', marker='o')
plt.fill_between(param_range, train_mean - train_std, train_mean + train_std, alpha=0.15, color='b')
plt.plot(param_range, test_mean, label='Validation score', color='g', marker='o')
plt.fill_between(param_range, test_mean - test_std, test_mean + test_std, alpha=0.15, color='g')
plt.title('Validation Curve for Logistic Regression')
plt.xlabel('Max depth')
plt.ylabel('Accuracy')
plt.legend(loc='best')
plt.grid(True)
plt.show()


# In[2521]:


import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_digits
from sklearn.model_selection import learning_curve
from sklearn.linear_model import LogisticRegression

# Load a dataset (example: the digits dataset)

# Create a logistic regression model
model = DecisionTreeClassifier(criterion='gini', max_depth=2, random_state=0)

# Create a function to plot the learning curve
def plot_learning_curve(estimator, title, X, y, ylim=None, cv=None,
                        n_jobs=None, train_sizes=np.linspace(.1, 1.0, 5)):
    plt.figure()
    plt.title(title)
    if ylim is not None:
        plt.ylim(*ylim)
    plt.xlabel("Training examples")
    plt.ylabel("Score")

    train_sizes, train_scores, test_scores = learning_curve(
        estimator, X, y, cv=cv, n_jobs=n_jobs, train_sizes=train_sizes)
    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)

    plt.grid()

    plt.fill_between(train_sizes, train_scores_mean - train_scores_std,
                     train_scores_mean + train_scores_std, alpha=0.1,
                     color="r")
    plt.fill_between(train_sizes, test_scores_mean - test_scores_std,
                     test_scores_mean + test_scores_std, alpha=0.1, color="g")
    plt.plot(train_sizes, train_scores_mean, 'o-', color="r",
             label="Training score")
    plt.plot(train_sizes, test_scores_mean, 'o-', color="g",
             label="Cross-validation score")

    plt.legend(loc="best")
    return plt

# Create the learning curve plot
plot_learning_curve(model, "Learning Curve (Logistic Regression)", X, y, cv=50)

plt.show()

#Learning curves are helpful for understanding whether your model is suffering 
#from overfitting (high variance) or underfitting (high bias).

#The learning curve will help you assess whether your logistic regression model is overfitting or underfitting. 
#If the training and validation scores are both low and close to each other, it may indicate underfitting (high bias). 
#If there's a large gap between the training and validation scores, it may indicate overfitting (high variance).

