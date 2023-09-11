import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score


df = pd.read_csv('diabetes.csv')
len(df)

from sklearn.model_selection import train_test_split
X = df.drop(['Outcome'], axis=1)
y = df['Outcome']
#function train test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

#scale and transform data
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

#Logistic Regression Function sci kit learn
log_reg = LogisticRegression(random_state = 0)

# fit function
#makes the model considered "trained" or "fitted" to the training data. It means that the model's internal parameters have been adjusted 
# to approximate the underlying patterns in the data, making it capable of making predictions or classifications on new, unseen data.

log_reg.fit(X_train, y_train)


y_pred_test = log_reg.predict(X_test)


#accuracy score
print('Model accuracy score: {0:0.4f}'. format(accuracy_score(y_test, y_pred_test)))


y_pred_train = log_reg.predict(X_train)

#check null accuracy
y_test.value_counts()
null_accuracy = (107/(107+47))

print('Null accuracy score: {0:0.4f}'. format(null_accuracy))

#validation #cross validation
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

#overfitting adn overfitting
print('Training set score: {:.4f}'.format(log_reg.score(X_train, y_train)))
print('Test set score: {:.4f}'.format(log_reg.score(X_test, y_test)))



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



#bias and variance
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


#confusion matrix
from sklearn.metrics import confusion_matrix
import seaborn as sbs
confusion_matrix = confusion_matrix(y_test, y_pred_test)
print(confusion_matrix)
sbs.heatmap(confusion_matrix, annot=True, fmt='d', cmap='YlGnBu')

print('\nTrue Positives(TP) = ', confusion_matrix[0,0])

print('\nTrue Negatives(TN) = ', confusion_matrix[1,1])

print('\nFalse Positives(FP) = ', confusion_matrix[0,1])

print('\nFalse Negatives(FN) = ', confusion_matrix[1,0])

#classification report
from sklearn.metrics import classification_report

print(classification_report(y_test, y_pred_test))

#validation curve
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


#learning curve
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


#Hyperparameter Optimization using GridSearch CV
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

# examine the best model

# best score achieved during the GridSearchCV
print('GridSearch CV best score : {:.4f}\n\n'.format(grid_search.best_score_))

# print parameters that give the best results
print('Parameters that give the best results :','\n\n', (grid_search.best_params_))

# print estimator that was chosen by the GridSearch
print('\n\nEstimator that was chosen by the search :','\n\n', (grid_search.best_estimator_))

print('GridSearch CV score on test set: {0:0.4f}'.format(grid_search.score(X_test, y_test)))
print('GridSearch CV score on train set: {0:0.4f}'.format(grid_search.score(X_train, y_train)))

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