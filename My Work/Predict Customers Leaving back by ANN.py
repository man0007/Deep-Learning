######################### Part-1 Data Preprocessing ###########################

# Importing the libraries
import pandas as pd

# Importing the dataset
dataset = pd.read_csv("D:\documents\Python\Deep Learning A-Z\Artificial_Neural_Networks\Churn_Modelling.csv")
X = dataset.iloc[:, 3:13].values
y = dataset.iloc[:, 13].values

# Encoding the Independent Variable
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelencoder_X = LabelEncoder()
X[:, 1] = labelencoder_X.fit_transform(X[:, 1]) #country
X[:, 2] = labelencoder_X.fit_transform(X[:, 2]) #Gender

# To avoid this priority (2 is !> 1) in column country we are hot encoding
onehotencoder = OneHotEncoder(categorical_features = [1])
X = onehotencoder.fit_transform(X).toarray()
X = X[:,1:] #removing the dummy variable column in country

# Feature Scaling
from sklearn.preprocessing import MinMaxScaler,StandardScaler
#fs = MinMaxScaler()
fs = StandardScaler()
X = fs.fit_transform(X) 

# Splitting the dataset into the training set and test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

########################## Part-2 Building the ANN ############################

# Importing Packages
from keras.models import Sequential
from keras.layers import Dense

# Initialising the ANN
classifier = Sequential()

# Adding the input layer and the first hidden layer
classifier.add(Dense(units = 6, kernel_initializer='uniform', activation = 'relu',input_dim = 11))

# Adding the second hidden layer
classifier.add(Dense(units = 6, kernel_initializer='uniform', activation = 'relu'))

# Adding the ouput layer
classifier.add(Dense(units = 1, kernel_initializer='uniform', activation = 'sigmoid')) # if the ouput category is more than 2 it will be softmax

# Compiling the ANN (accuracy metric calculates the accuracy of the model based on the training set only)
classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])

# Fitting the ANN to the Training Set (If we use validation_set = example(0.2) then no need to split the data. the model will do it by itself)
classifier.fit(X_train, y_train, batch_size = 10, epochs=100)

######################## part-3 Accuracy Of Test Set ##########################

from sklearn.metrics import accuracy_score

y_pred = classifier.predict(X_test) # gives the probability
y_pred = (y_pred>0.5) # True or False

score = accuracy_score(y_test, y_pred, normalize=True)*100 # True by default. Gives the fraction of correct classifications

print('Accuracy of the Model is:',score)

####################### Part-4 Predicting the Result ##########################

# Single prediction of customer with below features leaves the bank or not
'''
Geography: France
Credit Score: 600
Gender: Male
Age: 40 years old
Tenure: 3 years
Balance: $60000
Number of Products: 2
Does this customer have a credit card ? Yes
Is this customer an Active Member: Yes
Estimated Salary: $50000 
'''

import numpy as np
new_pred = classifier.predict(fs.transform(np.array([[0,0,600,1,40,3,60000,2,1,1,50000]]))) #Matching the dataset and X
new_pred = (new_pred>0.5) # True or False

################### Part-5 Evaluating the ANN by K-folds ######################

from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import cross_val_score
from keras.layers import Dropout
from keras.models import Sequential
from keras.layers import Dense

def build_classifier():
    classifier = Sequential()
    classifier.add(Dense(units = 6, kernel_initializer='uniform', activation = 'relu',input_dim = 11))
    #classifier.add(Dropout(p=0.1)) # Done when overfirtting. First drop 10% if it is again overfitting then try 0.2. Don't go beyond 0.5
    classifier.add(Dense(units = 6, kernel_initializer='uniform', activation = 'relu'))
    #classifier.add(Dropout(p=0.1)) # it si always advisable to apply dropout in all hidden layers.
    classifier.add(Dense(units = 1, kernel_initializer='uniform', activation = 'sigmoid')) # if the ouput category is more than 2 it will be softmax
    classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])
    return classifier

classifier = KerasClassifier(build_fn=build_classifier, batch_size=10, epochs = 100, validation_split =0.1)
accuracies = cross_val_score(estimator = classifier, X = X ,y = y, cv=10,n_jobs = 1) # data by itself will be broken down into train and teset set in CV so no need to use train set seperately

mean = accuracies.mean() # Very high acuuracy = high bias = Overfitting
variance = accuracies.std() # Very high varince = under fitting

###################### Part-6 Tunning ANN Parameters ##########################

from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import GridSearchCV
from keras.models import Sequential
from keras.layers import Dense

def build_classifier(optimizer):
    classifier = Sequential()
    classifier.add(Dense(units = 6, kernel_initializer='uniform', activation = 'relu',input_dim = 11))
    classifier.add(Dense(units = 6, kernel_initializer='uniform', activation = 'relu'))
    classifier.add(Dense(units = 1, kernel_initializer='uniform', activation = 'sigmoid')) # if the ouput category is more than 2 it will be softmax
    classifier.compile(optimizer = optimizer, loss = 'binary_crossentropy', metrics = ['accuracy'])
    return classifier

classifier = KerasClassifier(build_fn = build_classifier)

parameters = {'batch_size':[25,32],
              'epochs':[100,500],
              'optimizer':['adam','rmsprop']
              }

grid_search = GridSearchCV(estimator = classifier,
                           param_grid = parameters,
                           scoring = 'accuracy',
                           cv = 10)

grid_search = grid_search.fit(X, y) # CV by iteself splits Training ans test set, no need to specifically use the training set in here.

best_parameters = grid_search.best_params_ 
best_accuracy = grid_search.best_score_

####################### Part-7 Building the Best Model ########################

def best_classifier(parameters):
    optimizer = parameters[0]
    b_size = parameters[1]
    epochs = parameters[2]
    classifier = Sequential()
    classifier.add(Dense(units = 6, kernel_initializer='uniform', activation = 'relu',input_dim = 11))
    classifier.add(Dense(units = 6, kernel_initializer='uniform', activation = 'relu'))
    classifier.add(Dense(units = 1, kernel_initializer='uniform', activation = 'sigmoid')) # if the ouput category is more than 2 it will be softmax
    classifier.compile(optimizer = optimizer, loss = 'binary_crossentropy', metrics = ['accuracy'])
    classifier.fit(X, y, batch_size = b_size, epochs=n_epochs)
    return classifier

best_fit_model = best_classifier(best_parameters)

###############################################################################