#################### Part-1 identifying frauds using SOM ######################

# Importing the libraries
import numpy as np
import pandas as pd

# Importing the dataframe as df
dataset = pd.read_csv("Credit_Card_Applications.csv")
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, -1].values

# Feature Scaling 
from sklearn.preprocessing import MinMaxScaler
sc = MinMaxScaler(feature_range = (0,1))
X = sc.fit_transform(X)

# Training the SOM
from minisom import MiniSom
som = MiniSom(x = 10, y = 10, input_len = 15, sigma = 1.0, learning_rate = 0.5)
som.random_weights_init(X)
som.train_random(data = X, num_iteration = 1000)

# Visualising the results
from pylab import bone, pcolor, colorbar, plot, show
bone()
pcolor(som.distance_map().T)
colorbar()
markers = ['o','s']
colors = ['r','g']
for i,x in enumerate(X):
    w = som.winner(x)
    plot(w[0] + 0.5,
         w[1] + 0.5,
         markers[y[i]],
         markeredgecolor = colors[y[i]],
         markerfacecolor = 'None',
         markersize = 10,
         markeredgewidth = 2) 
show()    

# Finding the frauds
mappings = som.win_map(X)

frauds = np.concatenate((mappings[(8,1)],mappings[(6,8)]), axis = 0)
frauds = sc.inverse_transform(frauds)

############# Part-2 Finding the probability of fraud usin ANN ################

# Creating matrix of features

customers = dataset.iloc[:, 1:].values # Including all columns except customer id's

# Creating the dependent variable (labels)

is_fraud = np.zeros(len(dataset))
for i in range(len(dataset)):
    if dataset.iloc[i,0] in frauds:
        is_fraud[i] = 1
    
# Feature Scaling
from sklearn.preprocessing import MinMaxScaler,StandardScaler
fs = StandardScaler()
customers = fs.fit_transform(customers) 

# Importing Packages
from keras.models import Sequential
from keras.layers import Dense

# Initialising the ANN
classifier = Sequential()

# Adding the input layer and the first hidden layer
classifier.add(Dense(units = 2, kernel_initializer='uniform', activation = 'relu',input_dim = 15))

# Adding the ouput layer
classifier.add(Dense(units = 1, kernel_initializer='uniform', activation = 'sigmoid')) # if the ouput category is more than 2 it will be softmax

# Compiling the ANN 
classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])

# Fitting the ANN to the Training Set 
classifier.fit(customers, is_fraud, batch_size = 1, epochs=2)

# Predicting the probabilities of frauds
y_pred = classifier.predict(customers) # Probability of each customer for making fraud
y_pred = np.concatenate((dataset.iloc[:, 0:1].values, y_pred), axis = 1)
y_pred = y_pred[y_pred[:,1].argsort()] # sorting based on the increase in probability
















