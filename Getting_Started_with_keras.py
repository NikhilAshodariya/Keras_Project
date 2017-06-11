from keras.models import Sequential
from keras.layers import Dense
import numpy as np
import pandas as pd

# fix random seed for reproducibility
np.random.seed(7)

# load dataset
# dataset = np.loadtxt("../Data/Dataset.csv", delimiter=",")
dataset=pd.read_csv('../Dataset.csv')
# split into input (X) and output (Y) variables

dataset.sample(2)

features=dataset.columns[:-1]
target=dataset.columns[-1]

X=dataset[features]
y=dataset[target]

# Converting from dataframe to array
X=X.as_matrix()
y=y.as_matrix()

# create model
model = Sequential()
# This method creates a sequential neural network

# Dense is used to create a fully connected neural network
model.add(Dense(12, input_dim=8, activation='relu')) 
# This creates a layer with 12 neurons and 8 input neurons with relu as activation function
model.add(Dense(8, activation='relu'))
''' 
    since it is fully connected you do not need to specify the number of input neurons 
    the output of first layer will be fully connected with the input of the second layer
'''
# This creates a layer with 8 neuron in the layer with relu as activation function
model.add(Dense(1, activation='sigmoid'))
# This creates a layer with one neuron in the layer with sigmoid as activation function it act as output layer

# Compile model
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

# loss is the way in which the error is calculate it can be mean_squared_error also instead of binary_crossentropy

# fit the model
model.fit(X, y, epochs=150, batch_size=10,verbose=0)
# verbose=0 is used to avoid print the logs


# evaluate the model
scores = model.evaluate(X, y)
print("\n%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))


# Predicting the output
predictions = model.predict(X)
# round predictions
rounded = [round(x[0]) for x in predictions]
# print(rounded)


np.asarray(rounded)


