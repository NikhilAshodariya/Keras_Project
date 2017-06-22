import numpy as np
import pandas as pd
import math
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error


# declaring constants
train_ratio = 0.7
look_back = 2
no_rows = 1
n_epochs = 100
batch_size = 10
n_nodes_hl1 = 500


def make_Cancer_Prediction(arr):
	ans = []
	for index, element in enumerate(arr):
		if len(element) == n_classes:
			ans.append(np.argmax(element))
		else:
			ans.append(-1)

	return ans

def create_dataset(dataset,look_back = 1):
	X, y = [], []
	for i in range(len(dataset)-look_back-1):
		a = dataset[i:(i+look_back),:len(features)]
		X.append(a)
		y.append(dataset[i+look_back][-n_classes:])
	return np.array(X),np.array(y)



dataFrame = pd.read_csv('../../Data/balance.csv')
dataFrame = dataFrame.dropna()
features = dataFrame.columns[:-1]
target = dataFrame.columns[-1]

n_classes = len(set(dataFrame[target])) # for now hardcode it later change it

dataFrame = pd.get_dummies(dataFrame)


dataset = dataFrame.values


# spilt the data into train test 
train_size = int(len(dataset) * train_ratio)
test_size = len(dataset) - train_size
train_data, test_data = dataset[:train_size] , dataset[train_size:]


# reshaping the data so that the data may contain the values
train_X, train_y = create_dataset(train_data,look_back)
test_X, test_y = create_dataset(test_data,look_back)


# reshape input to be [samples, time steps, features]
train_X = np.reshape(train_X, (train_X.shape[0], look_back,train_X.shape[2]))
# train_y = np.reshape(train_y, (train_y.shape[0], 1,train_y.shape[2]))

test_X = np.reshape(test_X, (test_X.shape[0], look_back,test_X.shape[2]))
# test_y = np.reshape(test_y, (test_y.shape[0], 1,test_y.shape[1]))


model = Sequential()
model.add(LSTM(4, input_shape=(look_back,len(features)),activation='relu'))
model.add(Dense(n_nodes_hl1,activation='relu'))
model.add(Dense(n_classes,activation='sigmoid'))
model.compile(loss='categorical_crossentropy', optimizer='adam',metrics=['accuracy'])
model.fit(train_X, train_y, epochs=n_epochs, batch_size=batch_size, verbose=2)


# calculating error in the training the data
scores = model.evaluate(test_X, test_y)

print('')
print("\n%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))

predictions = model.predict(test_X)
print('')
print('--------------------Predictions------------------')
print('predicted value = ',make_Cancer_Prediction(predictions))
print('value = ',make_Cancer_Prediction(test_y))
print('---------------End of Predictions----------------')
