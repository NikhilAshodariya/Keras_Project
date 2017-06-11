import numpy as np
import pandas as pd
import sys
from keras.layers import Dense
from keras.models import Sequential
from sklearn.cross_validation import train_test_split


def make_Cancer_Prediction(arr):
	ans = []
	for index, element in enumerate(arr):
		if len(element) == n_classes:
			ans.append(np.argmax(element))
		else:
			ans.append(-1)

	return ans
	

def create_Dataset(sample):
	lis = []
	for index, element in enumerate(sample):
		# print('element = ',element)
		if element == 2:
			lis.append([2,0])
		else:
			lis.append([0,4])
	return lis
	pass



dataFrame = pd.read_csv('../../Data/cancerdata.csv')

features = dataFrame.columns[:-1]
target = dataFrame.columns[-1]

train, test = train_test_split(dataFrame)

train_X = train[features]
train_y = train[target]

test_X = test[features]
test_y = test[target]



train_X = train_X.as_matrix()
train_y = create_Dataset(train_y.as_matrix())

test_X = test_X.as_matrix()
test_y = create_Dataset(test_y.as_matrix())


n_input = len(features)
n_nodes_input_layer = 500
n_nodes_hl1 = 500
n_nodes_hl2 = 500
n_classes = len(set(dataFrame[target]))
n_epoches = 100
batch_size = 20


model = Sequential()
model.add(Dense(n_nodes_input_layer, input_dim=n_input, activation='relu'))
model.add(Dense(n_nodes_hl1,activation='relu'))
model.add(Dense(n_nodes_hl2,activation='relu'))
model.add(Dense(n_classes, activation='sigmoid'))

model.compile(loss = 'binary_crossentropy', optimizer='adam', metrics=['accuracy'])
model.fit(train_X, train_y, epochs=n_epoches, batch_size=batch_size,verbose=0)
scores = model.evaluate(test_X, test_y)

print('')
print("\n%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))

predictions = model.predict(test_X)
print('')
print('--------------------Predictions------------------')
print('predicted value = ',make_Cancer_Prediction(predictions))
print('value = ',make_Cancer_Prediction(test_y))
print('---------------End of Predictions----------------')
