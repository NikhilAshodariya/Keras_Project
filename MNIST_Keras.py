from keras.models import Sequential
from keras.layers import Dense
import numpy as np
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("/tmp/data/", one_hot = True)



def make_mnist_prediction(arr):
	ans = []
	for index, element in enumerate(arr):
		if len(element) == 10:
			ans.append(np.argmax(element))
		else:
			ans.append(-1)

	return ans
	


np.random.seed(7)

train_X, train_y = mnist.train.next_batch(mnist.train.num_examples)
test_X = mnist.test.images
test_y = mnist.test.labels


n_input = 28*28
n_nodes_input_layer = 500
n_nodes_hl1 = 500
n_nodes_hl2 = 500
n_classes = 10
n_epoches = 2
batch_size = 100


model = Sequential()


model.add(Dense( n_nodes_input_layer , input_dim=n_input, activation='relu')) 
model.add(Dense(n_nodes_hl1, activation='relu'))
model.add(Dense(n_nodes_hl2, activation='relu'))

model.add(Dense(n_classes, activation='sigmoid'))
# This creates a layer with one neuron in the layer with sigmoid as activation function it act as output layer


model.compile(loss = 'categorical_crossentropy', optimizer='adam', metrics=['accuracy'])


model.fit(train_X, train_y, epochs=n_epoches, batch_size=batch_size,verbose=0)

# evaluate the model
scores = model.evaluate(test_X, test_y)

print("\n%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))




predictions = model.predict(test_X)
print('')
print('--------------------Predictions------------------')
print('predicted value = ',make_mnist_prediction(predictions))
print('value = ',make_mnist_prediction(test_y))
print('---------------End of Predictions----------------')

