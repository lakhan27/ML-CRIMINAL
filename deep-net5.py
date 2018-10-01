import random
import tensorflow as tf 
import numpy as np
import csv

csv.register_dialect('ssv', delimiter=' ', skipinitialspace=True)



training_data = []
with open(r'C:\Users\Rohan Athawade\Desktop\Criminal\criminal_train3.csv', 'r') as f:
    reader = csv.reader(f)
    for row in reader:
        floats = [str(column) for column in row]
        training_data.append(floats)
		
del training_data[0]

for i in range(len(training_data)):
	for j in range(len(training_data[0])):
		training_data[i][j] = float(training_data[i][j])
	
for row in training_data:
	del row[0]
print(len(training_data))
batch_size = 10	
i = 0
final_train = []
while i < len(training_data):
	start = i
	end = i+batch_size 
	final_train1  = []
	
	for j in range(start, end):
		if j!= 599:
			final_train1.append(training_data[j])
		elif j == 599:
			break
	crim = []
	noncrim = []
	
	
	for row in final_train1:
		if row[70] == 1.0:
			crim.append(row)
			
		else:
			noncrim.append(row)
	
	noncrim1 = []
	for k in range(int(len(crim))):
		noncrim1.append(noncrim[k])

	
	final_train2 = [x.pop(0) for x in random.sample([crim]*len(crim) + [noncrim1]*len(noncrim1), len(crim)+len(noncrim1))]
	
	final_train += final_train2
	
	
	#for row in final_train1:
		#print(row[70])
	
	i += batch_size
	


labels1 = []	
for row in final_train:
	temp = [0.0, 0.0]
	if row[70] == 1.0:
		temp[0] = 1.0
	else:
		temp[1] = 1.0
	
	labels1.append(temp)

for row in final_train:
		del row[70]
		


testing_data = []
with open(r'C:\Users\Rohan Athawade\Desktop\Criminal\criminal_test.csv', 'r') as f:
    reader = csv.reader(f)
    for row in reader:
        floatss = [str(column) for column in row]
        testing_data.append(floatss)

testing_data.remove(testing_data[0])

for i in range(len(testing_data)):
	for j in range(len(testing_data[0])):
		testing_data[i][j] = float(testing_data[i][j])

for row in testing_data:
	del row[0]	
	

n_nodes_hl1 = 700
n_nodes_hl2 = 50 	#number of nodes in the hidden layer 
n_nodes_hl3 = 500

n_classes = 2 		#number of outputs or classes 
batch_size = 10	#run input in batches of 100 at a time through the NN




x = tf.placeholder('float', [None, 70])					#input array of none x 70 matrix
y = tf.placeholder('float')					#output 


#defining the neural network model, i.e. all the different things in it 
def neural_network_model(data): 
	hidden_1_layer = {'weights':tf.Variable(tf.random_normal([70, n_nodes_hl1])), 'biases':tf.Variable(tf.random_normal([n_nodes_hl1]))}
	hidden_2_layer = {'weights':tf.Variable(tf.random_normal([n_nodes_hl1, n_nodes_hl2])), 'biases':tf.Variable(tf.random_normal([n_nodes_hl2]))}
	hidden_3_layer = {'weights':tf.Variable(tf.random_normal([n_nodes_hl2, n_nodes_hl3])), 'biases':tf.Variable(tf.random_normal([n_nodes_hl3]))}
	
	output_layer = {'weights':tf.Variable(tf.random_normal([n_nodes_hl3, n_classes])), 'biases':tf.Variable(tf.random_normal([n_classes]))}
	
	#each layer => (input * weight) + biases
	
	l1 = tf.add(tf.matmul(data, hidden_1_layer['weights']), hidden_1_layer['biases'])
	l1 = tf.sigmoid(l1)
	
	l2 = tf.add(tf.matmul(l1, hidden_2_layer['weights']), hidden_2_layer['biases'])
	l2 = tf.sigmoid(l2)
	
	l3 = tf.add(tf.matmul(l2, hidden_3_layer['weights']), hidden_3_layer['biases'])
	l3 = tf.sigmoid(l3)
	
	output = tf.add(tf.matmul(l3, output_layer['weights']), output_layer['biases'])
	
	
	return output
	
lol = []
#defining what the neural network will do, i.e. how it will train itself 
def train_neural_network(x):
	prediction = neural_network_model(x)
	cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits = prediction, labels = labels1))
	optimizer = tf.train.ProximalGradientDescentOptimizer(learning_rate=0.01).minimize(cost)
	
	hm_epochs = 10
	
	#starting the session. Everything after this will happen at run-time. 
	#training
	with tf.Session() as sess:
		sess.run(tf.global_variables_initializer())
		
		for epoch in range(hm_epochs):
			epoch_loss = 0
			for _ in range(len(final_train)):

				_, c = sess.run([optimizer, cost], feed_dict = {x:final_train, y:labels1})
				epoch_loss += c 
					
			
			print("Epoch", epoch, "completed out of ", hm_epochs, " Loss: ", epoch_loss)	
			
		
		lol = prediction.eval(feed_dict={x:testing_data})
		print(lol)
		
		lol = np.array(lol)
		lol2 = []
		for i in range(len(lol)):
			if lol[i][0] > lol[i][1]:
				lol2.append("1")
			else:
				lol2.append("0")
				
		myFile = open(r'C:\Users\Rohan Athawade\Desktop\Criminal\sub5.csv', 'w')
		with myFile:
			writer = csv.writer(myFile)
			writer.writerows(lol2)
		

		#testing	
		correct = tf.equal(tf.argmax(prediction, 1), tf.argmax(y, 1))
		accuracy = tf.reduce_mean(tf.cast(correct, 'float'))
		print('Accuracy', accuracy.eval(feed_dict={x: final_train, y: labels1}))
		


def split(data):
	i = 0
	while i < len(data):
		start = i
		end = i+batch_size 				
		batch_x = np.array(data[start:end])
				
		i += batch_size
		return batch_x
		
train_neural_network(x)	
#print(labels1)


	
		