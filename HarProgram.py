import numpy as np
from scipy import stats
import pandas as pd
from sklearn import metrics
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from pylab import rcParams
import seaborn as sns
import tensorflow as tf
import pickle


# If we are using IPython notebook or sypder we can add this line to show all plots explicitly.
# %matplotlib inline

RANDOM_SEED = 42

# sns.set(style='whitegrid', palette='muted', font_scale=1.5)
# rcParams['figure.figsize'] = 14, 8
colnames = ['Users','Activity','Timestamp','x-axis','y-axis','z-axis']
dataset = pd.read_csv('C:/Users/girishp/ML project/data/WISDM_ar_v1.1_raw.txt', names=colnames)
dataset = dataset.dropna()

# print(dataset.head())
# dataset.info()

# Display the data samples with Activity type and by different Users

# dataset['Activity'].value_counts().plot(kind='bar', title='Training samples by Activity type:')
# plt.savefig('Activity.png')
# dataset['Users'].value_counts().plot(kind='bar',title='Training samples by User:')
# plt.savefig('Users.png')
# plt.show()

#  Display the accelerometer data by different activity types:

# def plot_activity(Activity, dataset):
#     data = dataset[dataset['Activity'] == Activity][['x-axis','y-axis','z-axis']][:200]
#     axis = data.plot(subplots=True, figsize=(16, 12), title=Activity)
#
#     for ax in axis:
#         ax.legend(loc='lower left', bbox_to_anchor=(1.0, 0.5))
#
# plot_activity("Sitting", dataset)
# plt.savefig('Sitting.png')
# plot_activity("Standing", dataset)
# plt.savefig('Standing.png')
# plot_activity("Walking", dataset)
# plt.savefig('Walking.png')
# plot_activity("Jogging", dataset)
# plt.savefig('Jogging.png')
# # plt.show()

# By checking the above graph we can assume that the first 200 entries
# in the dataset can be used to distinguish between different activities.
# We can use this to train our model.

# Data preprocessing

N_TIME_STEPS = 200
N_FEATURES = 3
step = 20
segments = []
labels = []
for i in range(0, len(dataset) - N_TIME_STEPS, step):
    xs = dataset['x-axis'].values[i: i + N_TIME_STEPS]
    ys = dataset['y-axis'].values[i: i + N_TIME_STEPS]
    zs = dataset['z-axis'].values[i: i + N_TIME_STEPS]
    label = stats.mode(dataset['Activity'][i: i + N_TIME_STEPS])[0][0]
    segments.append([xs,ys,zs])
    labels.append(label)

print("reduced size of data", np.array(segments).shape)

reshaped_segments = np.asarray(segments,dtype=np.float32).reshape(-1, N_TIME_STEPS, N_FEATURES)
labels = np.asarray(pd.get_dummies(labels),dtype=np.float32)

print("Reshape the segments", np.array(reshaped_segments).shape)

X_train, X_test, y_train, y_test = train_test_split(reshaped_segments, labels, test_size=0.2, random_state=RANDOM_SEED)

# BUILDING THE MODEL

N_CLASSES = 6
N_HIDDEN_UNITS = 64

def create_LSTM_model(inputs):
    W = {
        'hidden': tf.Variable(tf.random_normal([N_FEATURES, N_HIDDEN_UNITS])),
        'output': tf.Variable(tf.random_normal([N_HIDDEN_UNITS, N_CLASSES]))
    }
    biases = {
        'hidden': tf.Variable(tf.random_normal([N_HIDDEN_UNITS], mean=1.0)),
        'output': tf.Variable(tf.random_normal([N_CLASSES]))
    }

    X = tf.transpose(inputs, [1,0,2])
    X = tf.reshape(X, [-1, N_FEATURES])

    hidden =tf.nn.relu(tf.matmul(X, W['hidden']) + biases['hidden'])
    hidden =tf.split(hidden, N_TIME_STEPS, 0)

    #Stack 2 LSTM layers

    lstm_layers = [tf.contrib.rnn.BasicLSTMCell(N_HIDDEN_UNITS, forget_bias=1.0) for _ in range(2)]
    lstm_layers = tf.contrib.rnn.MultiRNNCell(lstm_layers)

    outputs, _ = tf.contrib.rnn.static_rnn(lstm_layers, hidden, dtype=tf.float32)

    lstm_last_output = outputs[-1]
    return tf.matmul(lstm_last_output, W['output']) + biases['output']

tf.reset_default_graph()

X = tf.placeholder(tf.float32, [None, N_TIME_STEPS, N_FEATURES], name="input")
Y = tf.placeholder(tf.float32, [None, N_CLASSES])


pred_Y = create_LSTM_model(X)
pred_softmax = tf.nn.softmax(pred_Y, name="y_")

#using L2 regularization for minimizing the loss

L2_LOSS = 0.0015
L2 = L2_LOSS * \
    sum(tf.nn.l2_loss(tf_var) for tf_var in tf.trainable_variables())

loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=pred_Y, labels= Y)) + L2

#Defining the optimizer for the model

LEARNING_RATE = 0.0025

optimizer = tf.train.AdamOptimizer(learning_rate=LEARNING_RATE).minimize(loss)

correct_pred = tf.equal(tf.argmax(pred_softmax, 1), tf.argmax(Y, 1))

accuracy = tf.reduce_mean(tf.cast(correct_pred, dtype=tf.float32))

#Training the model

N_EPOCHS = 50
BATCH_SIZE = 1024

saver = tf.train.Saver()

history = dict(train_loss = [], train_acc = [], test_loss = [], test_acc = [])

sess = tf.InteractiveSession()
sess.run(tf.global_variables_initializer())
train_count = len(X_train)


for i in range(1, N_EPOCHS + 1):
    for start, end in zip(range(0, train_count, BATCH_SIZE),
                          range(BATCH_SIZE, train_count + 1, BATCH_SIZE)):
        sess.run(optimizer, feed_dict={X:X_train[start:end],
                                       Y:y_train[start:end]})

    _, acc_train, loss_train = sess.run([pred_softmax, accuracy, loss], feed_dict={
        X: X_train, Y:y_train})
    _, acc_test, loss_test = sess.run([pred_softmax, accuracy, loss], feed_dict={
        X: X_test, Y:y_test})

    history['train_loss'].append(loss_train)
    history['train_acc'].append(acc_train)
    history['test_loss'].append(loss_test)
    history['test_acc'].append(acc_test)

    print("test accuracy in history {0:f}".format(acc_test))
    print("test loss in history {0:f}".format(loss_test))

    if i!=1 and i%10!=0:
        continue
    print("Results:")

    print("Epoch: {0}, Test accuracy: {1:f}, Loss: {2:f}".format(i,acc_test,loss_test))

predictions, acc_final, loss_final = sess.run([pred_softmax, accuracy, loss], feed_dict={
   X: X_test, Y:y_test})

print()
print("Final Results: Accuracy: {0:.2f}, Loss: {1:.2f}".format(acc_final,loss_final))

#saving all the predictions and history using the pickle library & create a graph.

pickle.dump(predictions, open("predictions.p", "wb"))
pickle.dump(history, open("history.p", "wb"))
tf.train.write_graph(sess.graph_def, '.', 'har.pbtxt')
saver.save(sess, save_path= "./checkpoint/har.ckpt")
sess.close()

# Loading the files back for evaluating the trained model w.r.t to number of EPOCHS

history = pickle.load(open("history.p", "rb"))
predictions = pickle.load(open("predictions.p", "rb"))

# Evaluations: Plotting the graph

plt.figure(figsize=(12, 8))

plt.plot(np.array(history['train_loss']), "r--", label="Training Loss")
plt.plot(np.array(history['train_acc']), "g--", label="Training Accuracy")

plt.plot(np.array(history['test_loss']), "r-", label="Test Loss")
plt.plot(np.array(history['train_acc']), "g-", label="Test Accuracy")

plt.title("Training session's progress over iterations")
plt.legend(loc='upper right', shadow=True)
plt.ylabel('Training progress(Loss or accuracy)')
plt.xlabel('Training EPOCH')
plt.ylim(0)
plt.savefig('Training iterations.png')
plt.show()

# Building the confusion matrix for display the model predictions vs actual predictions

LABELS = ['DOWNSTAIRS','JOGGING','SITTING','STANDING','UPSTAIRS','WALKING']

max_test = np.argmax(y_test, axis=1)
max_predictions = np.argmax(predictions, axis=1)
confusion_matrix = metrics.confusion_matrix(max_test, max_predictions)

plt.figure(figsize=(16,14))
sns.heatmap(confusion_matrix, xticklabels=LABELS, yticklabels=LABELS, annot=True, fmt="d");
plt.title("CONFUSION MATRIX : ")
plt.ylabel('True Label')
plt.xlabel('Predicted label')
plt.savefig('cmatrix.png')
plt.show();

#freeze the graph to save all the structure, graph and weights into a single protobuf file.

from tensorflow.python.tools import freeze_graph

input_graph_path = '' + 'har' + '.pbtxt'
checkpoint_path = './checkpoint/' + 'har' + '.ckpt'
restore_op_name = "save/Const:0"
output_frozen_graph_name = '' + 'har' + '.pb'

freeze_graph.freeze_graph(input_graph_path, input_saver="", input_binary=False,
                          input_checkpoint=checkpoint_path, output_node_names="y_", restore_op_name="save/restore_all",
                          filename_tensor_name="save/Const:0",
                          output_graph=output_frozen_graph_name, clear_devices=True, initializer_nodes="")