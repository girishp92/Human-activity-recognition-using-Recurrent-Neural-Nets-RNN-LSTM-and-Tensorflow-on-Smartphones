# Human activity recognition using Recurrent Neural Nets(RNN), LSTM and Tensorflow on Smartphones
This was my Master's project where i was involved using a dataset from Wireless Sensor Data Mining Lab (WISDM) to build a machine learning model for end-to-end systems to predict basic human activities using a smartphone accelerometer, Using Tensorflow API, recurrent neural nets and multiple stacks of Long-short-term memory units(LSTM)  for building a deep network with hidden units.  After the model was trained,  it was saved and exported to an android application and the predictions were made using the model as a proof of concept and the UI interface  to speak out the results using text-to-speech API.

Process:

1. Clean and combine the data
2. Do data pre-processing by taking fixed-length sequences(200) of each sequence activity as training data as per the model requirements for maximizing the efficiency of the model.
3. Split the data into training(80%) and test(20%) sets.
4. Build a deep network by stacking multiple layers of LSTM memory units(This will solve the vanishing gradient problem) with 2 fully connected RNN.
5. Build the whole model using the Tensorflow API and create the placeholders for the model to access at the end-to-end system.
6. Create a loss function for minimizing the loss, we use Least Squared Error(LSE) or L2-norm as it will provide a stable solution with a single solution.
7. Set an optimizer for the model with a single learning rate throughout the training period. Here we have used stochastic gradient descent or AdamOptimizer from the Tensorflow API. 
8. We train the model for 50 Epochs and with the batch size as 1024(32 x 32) i.e 64 hidden units. We got the accuracy above 97% on test sets.
9. Now we save the computation graph with weights, history, predictions and create a checkpoint folder.
10. Next we freeze the Tensorflow graph and save it in a single protobuf file.
11. Export the file to an android application using android studio and build a simple test application for doing predictions on a smartphone using its accelerometer sensor as inputs to the model.
