# Traffic_Sign_Classification

There are several different types of traffic signs like speed limits, no entry, traffic signals, turn left or right, children crossing, no passing of heavy vehicles, etc. Traffic signs classification is the process of identifying which class a traffic sign belongs to.

In this Python project, I have built a deep neural network model that can classify traffic signs present in the image into different categories. With this model, you can read and understand traffic signs which are a very important task for all autonomous vehicles.

The goals/steps of this project are the following:
* Load the data set
* Explore, summarize and visualize the data set
* Design, train and test a model architecture
* Use the model to make predictions on new images
* Analyze the softmax probabilities of the new images
* Summarize the results with a written report

### Dataset Summary
For this project, I have used the public dataset available at Kaggle:
[Traffic Signs Dataset](https://www.kaggle.com/meowmeowmeowmeowmeow/gtsrb-german-traffic-sign).

The dataset contains more than 50,000 images of different traffic signs. It is further classified into 43 different classes. The dataset is quite varying, some of the classes have many images while some classes have few images. The size of the dataset is around 300 MB. The dataset has a train folder which contains images inside each class and a test folder which we will use for testing our model.

### Dependencies
This project requires:
  * tensorflow==2.0
  * sklearn
  * pandas
  * numpy==1.14.2
  * opencv-contrib-python==3.4.0.12
  * sklearn==0.18.2
  * keras
  * Pillow
### Step 1: Explore the dataset

Our ‘train’ folder contains 43 folders each representing a different class. The range of the folder is from 0 to 42. With the help of the OS module, we iterate over all the classes and append images and their respective labels in the data and labels list. The PIL library is used to open image content into an array. Finally, we have stored all the images and their labels into lists (data and labels).We need to convert the list into numpy arrays for feeding to the model. The shape of data is (39209, 30, 30, 3) which means that there are 39,209 images of size 30×30 pixels and the last 3 means the data contains colored images (RGB value). With the sklearn package, we use the train_test_split() method to split training and testing data. From the keras.utils package, we use to_categorical method to convert the labels present in y_train and t_test into one-hot encoding.

### Step 2: Build a CNN model
To classify the images into their respective categories, we will build a CNN model (Convolutional Neural Network). CNN is best for image classification purposes.

My final model consisted of the following layers:

| Layer         		      |     Description	        					| 
|:---------------------:|:---------------------------------------------:| 
| Input         		      | 32x32x1 grayscale image   							             | 
| Convolution 5x5     	 | 1x1 stride, valid padding, activation: relu, outputs 28x28x32 	 |
| Convolution 5x5     	 | 1x1 stride, valid padding, activation: relu, outputs 24x24x32 	 |
| Max pooling	      	   | 2x2 stride,  outputs 12x12x32 				|
| Convolution 5x5     	 | 1x1 stride, valid padding, activation: relu, outputs 8x8x64 	 |
| Convolution 5x5     	 | 1x1 stride, valid padding, activation: relu, outputs 4x4x64 	 |
| Max pooling	      	   | 2x2 stride,  outputs 2x2x64 				|
| Flatten     	         |	256                          |
| Fully connected		     | 128, activation: relu, keep prob: 0.5        									|
| Fully connected		     | 64, activation: relu, keep prob: 0.75        									|
| Output				            | 43, activation: softmax        									|

We compile the model with Adam optimizer which performs well and loss is “categorical_crossentropy” because we have multiple classes to categorise.

### Steps 3: Train and validate the model

After building the model architecture, we then train the model using model.fit(). I tried with batch size 32 and 64. Our model performed better with 64 batch size. And after 15 epochs the accuracy was stable. Our model got a 95% accuracy on the training dataset. With matplotlib, we plot the graph for accuracy and the loss.

### Step 4: Test our model with test dataset

Our dataset contains a test folder and in a test.csv file, we have the details related to the image path and their respective class labels. We extract the image path and labels using pandas. Then to predict the model, we have to resize our images to 30×30 pixels and make a numpy array containing all image data. From the sklearn.metrics, we imported the accuracy_score and observed how our model predicted the actual labels. We achieved a 95% accuracy in this model. In the end, we are going to save the model that we have trained using the Keras model.save() function 

model.save(‘traffic_classifier.h5’)


### Traffic Signs Classifier GUI

Built a graphical user interface for our traffic signs classifier with Tkinter. Tkinter is a GUI toolkit in the standard python library.

In gui.py file, we have first loaded the trained model ‘traffic_classifier.h5’ using Keras. And then we build the GUI for uploading the image and a button is used to classify which calls the classify() function. The classify() function is converting the image into the dimension of shape (1, 30, 30, 3). This is because to predict the traffic sign we have to provide the same dimension we have used when building the model. Then we predict the class, the model.predict_classes(image) returns us a number between (0-42) which represents the class it belongs to. We use the dictionary to get the information about the class.

### Summary
In this project, I have successfully classified the traffic signs classifier with 95% accuracy and also visualized how accuracy and loss changes with time.
