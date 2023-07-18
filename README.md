# Email-Spam-Detection
This is a Project on Email Spam detection, using deep learning.

The main aim of the project is to tell whether the email is a spam or not. The project used tensorflow, pandas and numpy for importing and dataset and creating the model. The model so formed has a accuracy score of 97.06%.

The dataset csv file is first loaded into the jupyter notebook using pandas library. The dataset is then checked for null values, which this dataset does not possess. the dataset is then divided into x and y dataset With the x dataset containg all the features from first feature to second last, while the y dataset only containing the last Feature.

The dataset is then split into train and test datasets. With the test size of data being 25% and the remaining beign 75%. The data is then scaled using standard scaler. 

The deep learning model used here is made using ANN, with 5 layers. The first 4 layers having 6 units and the last output layer having 1 Unit. The first 4 layers have 'relu' as the activation function and the last layer has 'sigmoid' function. The model is then compiled using adam optimizer to perform stochastic gradient descent, binary_crossentropy for loss function. The model is trained on 100 epochs. 

If the prediction probability value is greater than 0.5 than the mail is a spam mail, otherwise not.
