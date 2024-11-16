# Importing Libraries
import pandas as pd
import numpy as np
import tensorflow as tf

# Importing Dataset
dataset=pd.read_csv(r'C:\Users\vibho\Documents\Deep Learning\emails.csv')
dataset.head()

# Analysing th dataset
dataset.isna().values
dataset.info()
dataset.isna().sum()
dataset.isna().sum()
x=dataset.iloc[:, 1:-1].values
y=dataset.iloc[:, -1].values
print(y)
print(x)
    
# Splitting the Dataset
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y, test_size=0.25, random_state=0)
print(x_train)
print(x_test)

from sklearn.preprocessing import StandardScaler
sc=StandardScaler()
x_train=sc.fit_transform(x_train)
x_test=sc.transform(x_test)

# Model Creation
ann=tf.keras.models.Sequential()
ann.add(tf.keras.layers.Dense(units=6,activation='relu'))
ann.add(tf.keras.layers.Dense(units=6,activation='relu'))
ann.add(tf.keras.layers.Dense(units=6,activation='relu'))
ann.add(tf.keras.layers.Dense(units=6,activation='relu'))
ann.add(tf.keras.layers.Dense(units=1,activation='sigmoid'))
ann.compile(optimizer='adam',loss='binary_crossentropy',metrics=['accuracy'])

# Training the Model
ann.fit(x_train,y_train,batch_size=32,epochs=100)

# Evaluating the Model
y_pred = ann.predict(x_test)
y_pred = (y_pred > 0.5)
print(np.concatenate((y_pred.reshape(len(y_pred),1), y_test.reshape(len(y_test),1)),1))

# Cofusion Matrix
from sklearn.metrics import confusion_matrix,accuracy_score
cm=confusion_matrix(y_test,y_pred)
print(cm)
accuracy_score(y_test,y_pred)