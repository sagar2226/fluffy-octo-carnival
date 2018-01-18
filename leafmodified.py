import pandas as pd
import os
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder
from keras.utils.np_utils import to_categorical

#importing the data
train_df= pd.read_csv('C:\\Users\\sagar\\Downloads\\LeafClassification\\train\\train.csv')
test_df=pd.read_csv('C:\\Users\\sagar\\Downloads\\LeafClassification\\test\\test.csv')

#checking the rowscount and feature count
print("Number of features in Train : ", train_df.shape)
print("Number of features in Test  : ",test_df.shape)

#checking null values
print(train_df.isnull().any().any())
print(test_df.isnull().any().any())

#dropping the columns and creating a array for target variable in train set
trainLabel = train_df.species.values
train = train_df.drop(['id','species'] , axis=1).values


#dropping the columns and creating a array for target variable in test set
index=test_df.id.values
test = test_df.drop(['id'] , axis=1).values

scaler=StandardScaler()
train_std=scaler.fit_transform(train)
test_std=scaler.fit_transform(test)

## Since the labels are textual, so we encode them categorically
trainLabel = LabelEncoder().fit(trainLabel).transform(trainLabel)
print(trainLabel.shape)

## We will be working with categorical crossentropy function
## It is required to further convert the labels into "one-hot" representation
trainLabel = to_categorical(trainLabel)

#splitting the testdata into validation and test
from sklearn.model_selection import train_test_split
X_train,X_val,Y_train,Y_val = train_test_split(train_std,trainLabel,test_size=0.2,random_state=0)

#Importing the Keras libraries and packages
import keras
from keras.models import Sequential
from keras.layers import Dense

# Initialising the ANN
classifier = Sequential()

# Adding the input layer and the first hidden layer
classifier.add(Dense(output_dim = 792, init = 'uniform', activation = 'relu', input_dim = 192))

# Adding the second hidden layer
#classifier.add(Dense(output_dim = , init = 'uniform', activation = 'relu'))

# Adding the output layer
classifier.add(Dense(output_dim = 99, init = 'uniform', activation = 'sigmoid'))

# Compiling the ANN
classifier.compile(optimizer = 'adam', loss = 'categorical_crossentropy', metrics = ['accuracy'])

# Fitting the ANN to the Training set
history=classifier.fit(X_train, Y_train, batch_size = 10, nb_epoch = 500,validation_data=(X_val, Y_val))

#plot
import matplotlib.pyplot as plt
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()

print('Maximum accuracy', max(history.history['val_acc']))
#val nloss
plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()
print('Minimum loss', min(history.history['val_loss']))

##########
test_std=scaler.fit_transform(test)
yPred = classifier.predict_proba(test_std)
yPred = pd.DataFrame(yPred,index=index,columns=sort(train_df.species.unique()))
fp = open('C:\\Users\\sagar\\Downloads\\LeafClassification\\train\\submission_nn_kernel.csv','w')
fp.write(yPred.to_csv())