import tensorflow as tf
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization
from tensorflow.keras.layers import Activation, Dense
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras import optimizers

import pandas as pd
import numpy as np
import pickle
dataset = pd.read_csv('Heart_failure_clinical_records_dataset.csv')
dataset = dataset.drop(dataset[dataset['platelets']>420000].index)
dataset[dataset['ejection_fraction']>=70]
dataset = dataset[dataset['ejection_fraction']<70]
dataset = dataset.drop(dataset[dataset['serum_creatinine']>2.5].index)
dataset = dataset.drop(dataset[dataset['creatinine_phosphokinase']>1500].index)
x = dataset.iloc[:,:12].values
y = dataset.iloc[:,-1].values

from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
x = sc.fit_transform(x)
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.33, random_state =42)
np.random.seed(0)


ann = tf.keras.models.Sequential()
# Adding the input layer and the first hidden layer

ann.add(tf.keras.layers.Dense(units = 7,activation='relu'))

ann.add(tf.keras.layers.Dense(units = 7, activation = 'relu'))
# Adding the third hidden layer

ann.add(tf.keras.layers.Dense(units = 7, activation = 'relu'))
# Adding the fourth hidden layer

ann.add(tf.keras.layers.Dense(units = 7, activation = 'relu'))
# Adding the output layer

ann.add(tf.keras.layers.Dense(units = 1, activation = 'sigmoid'))
# Compiling the ANN

ann.compile(optimizer = 'adam', loss = 'binary_crossentropy' , metrics = ['accuracy'] )
# Training the ANN on the training set

ann.fit(x_train, y_train, batch_size = 32, epochs = 100, validation_data =(x_test,y_test))
pickle.dump(ann, open('ann_model.pkl', 'wb'))



from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer 


def preprocess(input_dict): 
    #ip = np.array(input_dict)
    # ip = input_dict.reshape(-1,1)

    
    enc = OneHotEncoder(handle_unknown='ignore')
    ct = ColumnTransformer([(input_dict,OneHotEncoder(categories='auto'),[1])],remainder='passthrough')
    processed = ct.fit_transform(input_dict)
    return processed

def prediction(int_features):
    sc = StandardScaler()
    x = sc.fit_transform(int_features)
	#model = pickle.load(open('ann_model.pkl','rb'))
    # ip = list(int_features)
    result = ann.predict(x)
    return result