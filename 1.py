import requests
import math
import numpy as np
import pandas as pd
import pandas_datareader as web
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import LSTM
from tensorflow.keras.layers import Dense
from sklearn.preprocessing import MinMaxScaler
from bs4 import BeautifulSoup
import matplotlib.pyplot as plt
# from datetime import datetime, dat

df=pd.read_csv('./givendata202103231157.txt',usecols=[0],engine='python',skipfooter=20,header=None)
ds=df.values
ds=ds.astype('float128')

# print(ds)

train=ds[0:9960,:]
test=ds[9960:9980,:]

print(len(ds),len(train),len(test))

def create_ds(ds,look_back):
    data_X,data_Y=[],[]
    for i in range(look_back,len(ds)):
        data_X.append(ds[i-look_back:i,0])
        data_Y.append(ds[i,0])
    return np.array(data_X),np.array(data_Y)

look_back=2

train_X,train_Y=create_ds(train,look_back)
test_X,test_Y=create_ds(test,look_back)


#3次元にする
train_X=train_X.reshape(train_X.shape[0],train_X.shape[1],1)
test_X=test_X.reshape(test_X.shape[0],test_X.shape[1],1)

model=Sequential()
model.add(LSTM(128,input_shape=(look_back,1),return_sequences=True))
model.add(LSTM(64))
model.add(Dense(1))

model.compile(loss='mean_squared_error',optimizer='adam')

model.fit(train_X,train_Y,epochs=2,batch_size=1)

train_predict=scaler_train.inverse_transform(train_predict)
train_Y-scaler_train.inverse_transform([train_Y])


train_predict_plot=np.empty_like(ds)
train_predict_plot[:,:]=np.nan
train_predict_plot[look_back:len(train_predict)+look_back,:]=train_predict

test_predict_plot=np.empty_like(ds)

test_predict_plot[len(train_predict)+(look_back*2):len(ds),:]=test_predict

plt.plot(ds,label='ds')
plt.plot(train_predict_plot,label='train_predict')
plt.plot(test_predict_plot,label='test_predict')

plt.legend(loc='lower right')
plt.show()
