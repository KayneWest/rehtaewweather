import keras, urllib2 
import numpy as np
from keras.preprocessing import sequence
from keras.optimizers import SGD, RMSprop, Adagrad
from keras.utils import np_utils
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation
from keras.layers.embeddings import Embedding
from keras.layers.recurrent import LSTM, GRU



# download data from http://mesonet.agron.iastate.edu/request/download.phtml?network=IL_ASOS
data  = urllib2.urlopen('http://mesonet.agron.iastate.edu/cgi-bin/request/asos.py?station=MSP&data=tmpf&year1=2011&month1=1&day1=1&year2=2015&month2=6&day2=22&tz=Etc%2FUTC&format=tdf&latlon=no&direct=no').read()

# split data into viable format
data = [x.split('\t') for x in data.split('\n')]

# the third row is the temp, put it into another list
# only ~2 NaNs. too few to really matter with this
# so I just get rid of them
data2=[]
for x in data:
    try:
        data2.append(x[2])
    except:    
        pass

data2 = data2[6:]
# change to float32 in case we have a GPU
data2 = [np.float32(x) for x in data2 if x!='M']


# offset data by one
X = data2[:-1]
y = data2[1:]

# change format for keras
X = np.array([X] * 1).T
y = np.array([y] * 1).T

# train on 40K+ points 
Xtrain = X[:-1000]
ytrain = y[:-1000]

# try to predict final 1000 points
Xtest = X[-1000:]
Ytest = y[-1000:]

# build sequential model 
# Embedding -> LSTM -> Dropout -> LSTM 
# -> Dropout -> Dense -> Softmax
# Cost function = Mean Absolute Error
# Gradient Decent with RMSprop 

model = Sequential()
model.add(Embedding(1000, 200))
model.add(LSTM(200, 200, return_sequences=True)) # try using a GRU instead, for fun
model.add(Activation('tanh'))
model.add(Dropout(0.5))
model.add(LSTM(200, 100))
model.add(Activation('tanh'))
model.add(Dropout(0.5))
model.add(Dense(100, 1))
model.compile(loss='mean_absolute_error', optimizer='rmsprop')
model.fit(Xtrain, ytrain, batch_size=250)
score = model.evaluate(Xtest, Ytest, batch_size=100)
pred = model.predict_proba(Xtest)

if __name__ == '__main__':
    import matplotlib.pyplot as plt
    fig = plt.figure()
    fig.suptitle('RNN for Weather Prediction in Minneapolis', fontsize=20)
    plt.xlabel('Time- - - >', fontsize=18)
    plt.ylabel('Temp', fontsize=16)
    #fig.savefig('test.jpg')
    plt.plot(pred, color="darkred", label = "Model Predicted Temp")#red")
    plt.plot(Ytest, color="steelblue", label = "Actual Temp")
    plt.legend()
    plt.show()

