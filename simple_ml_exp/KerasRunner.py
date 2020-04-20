from keras.optimizers import Adam
import os
if os.name == 'nt':
    from keras.models import Sequential
    from keras.layers import Dropout
    from keras.layers import Dense
    from keras.layers import LSTM
elif os.name == 'posix':
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import Dropout
    from tensorflow.keras.layers import Dense
    from tensorflow.keras.layers import LSTM

class KerasRunner(object):
     
    @staticmethod
    def getKerasFCNModel(inputD, outputD, hiddenC = 2, hiddenD = 36, lr = 0.0001): 
        #neurons = [12, 24, 36, 48, 48, 36, 24, 12]
        neurons = [hiddenD] * hiddenC
        model = Sequential()
        model.add(Dense(neurons[0], input_dim=inputD, activation='relu'))
        for n in neurons[1:]:
            #model.add(Dropout(0.2))
            model.add(Dense(n, activation = 'relu'))
        model.add(Dense(units = outputD))
        adam = Adam(lr = lr)
        model.compile(loss='mse', optimizer= 'adam')
        return model
    