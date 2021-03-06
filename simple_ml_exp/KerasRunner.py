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
from random import Random
    
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
        model.compile(loss='mse', optimizer= 'sgd')
        return model
         
    @staticmethod
    def getKerasLSTMModel(inputD, outputD, hiddenC = 2, hiddenD = 36, lr = 0.0001): 
        model = Sequential()
        model.add(LSTM(units = hiddenD, return_sequences = True, 
                           input_shape = (1, inputD)))    

        model.add(Dense(units = outputD))

        model.compile(optimizer = 'adam', loss = 'mean_squared_error')

        return model
    
    @staticmethod
    def fitLSTM(self, model, xList, yList, batch_size, epochs):
        totalSampleCount = sum([x.shape[0] for x in xList])
        batchPerEpoch = len(xList)
        r = Random()
        for epoch in range(epochs):
            start = time.time()
            print('\nEpoch %d/%d' % (epoch+1, epochs))
            trainingData = list(zip(xList, yList))
            r.shuffle(trainingData)
            i = 0
            for currentBatch, (xx, yy) in enumerate(trainingData):
                self.batch_size = xx.shape[0]
                i += 1
                h = model.fit(xx, yy, batch_size, 1)
                print('\rEpoch %d/%d' % (i, totalSampleCount, \
                    h.history['loss'][-1]), end = '\r')
        return model, h