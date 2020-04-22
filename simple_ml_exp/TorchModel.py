import time
import numpy as np
import torch
from random import Random
torch.manual_seed(1)

class TorchFCNModel(torch.nn.Module):
    def __init__(self, inputD, outputD, hiddenC = 2, hiddenD = 36):
        super(TorchFCNModel, self).__init__()
        self.device = torch.device("cuda:0" 
                                   if torch.cuda.is_available() else "cpu")
        self.inputD, self.outputD = inputD, outputD
        self.hiddenC, self.hiddenD = hiddenC, hiddenD
        self.linearBegin = torch.nn.Linear(inputD, hiddenD).to(self.device)
        #self.linearHiddens = [torch.nn.Linear(hiddenD, hiddenD).to(self.device)]*hiddenC
        self.linearHidden1 = torch.nn.Linear(hiddenD, hiddenD).to(self.device)
        self.linearHidden2 = torch.nn.Linear(hiddenD, hiddenD).to(self.device)
        #self.linearHidden3 = torch.nn.Linear(hiddenD, hiddenD).to(self.device)
        self.linearOut = torch.nn.Linear(hiddenD, outputD).to(self.device)

    def forward(self, x):
        h_relu = self.linearBegin(x).clamp(min=0)
        #for h in range(len(self.linearHiddens)):
        #    h_relu = self.linearHiddens[h](h_relu).clamp(min=0)
        h_relu1 = self.linearHidden1(h_relu).clamp(min=0)
        h_relu2 = self.linearHidden2(h_relu1).clamp(min=0)
        #h_relu3 = self.linearHidden3(h_relu2).clamp(min=0)
        y_pred = self.linearOut(h_relu2)#.clamp(min=0)
        return y_pred

class TorchLSTMModel(torch.nn.Module):
    def __init__(self, inputD, outputD, hiddenC = 2, hiddenD = 36):
        super(TorchLSTMModel, self).__init__()
        self.device = torch.device("cuda:0" 
                                   if torch.cuda.is_available() else "cpu")
        self.inputD, self.outputD = inputD, outputD
        self.hiddenC, self.hiddenD = hiddenC, hiddenD
        self.lstm1 = torch.nn.LSTM(inputD, hiddenD).to(self.device)
        self.linearOut = torch.nn.Linear(hiddenD, outputD).to(self.device)
        self.nlayers = 1

    def forward(self, x):
        lstm_out, hidden = self.lstm1(x)
        y_pred = self.linearOut(lstm_out)
        return y_pred

    
    def init_hidden(self, bsz):
        weight = next(self.parameters())
        return (weight.new_zeros(self.nlayers, bsz, self.hiddenD),
                    weight.new_zeros(self.nlayers, bsz, self.hiddenD))


class TorchModel(object):
    FCN = 'TorchFCN'
    def __init__(self, inputD, outputD, hiddenC = 3, hiddenD = 36, 
                 batch_size = 30, lr = 0.001, Model = TorchFCNModel):
        super()
        self.batch_size = batch_size
        self.loss_fn = torch.nn.MSELoss()
        self.loss_list = []
        self.lr = lr
        self.device = torch.device("cuda:0" 
                                   if torch.cuda.is_available() else "cpu")
        print()
        print('Training on', self.device)
        self.model = Model(inputD, outputD, hiddenC, hiddenD)
        self.model.to(self.device)

    def _getProcessBar(self, epoch, batch, batchPerEpoch):
        barLength = 40 
        if batch < batchPerEpoch: 
            progress = '=' * int(barLength * batch / batchPerEpoch)
            rest = '.' * int(barLength * (batchPerEpoch - batch)/batchPerEpoch)
            return '%s>%s' % (progress, rest) 
        else: 
            return '=' * barLength

    def _log(self, epoch, batch, batchPerEpoch, 
             totalSampleCount, computation_time, epoch_loss, samplesSoFar = None):
        draft = '%d/%d [%s] - %ds %dus/sample loss: %.4f'
        epoch, batch = epoch + 1, batch + 1
        if samplesSoFar is None: samplesSoFar = batch * self.batch_size
        bar = self._getProcessBar(epoch, batch, batchPerEpoch)
        computation_rate = int((samplesSoFar / computation_time))
        log = draft % (samplesSoFar, totalSampleCount, bar, 
                       computation_time, computation_rate, epoch_loss)
        print('\r%s' % log, end ='\r')
        return 
            
    def fit(self, x, y, batch_size, epochs):
        self.batch_size = batch_size
        totalSampleCount = x.shape[0]
        batchPerEpoch = int(totalSampleCount / self.batch_size) 
        x, y = torch.from_numpy(x).float(), torch.from_numpy(y).float()
        #self.optimizer = torch.optim.Adam(self.model.parameters(), lr = self.lr)
        self.optimizer = torch.optim.SGD(self.model.parameters(), 
                                         lr = self.lr, momentum = 0.9)
        for epoch in range(epochs):
            start = time.time()
            print('\nEpoch %d/%d' % (epoch+1, epochs))
            permutation = torch.randperm(x.size()[0])
            for batch in range(0, totalSampleCount, self.batch_size):
                indices = permutation[batch:batch+self.batch_size]
                x_batch = x[indices].to(self.device) 
                y_batch = y[indices].to(self.device)
                y_pred = self.model(x_batch)

                loss = self.loss_fn(y_pred, y_batch)
                self.loss_list.append(loss.item())

                self.optimizer.zero_grad()

                loss.backward() 
                self.optimizer.step()

                currentBatch = batch / self.batch_size
                computation_time = time.time() - start
                self._log(epoch, currentBatch, batchPerEpoch, totalSampleCount,
                          computation_time, self.loss_list[-1]) 

            epoch_loss = sum(self.loss_list[-batchPerEpoch:])/batchPerEpoch
            self._log(epoch, currentBatch, batchPerEpoch, 
                      totalSampleCount, computation_time, epoch_loss)
            torch.cuda.empty_cache()
        return {'loss': [sum(self.loss_list[-batchPerEpoch:])/batchPerEpoch]}

    def fitLSTM(self, xList, yList, batch_size, epochs):
        totalSampleCount = sum([x.shape[0] for x in xList])
        batchPerEpoch = len(xList)
        self.optimizer = torch.optim.SGD(self.model.parameters(), 
                                         lr = self.lr, momentum = 0.9)
        r = Random()

        for epoch in range(epochs):
            start = time.time()
            print('\nEpoch %d/%d' % (epoch+1, epochs))
            trainingData = list(zip(xList, yList))
            r.shuffle(trainingData)
            i = 0
            for currentBatch, (xx, yy) in enumerate(trainingData):
                self.batch_size = xx.shape[0]
                i += xx.shape[0]
                xxx, yyy = torch.from_numpy(xx).float(),torch.from_numpy(yy).float()
                self.model.zero_grad()
                
                y_pred = self.model(xxx)

                loss = self.loss_fn(y_pred, yyy)
                self.loss_list.append(loss.item())

                self.optimizer.zero_grad()

                loss.backward() 
                self.optimizer.step()

                #currentBatch = batch / self.batch_size
                computation_time = time.time() - start
                self._log(epoch, currentBatch, batchPerEpoch, totalSampleCount,
                          computation_time, self.loss_list[-1], i) 

            self._log(epoch, currentBatch, batchPerEpoch, 
                      totalSampleCount, computation_time, self.loss_list[-1], i)
            torch.cuda.empty_cache()
        return {'loss': [self.loss_list[-1]]}

    def predict(self, x):
        x = torch.from_numpy(x).float().to(self.device)
        return self.model(x).cpu().detach().numpy()