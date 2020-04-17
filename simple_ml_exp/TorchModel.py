import time
import numpy as np
import torch

class TorchModel(torch.nn.Module):
    def __init__(self, inputD, outputD, hiddenC = 3, hiddenD = 36, 
                 batch_size = 30, learning_rate = 0.001):
        super(TorchModel, self).__init__()
        self.batch_size = batch_size
        self.inputD, self.outputD = inputD, outputD
        self.hiddenC, self.hiddenD = hiddenC, hiddenD
        self.linearBegin = torch.nn.Linear(inputD, hiddenD)
        self.linearHiddens = [torch.nn.Linear(hiddenD, hiddenD)]*hiddenC
        self.linearOut = torch.nn.Linear(hiddenD, outputD)
        # The nn package also contains definitions of popular loss functions; 
        # in this case we will use Mean Squared Error (MSE) as our loss function.
        self.loss_fn = torch.nn.MSELoss(reduction='sum')
        self.loss_list = []
        self.learning_rate = learning_rate

    def getFCNModel(self, x, y, batch_size):
        # batch_size is batch size; inputD is input dimension;
        # hiddenD is hidden dimension; outputD is output dimension.
        self.batch_size, self.inputD, self.hiddenD, self.outputD = \
            batch_size, x.shape[-1], 36, y.shape[-1]

        # Use the nn package to define our model as a sequence of layers. nn.Sequential
        # is a Module which contains other Modules, and applies them in sequence to
        # produce its output. Each Linear Module computes output from input using a
        # linear function, and holds internal Tensors for its weight and bias.
        self.model = torch.nn.Sequential(
            torch.nn.Linear(self.inputD, self.hiddenD),
            torch.nn.ReLU(),
            torch.nn.Linear(self.hiddenD, self.outputD),
        )
        return self
        
    def forward(self, x):
        """
        In the forward function we accept a Tensor of input data and we must return
        a Tensor of output data. We can use Modules defined in the constructor as
        well as arbitrary operators on Tensors.
        """
        h_relu = self.linearBegin(x).clamp(min=0)
        for hiddenLayer in self.linearHiddens:
            h_relu = hiddenLayer(h_relu)
        y_pred = self.linearOut(h_relu).clamp(min=0)
        return y_pred


    def fit(self, x, y, epochs):
        print()
        n = x.shape[0]
        bCount = int(n/self.batch_size) 
        x, y = torch.from_numpy(x).float(), torch.from_numpy(y).float()
        self.optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        for t in range(epochs):
            start = time.time()
            print('Epoch %d/%d' % (t, epochs))
            permutation = torch.randperm(x.size()[0])
            le = 40 
            bar = lambda b: '%s>%s' % ('=' * int(le*b/bCount), '.' * int(le*(bCount - b)/bCount)) \
                if b+1<bCount else '='*le
            s = lambda b, t, e, l: '%d/%d [%s] - %ds %dus/sample loss: %.4f' % \
                (b, n, bar(b/self.batch_size), t, e, l)
            for b in range(0, n, self.batch_size):
                indices = permutation[b:b+self.batch_size]
                x_batch, y_batch = x[indices], y[indices]
                # Forward pass: compute predicted y by passing x to the model. Module objects
                # override the __call__ operator so you can call them like functions. When
                # doing so you pass a Tensor of input data to the Module and it produces
                # a Tensor of output data.
                y_pred = self(x_batch)

                # Compute and print loss. We pass Tensors containing the predicted and true
                # values of y, and the loss function returns a Tensor containing the
                # loss.
                loss = self.loss_fn(y_pred, y_batch)
                self.loss_list.append(np.sqrt(loss.item()/self.batch_size))
                # Zero the gradients before running the backward pass.
                self.optimizer.zero_grad()

                loss.backward() 
                self.optimizer.step()
                t = time.time() - start
                e = int((t / (b+1)*self.batch_size)*1000000)
                print('\r%s' % s(b, t, e, loss.item()), end ='\r')
            #if t % 10 == 9:
            #    print('Epoch: %d, RMSE: %.3f' % (t, np.sqrt(self.loss_list[-1])))
            
            print('%s' % s(b, t, e, sum(self.loss_list[-self.batch_size:])/self.batch_size))

        return {'loss': self.loss_list}
    def predict(self, x):
        x = torch.from_numpy(x).float()
        return self(x).detach().numpy()