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
        self.linearBegin = torch.nn.Linear(inputD, hiddenD).cuda()
        self.linearHiddens = [torch.nn.Linear(hiddenD, hiddenD).cuda()]*hiddenC
        self.linearOut = torch.nn.Linear(hiddenD, outputD).cuda()
        # The nn package also contains definitions of popular loss functions; 
        # in this case we will use Mean Squared Error (MSE) as our loss function.reduction='sum'
        self.loss_fn = torch.nn.MSELoss()
        self.loss_list = []
        self.learning_rate = learning_rate
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        print()
        print('Training on', self.device)
        self.to(self.device)


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
        self.cuda()
        return self
        
    def forward(self, x):
        """
        In the forward function we accept a Tensor of input data and we must return
        a Tensor of output data. We can use Modules defined in the constructor as
        well as arbitrary operators on Tensors.
        """
        self.to(self.device)
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
        #x, y = x.to(self.device), y.to(self.device)
        self.optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)

        le = 40 
        bar = lambda b: '%s>%s' % ('=' * int(le*b/bCount), '.' * int(le*(bCount - b)/bCount)) \
            if b+1<bCount else '='*le
        s = lambda b, t, e, lo: '%d/%d [%s] - %ds %dus/sample loss: %.4f' % \
            (b, n, bar(b/self.batch_size), t, e, lo)
        for t in range(epochs):
            start = time.time()
            print('Epoch %d/%d' % (t, epochs))
            permutation = torch.randperm(x.size()[0])
            for b in range(0, n, self.batch_size):
                indices = permutation[b:b+self.batch_size]
                x_batch, y_batch = x[indices].to(self.device), y[indices].to(self.device)
                # Forward pass: compute predicted y by passing x to the model. Module objects
                # override the __call__ operator so you can call them like functions. When
                # doing so you pass a Tensor of input data to the Module and it produces
                # a Tensor of output data.
                y_pred = self(x_batch)

                # Compute and print loss. We pass Tensors containing the predicted and true
                # values of y, and the loss function returns a Tensor containing the
                # loss.
                loss = self.loss_fn(y_pred, y_batch)
                self.loss_list.append(loss.item())
                # Zero the gradients before running the backward pass.
                self.optimizer.zero_grad()

                loss.backward() 
                self.optimizer.step()
                t = time.time() - start
                e = int((t / (b+1)*self.batch_size)*1000000)
                print('\r%s' % s(b, t, e, self.loss_list[-1]), end ='\r')
            #if t % 10 == 9:
            #    print('Epoch: %d, RMSE: %.3f' % (t, self.loss_list[-1]))np.sqrt()
            
            print('%s' % s(b, t, e, sum(self.loss_list[-bCount:])/bCount))

        return {'loss': self.loss_list}

    def predict(self, x):
        x = torch.from_numpy(x).float()
        return self(x.to(self.device)).cpu().detach().numpy()