import torch
import tensorflow
import sys
import ds

class Model(torch.nn.Module):
    """
    Linear Regressoin Module, the input features and output 
    features are defaults both 1
    """
    def __init__(self, in_features, out_features, device):
        super().__init__()
        self.seq = torch.nn.Sequential(
            torch.nn.Linear(in_features,10,1),
            torch.nn.ReLU(),
            torch.nn.Linear(10,128,4),
            torch.nn.ReLU(),
            torch.nn.Linear(128,1024,1),
            torch.nn.ReLU(),
            torch.nn.Linear(1024,out_features,1),
            torch.nn.ReLU(),
        )  

        if device:
            self.seq = self.seq.cuda(0)
        self.in_features = in_features
        self.out_features = out_features
        self.optimizer = torch.optim.SGD(self.parameters(), lr=0.06)
        self.loss_function = torch.nn.MSELoss()
        
    def forward(self,x):
        out = self.seq(x)
        return out
    
    def train(self, epochs: int, data, label, model_save_path="model.pth"):
        for epoch in range(epochs):
            prediction = self.forward(data)
            loss = self.loss_function(prediction, label)
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            if epoch % 100 == 0:
                train_correct = (prediction == label).sum().item() / data.size()[0] * 100
                print("epoch: {}, loss = {}, train_correct = {}%".format(epoch, loss.item(), train_correct))
        torch.save(self, model_save_path)

def main(): 
    args = sys.argv
    assert(len(args) > 2)

    train = ds.get_data(args[1])
    test = ds.get_data(args[2])
    
    device = torch.cuda.is_available()

    train_data = torch.from_numpy(train[:,:10]).float()
    train_label = torch.from_numpy(train[:,-1]).float().reshape(train_data.size()[0], 1)
    test_data = torch.from_numpy(test).float()
    if device:
        train_data = train_data.cuda(0)
        train_label = train_label.cuda(0)
        test_data = test_data.cuda(0)

    model = Model(10, 1, device)
    model.train(1000, train_data, train_label, "linear.pth")

if __name__ == '__main__':
    main()