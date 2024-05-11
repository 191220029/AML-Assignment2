import torch
import sys
import ds
from torch.utils.data import TensorDataset, DataLoader
from sklearn.metrics import f1_score

class Model(torch.nn.Module):
    """
    Linear Regressoin Module, the input features and output 
    features are defaults both 1
    """
    def __init__(self, in_features, out_features, device):
        super().__init__()
        self.seq = torch.nn.Sequential(
            torch.nn.Linear(in_features, 10),
            torch.nn.ReLU(),
            torch.nn.Linear(10, 128),
            torch.nn.ReLU(),
            torch.nn.Linear(128, 1024),
            torch.nn.ReLU(),
            torch.nn.Linear(1024, out_features),
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
    
    def train(self, epochs: int, train_loader, model_save_path="model.pth"):
        for epoch in range(epochs):
            for data, label in train_loader:
                if torch.cuda.is_available():
                    data = data.cuda()
                    label = label.cuda()
                
                prediction = self.forward(data)
                loss = self.loss_function(prediction, label)
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                
            acc = self.compute_accuracy(train_loader)
            print("epoch: {}, loss = {}, acc = {}".format(epoch, loss.item(), acc))
        torch.save(self, model_save_path)
        
    def compute_accuracy(self, data_loader):
        correct = 0
        total = 0
        with torch.no_grad():
            for data, label in data_loader:
                if torch.cuda.is_available():
                    data = data.cuda()
                    label = label.cuda()
                
                prediction = self.forward(data)
                # Assuming binary classification
                predicted_classes = torch.round(prediction)
                correct += (predicted_classes == label).sum().item()
                total += label.size(0)
        return (correct / total) * 100

def main(): 
    args = sys.argv
    assert(len(args) > 2)

    train = ds.get_data(args[1])
    test = ds.get_data(args[2])
    
    device = torch.cuda.is_available()

    train_data = torch.from_numpy(train[:, :10]).float()
    train_label = torch.from_numpy(train[:, -1]).float().reshape(train_data.size()[0], 1)
    test_data = torch.from_numpy(test).float()
    
    train_dataset = TensorDataset(train_data, train_label)
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    
    if device:
        train_data = train_data.cuda(0)
        train_label = train_label.cuda(0)
        test_data = test_data.cuda(0)

    model = Model(10, 1, device)
    model.train(2, train_loader, "linear.pth")


    test_predictions = model.forward(test_data)
    print(test_predictions)

if __name__ == '__main__':
    main()
