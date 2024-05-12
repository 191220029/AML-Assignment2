import torch
import sys
import ds
import numpy as np
from torch.utils.data import TensorDataset, DataLoader

class Model(torch.nn.Module):
    def __init__(self, in_features, out_features, device):
        super().__init__()
        self.seq = torch.nn.Sequential(
            torch.nn.Linear(in_features, 10),
            torch.nn.Sigmoid(),
            torch.nn.Linear(10, 128),
            torch.nn.ReLU(),
            torch.nn.Linear(128, 128),
            torch.nn.ReLU(),
            torch.nn.Linear(128, out_features),
            torch.nn.Sigmoid(),
        )  

        if device:
            self.seq = self.seq.cuda(0)
        self.in_features = in_features
        self.out_features = out_features
        self.optimizer = torch.optim.SGD(self.parameters(), lr=0.01)
        self.loss_function = torch.nn.SmoothL1Loss()
        
    def forward(self,x):
        out = self.seq(x)
        return out
    
    def train(self, epochs: int, train_loader: DataLoader, model_save_path="model.pth"):
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

    label_0 = (train_label == 0).sum().item()
    label_1 = (train_label == 1).sum().item()

    data_loaders = []

    train_dataset = TensorDataset(train_data, train_label)
    entire_train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

    if label_0 == label_1:
        train_dataset = TensorDataset(train_data, train_label)
        train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
        data_loaders.append(train_loader)
    else:
        greater_label = 1
        if label_0 > label_1:
            greater_label = 0
        
        index_label_g = np.where(train_label == greater_label)[0]
        index_label_l = np.where(train_label == 1-greater_label)[0]

        i = 0
        j = index_label_l.size
        while j < index_label_g.size:
            index_g = index_label_g[i:j]

            selected_indexes = np.concatenate((index_g, index_label_l))
            new_train_data = train_data[selected_indexes]
            new_train_label = train_label[selected_indexes]
            train_dataset = TensorDataset(new_train_data, new_train_label)
            train_loader = DataLoader(train_dataset, batch_size=index_label_l.size, shuffle=True)
            data_loaders.append(train_loader)

            i += index_label_l.size
            j += index_label_l.size

    if device:
        train_data = train_data.cuda(0)
        train_label = train_label.cuda(0)
        test_data = test_data.cuda(0)

    model = Model(10, 1, device)
    for data_loader in data_loaders:
        model.train(1000, data_loader, "linear.pth")

    test_predictions = torch.round(model.forward(test_data))
    with open("522023330025.txt", "w") as f:
        for x in test_predictions.tolist():
            for y in x:
                f.write(f"{int(y)}\n")

    verify = ds.get_data("data/fit_Churn_Modelling.csv")
    verify_data = torch.from_numpy(verify[:, :10]).float()
    verify_label = torch.from_numpy(verify[:, -1]).float().reshape(verify_data.size()[0], 1)
    verify_dataset = TensorDataset(verify_data, verify_label)
    verify_loader = DataLoader(verify_dataset, batch_size=verify_label.size(dim=0), shuffle=True)

    print(f"{model.compute_accuracy(verify_loader)}%")

if __name__ == '__main__':
    main()
