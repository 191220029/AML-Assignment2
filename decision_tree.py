import numpy as np
import sys
import ds
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, f1_score

class Model:
    def __init__(self):
        self.model = DecisionTreeClassifier()

    def train(self, train_data, train_label):
        self.model.fit(train_data, train_label.ravel())
    
    def predict(self, data):
        return self.model.predict(data)
    
    def compute_metrics(self, data, labels):
        predictions = self.predict(data)
        accuracy = accuracy_score(labels, predictions) * 100
        f1 = f1_score(labels, predictions)
        return accuracy, f1

def main(): 
    args = sys.argv
    assert(len(args) > 2)

    # Load data
    train = ds.get_data(args[1])
    test = ds.get_data(args[2])
    
    train_data = train[:, :-1]
    train_label = train[:, -1]
    test_data = test

    # Handle class imbalance by oversampling the minority class
    label_0 = (train_label == 0).sum()
    label_1 = (train_label == 1).sum()

    if label_0 > label_1:
        greater_label = 0
        lesser_label = 1
    else:
        greater_label = 1
        lesser_label = 0
    
    index_label_g = np.where(train_label == greater_label)[0]
    index_label_l = np.where(train_label == lesser_label)[0]

    # Oversample the lesser label to balance the dataset
    new_indices = np.concatenate([index_label_g, np.tile(index_label_l, label_0 // label_1)])
    new_train_data = train_data[new_indices]
    new_train_label = train_label[new_indices]

    # Train the decision tree model
    model = Model()
    model.train(new_train_data, new_train_label)

    # Evaluate the model on the validation set
    accuracy, f1 = model.compute_metrics(new_train_data, new_train_label)
    print(f"Validation Accuracy on train data: {accuracy:.2f}%, F1 Score: {f1:.2f}")

    # Predict on the test data
    test_predictions = model.predict(test_data)
    with open("522023330025.txt", "w") as f:
        for pred in test_predictions:
            f.write(f"{int(pred)}\n")
    
    # Verify the model on a separate dataset
    verify = ds.get_data("data/fit_test_verify.csv")
    verify_data = verify[:, :-1]
    verify_label = verify[:, -1]

    accuracy, f1 = model.compute_metrics(verify_data, verify_label)
    print(f"Verification Accuracy on test: {accuracy:.2f}%, F1 Score: {f1:.2f}")

    verify = ds.get_data("data/fit_Churn_Modelling.csv")
    verify_data = verify[:, :-1]
    verify_label = verify[:, -1]

    accuracy, f1 = model.compute_metrics(verify_data, verify_label)
    print(f"Verification Accuracy on entire dataset: {accuracy:.2f}%, F1 Score: {f1:.2f}")

    predict_label = model.predict(verify_data)
    with open("predict.txt", "w") as f:
        for pred in predict_label:
            f.write(f"{int(pred)}\n")


if __name__ == '__main__':
    main()
