import pandas as pd
import copy
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score
from sklearn.linear_model import LogisticRegression, RidgeClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier
from sklearn.preprocessing import LabelEncoder
from imblearn.over_sampling import RandomOverSampler
from mlxtend.feature_selection import SequentialFeatureSelector as SFS
from catboost import CatBoostClassifier
from scipy.sparse import csr_matrix
from sklearn.impute import SimpleImputer

class CustomDataset:
    def __init__(self, file_path, selected_columns=None, label_column=None):
        self.data = pd.read_csv(file_path)
        self.selected_columns = selected_columns
        self.label_column = label_column
        if self.selected_columns is None:
            self.selected_columns = self.data.columns
        if self.label_column is None:
            self.label_column = self.selected_columns[-1]
        
        self.data_full = self.data.copy()
        self.Handling_missing_values()

    def predict_missing_values(self):
        data_matrix = self.data_full[self.selected_columns].values
        mask = np.isnan(data_matrix)
        
        # 简单填补初始缺失值，避免SVD错误
        imp = SimpleImputer(strategy="mean")
        data_matrix_imputed = imp.fit_transform(data_matrix)
        
        # 使用SVD进行矩阵补全
        U, sigma, Vt = np.linalg.svd(data_matrix_imputed, full_matrices=False)
        sigma = np.diag(sigma)
        
        # 选择合适的秩，进行低秩近似
        k = np.linalg.matrix_rank(sigma) // 2
        U_k = U[:, :k]
        sigma_k = sigma[:k, :k]
        Vt_k = Vt[:k, :]
        
        # 重建矩阵
        data_matrix_reconstructed = np.dot(U_k, np.dot(sigma_k, Vt_k))
        
        # 仅填补原始数据中的缺失值
        data_matrix[mask] = data_matrix_reconstructed[mask]
        self.data_full[self.selected_columns] = data_matrix
        self.data = self.data_full

    def Handling_missing_values(self):
        self.data = self.data.dropna()

    def discretization(self):
        for column in self.selected_columns:
            data_column = self.data[column]
            if data_column.dtype == 'str' or data_column.dtype == 'object':
                self.data[column] = self.data[column].astype('category').cat.codes
            data_column = self.data_full[column]
            if data_column.dtype == 'str' or data_column.dtype == 'object':
                self.data_full[column] = self.data_full[column].astype('category').cat.codes

    def normalized(self):
        legal_columns = copy.deepcopy(self.selected_columns)
        if self.label_column in legal_columns:
            legal_columns.remove(self.label_column)
        mean = self.data[legal_columns].mean()
        std = self.data[legal_columns].std()
        self.data[legal_columns] = (self.data[legal_columns] - mean) / std
        mean = self.data_full[legal_columns].mean()
        std = self.data_full[legal_columns].std()
        self.data_full[legal_columns] = (self.data_full[legal_columns] - mean) / std
    
    def getx(self):
        legal_columns = copy.deepcopy(self.selected_columns)
        if self.label_column in legal_columns:
            legal_columns.remove(self.label_column)
        
        # 使用LabelEncoder对分类特征进行编码
        label_encoders = {}
        for column in legal_columns:
            if self.data[column].dtype == 'object':
                label_encoders[column] = LabelEncoder()
                self.data[column] = label_encoders[column].fit_transform(self.data[column])
        data_sparse = csr_matrix(self.data[legal_columns].values)  # 将数据转换为稀疏矩阵
        return data_sparse

    def gety(self):
        return self.data[self.label_column].to_numpy()

    def __len__(self):
        return len(self.data)

def eval(pred, y_true, mu):
    y_pred = np.where(pred[:, 1] > mu, 1, 0)
    f1 = f1_score(y_true, y_pred)
    print("f1:", f1)

train_field = ['CreditScore', 'Geography', 'Gender', 'Age', 'Tenure', 'Balance', 'NumOfProducts', 'HasCrCard', 'IsActiveMember', 'EstimatedSalary', 'Exited']

# 读取并处理训练集数据
train_dataset = CustomDataset('./data/train.csv', train_field)
train_dataset.discretization()
train_dataset.predict_missing_values()  # 使用矩阵补全
train_dataset.normalized()

data, target = train_dataset.getx(), train_dataset.gety()
X_train, X_valid, y_train, y_valid = train_test_split(data, target, test_size=0.2, random_state=42, shuffle=True)

# 处理标签不平衡问题
# oversample = RandomOverSampler(random_state=42)
# X_train, y_train = oversample.fit_resample(X_train, y_train)

# 根据选择的特征重新构建数据集
X_train_selected = X_train
X_valid_selected = X_valid

# 定义投票分类器，添加岭回归模型
voting_clf = VotingClassifier(estimators=[
    ('gbdt', GradientBoostingClassifier()),
    ('catboost', CatBoostClassifier(verbose=0))
], voting='soft')

# 训练和评估
voting_clf.fit(X_train_selected, y_train)

y_pred_voting_valid = voting_clf.predict_proba(X_valid_selected)
print("---------- VotingClassifier Valid Eval ----------")
eval(y_pred_voting_valid, y_valid, 0.4)
