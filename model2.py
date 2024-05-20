import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
import numpy as np
from sklearn.metrics import f1_score
import copy
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier, GradientBoostingRegressor
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier
from sklearn.preprocessing import LabelEncoder
from imblearn.over_sampling import RandomOverSampler
from imblearn.under_sampling import RandomUnderSampler

# 自定义数据集类省略，假设已定义
class CustomDataset():
    def __init__(self, file_path, selected_columns=None, label_column=None):
        self.data = pd.read_csv(file_path)  # 读取CSV文件
        self.selected_columns = selected_columns
        self.label_column = label_column
        if(self.selected_columns == None):
            self.selected_columns = self.data.columns
        if(self.label_column == None):
            self.label_column = self.selected_columns[-1]
        
        self.data_full = self.data.copy()  # 保存完整的数据
        self.Handling_missing_values()


    def predict_missing_values(self):
        # 用训练好的模型预测缺失值
        missing_data = self.data_full[self.data_full.isna().any(axis=1)]
        non_missing_data = self.data_full[self.selected_columns].dropna()
        
        for column in self.selected_columns:
            if column == self.label_column:
                continue

            cur_missing_data = self.data_full[self.data_full[column].isna()]
            if(cur_missing_data.size == 0):
                continue

            model = GradientBoostingRegressor(random_state=42)
            y_non_missing = non_missing_data[column]
                
            # 拆分有缺失值的列为特征和标签
            X_non_missing = non_missing_data.drop(columns=[column])
            
            # 训练模型
            model.fit(X_non_missing, y_non_missing)
            
            # 预测缺失值
            X_missing = cur_missing_data[self.selected_columns].drop(columns=[column])
            predicted = model.predict(X_missing)
            cur_missing_data.loc[:, column] = predicted
            self.data_full = pd.concat([non_missing_data, cur_missing_data], axis=0)
    
        self.data_full = self.data_full[self.selected_columns]
        train_dataset.data = train_dataset.data_full
    
    def Handling_missing_values(self):
        self.data = self.data.dropna()


    def print(self):
        dataset = self.data[self.selected_columns]
        print(dataset)
    def discretization(self):
        for column in self.selected_columns:
            data_column = self.data[column]
            if(data_column.dtype == 'str' or data_column.dtype == 'object'):
                self.data[column] = self.data[column].astype('category').cat.codes
            data_column = self.data_full[column]
            if(data_column.dtype == 'str' or data_column.dtype == 'object'):
                self.data_full[column] = self.data_full[column].astype('category').cat.codes
    def normalized(self):
        legal_columns = copy.deepcopy(self.selected_columns)
        if(self.label_column in legal_columns):
            legal_columns.remove(self.label_column)
        mean = self.data[legal_columns].mean()
        std = self.data[legal_columns].std()
        self.data[legal_columns] = (self.data[legal_columns] - mean)/std
        mean = self.data_full[legal_columns].mean()
        std = self.data_full[legal_columns].std()
        self.data_full[legal_columns] = (self.data_full[legal_columns] - mean)/std
    
    def getx(self):
        legal_columns = copy.deepcopy(self.selected_columns)
        if(self.label_column in legal_columns):
            legal_columns.remove(self.label_column)
        return self.data[legal_columns].to_numpy()
    def gety(self):
        return self.data[self.label_column].to_numpy()
    def __len__(self):
        return len(self.data)

class CustomTestDataset():
    def __init__(self, file_path, selected_columns=None):
        self.data = pd.read_csv(file_path)  # 读取CSV文件
        self.selected_columns = selected_columns
        if(self.selected_columns == None):
            self.selected_columns = self.data.columns
        
        self.data_full = self.data.copy()  # 保存完整的数据

    def discretization(self):
        for column in self.selected_columns:
            data_column = self.data[column]
            if(data_column.dtype == 'str' or data_column.dtype == 'object'):
                self.data[column] = self.data[column].astype('category').cat.codes
    def normalized(self):
        legal_columns = copy.deepcopy(self.selected_columns)
        mean = self.data[legal_columns].mean()
        std = self.data[legal_columns].std()
        self.data[legal_columns] = (self.data[legal_columns] - mean)/std
    def getx(self):
        legal_columns = copy.deepcopy(self.selected_columns)
        return self.data[legal_columns].to_numpy()
    def __len__(self):
        return len(self.data)


def eval(pred,y_true,mu):
    y_pred = np.where(pred[:,1] > mu,1,0)
    f1 = f1_score(y_true, y_pred)
    print("f1 : ",f1)

def save_pred(pred, mu, pth):
    y_pred = np.where(pred[:,1] > mu,1,0)
    with open(pth, "w") as f:
        for x in y_pred.tolist():
            f.write(f"{int(x)}\n")

train_field = ['CreditScore','Geography','Gender','Age',\
               'Tenure','Balance','NumOfProducts','HasCrCard','IsActiveMember','EstimatedSalary', 'Exited']
test_field = ['CreditScore','Geography','Gender','Age',\
               'Tenure','Balance','NumOfProducts','HasCrCard','IsActiveMember','EstimatedSalary']

# 读取并处理训练集数据
train_dataset = CustomDataset('./data/train.csv', train_field)
train_dataset.discretization()
train_dataset.normalized()

# 定义各模型
clf_knn = KNeighborsClassifier()
clf_nb = GaussianNB()
clf_dt = DecisionTreeClassifier()
clf_rf = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42)
clf_xgbc = XGBClassifier()
clf_lr = LogisticRegression()
clf_svm = SVC(probability=True)
clf_mlp = MLPClassifier()
clf_gbdt = GradientBoostingClassifier()
clf_xgb = XGBClassifier()
clf_lgb = LGBMClassifier()
clf_catboost = CatBoostClassifier(verbose=0)

# train_dataset.predict_missing_values()
# train_dataset.normalized()

data, target = train_dataset.getx(), train_dataset.gety()
X_train, X_valid, y_train, y_valid = train_test_split(data, target, test_size=0.2, random_state=42, shuffle=True)



# oversample = RandomOverSampler(random_state=42)
# X_train, y_train = oversample.fit_resample(X_train, y_train)
# undersample = RandomUnderSampler(random_state=42)
# X_train, y_train = undersample.fit_resample(X_train, y_train)

voting_clf = VotingClassifier(estimators=[
    # ('knn', clf_knn),
    # ('nb', clf_nb),
    # ('dt', clf_dt),
    # ('rf', clf_rf),
    # ('lr', clf_lr),
    # ('svm', clf_svm),
    # ('mlp', clf_mlp),
    # ('xgbc', clf_xgbc),
    # ('lgb', clf_lgb),
    ('gbdt', clf_gbdt),
    ('catboost', clf_catboost)
], voting='soft')


thre = 0.4

# 训练和评估
voting_clf.fit(X_train, y_train)


y_pred_voting_valid = voting_clf.predict_proba(X_valid)
print("---------- VotingClassifier Valid Eval ----------")
eval(y_pred_voting_valid, y_valid, thre)

verify_dataset = CustomDataset('./data/test_verify.csv', train_field)
verify_dataset.discretization()
verify_dataset.normalized()
_, target = verify_dataset.getx(), verify_dataset.gety()
# y_pred_voting_valid = voting_clf.predict_proba(data)
# print("---------- VotingClassifier Valid Eval ----------")
# eval(y_pred_voting_valid, target, thre)
# save_pred(y_pred_voting_valid, thre, 'verify.txt')

test_dataset = CustomTestDataset('./data/test.csv', test_field)
test_dataset.discretization()
test_dataset.normalized()
data=test_dataset.getx()
y_pred_voting_valid = voting_clf.predict_proba(data)
print("---------- VotingClassifier Test Eval ----------")
eval(y_pred_voting_valid, target, thre)
save_pred(y_pred_voting_valid, thre, '522023330025.txt')


