import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
import numpy as np
from sklearn.metrics import f1_score
import copy
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier

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
        
        self.Handling_missing_values()

        print("-"*50)
        print("selected_columns : ",self.selected_columns)
        print("DataFrame.column : ",self.data.columns)
    
    def Handling_missing_values(self):
        self.data = self.data.dropna()

    def print(self):
        dataset = self.data[self.selected_columns]
        print(dataset)
    def describe(self):
        # 使用describe()函数获取每一列的统计信息，包括最大值和最小值
        column_stats = self.data[self.selected_columns].describe()
        # 使用value_counts()函数获取每一列不同属性值的数量和属性值列表
        column_value_counts = {}
        for column in self.selected_columns:
            value_counts = self.data[column].value_counts()
            column_value_counts[column] = {
                'count': len(value_counts),
                'values': value_counts.index.tolist()
            }
        print("-"*20)
        # 打印每一列的统计信息
        print("列的最大值和最小值：")
        print(column_stats)

        # 打印每一列不同属性值的数量和属性值列表
        print("每一列不同属性值的数量和属性值列表：")
        for column, info in column_value_counts.items():
            print(f"列名: {column}")
            print(f"不同属性值的数量: {info['count']}")
            if(info['count'] <= 10):
                print(f"属性值列表: {info['values']}")
            else:
                print(f"属性值列表: {info['values'][:10]} ...")
            print()
        print("每一列的属性类型:")
        print(self.data[self.selected_columns].dtypes)

    def discretization(self):
        print("-"*50)
        for column in self.selected_columns:
            data_column = self.data[column]
            if(data_column.dtype == 'str' or data_column.dtype == 'object'):
                print(f'Column [{column}] need to be discretized (No Number Relationship! If needed, please code separately)')
                self.data[column] = self.data[column].astype('category').cat.codes
    def normalized(self):
        legal_columns = copy.deepcopy(self.selected_columns)
        if(self.label_column in legal_columns):
            legal_columns.remove(self.label_column)
        mean = self.data[legal_columns].mean()
        std = self.data[legal_columns].std()
        self.data[legal_columns] = (self.data[legal_columns] - mean)/std
    def getx(self):
        legal_columns = copy.deepcopy(self.selected_columns)
        if(self.label_column in legal_columns):
            legal_columns.remove(self.label_column)
        return self.data[legal_columns].to_numpy()
    def gety(self):
        return self.data[self.label_column].to_numpy()
    def __len__(self):
        return len(self.data)

def eval(pred,y_true,mu):
    # print(pred.shape,y_true.shape)
    y_pred = np.where(pred[:,1] > mu,1,0)
    y_score = pred[:,1]
    # accuracy = accuracy_score(y_true, y_pred)
    # recall = recall_score(y_true, y_pred)
    # precision = precision_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)
    # p,r,_ = precision_recall_curve(y_true,y_score)
    # auprc = auc(r,p)
    # print("acc : ",accuracy)
    # print("recall : ",recall)
    # print("precision : ",precision)
    print("f1 : ",f1)
    # print("auprc : ",auprc)


train_field = ['Surname','CreditScore','Geography','Gender','Age',\
               'Tenure','Balance','NumOfProducts','HasCrCard','IsActiveMember','EstimatedSalary', 'Exited']

# 读取并处理训练集数据
train_dataset = CustomDataset('./data/train.csv', train_field)
train_dataset.describe()
train_dataset.discretization()
train_dataset.normalized()
train_dataset.describe()

data, target = train_dataset.getx(), train_dataset.gety()
X_train, X_valid, y_train, y_valid = train_test_split(data, target, test_size=0.2, random_state=42, shuffle=True)

# 定义各模型
clf_knn = KNeighborsClassifier()
clf_nb = GaussianNB()
clf_dt = DecisionTreeClassifier()
clf_rf = RandomForestClassifier(random_state=42)
clf_lr = LogisticRegression()
clf_svm = SVC(probability=True)
clf_mlp = MLPClassifier()
clf_gbdt = GradientBoostingClassifier()
clf_xgb = XGBClassifier()
clf_lgb = LGBMClassifier()
clf_catboost = CatBoostClassifier(verbose=0)

# 组合成投票分类器
voting_clf = VotingClassifier(estimators=[
    ('knn', clf_knn),
    ('nb', clf_nb),
    ('dt', clf_dt),
    ('rf', clf_rf),
    ('lr', clf_lr),
    ('svm', clf_svm),
    ('mlp', clf_mlp),
    ('gbdt', clf_gbdt),
    ('xgb', clf_xgb),
    ('lgb', clf_lgb),
    ('catboost', clf_catboost)
], voting='soft')

# 训练和评估
voting_clf.fit(X_train, y_train)
y_pred_voting_valid = voting_clf.predict_proba(X_valid)
print("---------- VotingClassifier Valid Eval ----------")
eval(y_pred_voting_valid, y_valid, 0.3)
