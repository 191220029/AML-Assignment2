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
from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve
import matplotlib.pyplot as plt

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
            if column == self.label_column or not missing_data[column].isna().any():
                continue

            cur_missing_data = self.data_full[self.data_full[column].isna()]
            if(cur_missing_data.size == 0):
                continue
            # model = GradientBoostingRegressor(random_state=42)
            model = CatBoostClassifier(verbose=0, learning_rate=0.14, max_depth=8, n_estimators=105)
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
        # print("-"*50)
        for column in self.selected_columns:
            data_column = self.data[column]
            if(data_column.dtype == 'str' or data_column.dtype == 'object'):
                # print(f'Column [{column}] need to be discretized (No Number Relationship! If needed, please code separately)')
                self.data[column] = self.data[column].astype('category').cat.codes
            data_column = self.data_full[column]
            if(data_column.dtype == 'str' or data_column.dtype == 'object'):
                # print(f'Column [{column}] need to be discretized (No Number Relationship! If needed, please code separately)')
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
    f1 = f1_score(y_true, y_pred, average='macro')
    print("f1 : ",f1)
    return f1

def save_pred(pred, mu, pth):
    y_pred = np.where(pred[:,1] > mu,1,0)
    with open(pth, "w") as f:
        for x in y_pred.tolist():
            f.write(f"{int(x)}\n")

def get_auc_scores(y_actual, method,method2):
    auc_score = roc_auc_score(y_actual, method); 
    fpr_df, tpr_df, _ = roc_curve(y_actual, method2); 
    return (auc_score, fpr_df, tpr_df)

train_field = ['CreditScore','Geography','Gender','Age',\
               'Tenure','Balance','NumOfProducts','HasCrCard','IsActiveMember','EstimatedSalary', 'Exited']
test_field = ['CreditScore','Geography','Gender','Age',\
               'Tenure','Balance','NumOfProducts','HasCrCard','IsActiveMember','EstimatedSalary']

thre = 0.4
train_dataset = CustomDataset('./data/train.csv', train_field)
train_dataset.discretization()
train_dataset.normalized()

clf_knn = KNeighborsClassifier()
clf_nb = GaussianNB()
clf_dt = DecisionTreeClassifier()
clf_rf = RandomForestClassifier(random_state=42)
clf_lr = LogisticRegression()
clf_svm = SVC(probability=True)
clf_mlp = MLPClassifier(beta_1=0.7, beta_2=0.777)
clf_gbdt = GradientBoostingClassifier(verbose=0, learning_rate=0.1, random_state=42, n_estimators=80)
clf_xgb = XGBClassifier()
clf_lgb = LGBMClassifier(verbose=0)
clf_catboost = CatBoostClassifier(verbose=0, learning_rate=0.14, max_depth=8, n_estimators=105)

train_dataset.discretization()
train_dataset.normalized()
# train_dataset.predict_missing_values()
# train_dataset.normalized()

data, target = train_dataset.getx(), train_dataset.gety()
X_train, X_valid, y_train, y_valid = train_test_split(data, target, test_size=0.2, random_state=42, shuffle=True)

# oversample = RandomOverSampler(random_state=42)
# X_train, y_train = oversample.fit_resample(X_train, y_train)
# undersample = RandomUnderSampler(random_state=42)
# X_train, y_train = undersample.fit_resample(X_train, y_train)


verify_dataset = CustomDataset('./data/test_verify.csv', train_field)
verify_dataset.discretization()
verify_dataset.normalized()
_, target = verify_dataset.getx(), verify_dataset.gety()

test_dataset = CustomTestDataset('./data/test.csv', test_field)
test_dataset.discretization()
test_dataset.normalized()
test_data=test_dataset.getx()

plt.figure(dpi=800,figsize=(14,8))
estimators = [
    # [[('gbdt', GradientBoostingClassifier(verbose=0, learning_rate=0.1, random_state=42, n_estimators=80))], 'gdbt'],
    # [[('catboost', CatBoostClassifier(verbose=0, learning_rate=0.14, max_depth=8, n_estimators=105))], 'catboost'],
    # [[('lgbm', LGBMClassifier(verbose=0))], 'lgbm'],
    # [[('mlp', MLPClassifier(beta_1=0.8, beta_2=0.888))], 'mlp'],
    [[('gbdt', GradientBoostingClassifier(verbose=0, learning_rate=0.1, random_state=42, n_estimators=80)),('catboost', CatBoostClassifier(verbose=0, learning_rate=0.14, max_depth=8, n_estimators=105))], 'catboost+gdbt'],
    # [[('gbdt', GradientBoostingClassifier(verbose=0, learning_rate=0.1, random_state=42, n_estimators=80)),('catboost', LGBMClassifier(verbose=0))], 'lgbm+gdbt'],
    # [[('lgbm', LGBMClassifier(verbose=0)),('catboost', CatBoostClassifier(verbose=0, learning_rate=0.14, max_depth=8, n_estimators=105))], 'catboost+lgbm'],
    # [[('gbdt', GradientBoostingClassifier(verbose=0, learning_rate=0.1, random_state=42, n_estimators=80)),('catboost', CatBoostClassifier(verbose=0, learning_rate=0.14, max_depth=8, n_estimators=105)),('mlp', MLPClassifier(beta_1=0.8, beta_2=0.888))], 'catboost+gdbt+mlp'],
    # [[('gbdt', GradientBoostingClassifier(verbose=0, learning_rate=0.1, random_state=42, n_estimators=80)),('catboost', CatBoostClassifier(verbose=0, learning_rate=0.14, max_depth=8, n_estimators=105)),('lgbm', LGBMClassifier(verbose=0))], 'catboost+gdbt+lgbm'],
    # [[('catboost', CatBoostClassifier(verbose=0, learning_rate=0.14, max_depth=8, n_estimators=105)),('lgbm', LGBMClassifier(verbose=0)),('mlp', MLPClassifier(beta_1=0.8, beta_2=0.888))], 'catboost+lgbm+mlp'],
    # [[('gbdt', GradientBoostingClassifier(verbose=0, learning_rate=0.1, random_state=42, n_estimators=80)),('lgbm', LGBMClassifier(verbose=0)),('mlp', MLPClassifier(beta_1=0.8, beta_2=0.888))], 'gdbt+lgbm+mlp'],
    # [[('gbdt', GradientBoostingClassifier(verbose=0, learning_rate=0.1, random_state=42, n_estimators=80)),('catboost', CatBoostClassifier(verbose=0, learning_rate=0.14, max_depth=8, n_estimators=105)),('lgbm', LGBMClassifier(verbose=0)),('mlp', MLPClassifier(beta_1=0.8, beta_2=0.888))], 'catboost+gdbt+lgbm+mlp'],
]

best_f1 = 0.0
best_model = ''
for estimator in estimators:
    voting_clf = VotingClassifier(estimators=estimator[0], voting='soft')
    voting_clf.fit(X_train, y_train)
    
    y_pred_voting_valid = voting_clf.predict_proba(X_valid)
    print(f"---------- {estimator[1]} Verify Eval ----------")
    f1 = eval(y_pred_voting_valid, y_valid, thre)

    y_pred_voting_valid = voting_clf.predict_proba(test_data)
    print(f"---------- {estimator[1]} Test Eval ----------")
    f1 = eval(y_pred_voting_valid, target, thre)
    if f1 > best_f1:
        print(f"save predicts with better macro-f1={f1}")
        save_pred(y_pred_voting_valid, thre, '522023330025.txt')
        best_f1 = f1
        best_model = estimator[1]

    # auc_vote, fpr_vote, tpr_vote = get_auc_scores(target, voting_clf.predict(test_data),voting_clf.predict_proba(test_data)[:,1])
    # plt.plot(fpr_vote, tpr_vote, label = estimator[1] + ' Score: ' + str(round(auc_vote, 5)))
    auc_vote, fpr_vote, tpr_vote = get_auc_scores(y_train, voting_clf.predict(X_train),voting_clf.predict_proba(X_train)[:,1])
    plt.plot(fpr_vote, tpr_vote, label = estimator[1] + ' Score: ' + str(round(auc_vote, 5)))

    

plt.plot([0,1], [0,1], 'k--', label = 'Random: 0.5')

plt.xlabel('False positive rate')
plt.ylabel('True positive rate')
plt.title('ROC Curve')
plt.legend(loc='best')
plt.savefig('roc_results_ratios.png')

print(f"Best model is {best_model}, macro-f1={best_f1}")

# voting_clf = VotingClassifier(estimators=[
#     # ('knn', clf_knn),
#     # ('nb', clf_nb),
#     # ('dt', clf_dt),
#     # ('rf', clf_rf),
#     # ('lr', clf_lr),
#     # ('svm', clf_svm),
#     # ('mlp', clf_mlp),
#     ('gbdt', clf_gbdt),
#     ('catboost', clf_catboost),
#     # ('rf', RandomForestClassifier(max_depth=5, max_features=6, n_estimators=104,)),
#     ('mlp', clf_mlp)
# ], voting='soft')



# # 训练和评估
# voting_clf.fit(X_train, y_train)
# y_pred_voting_valid = voting_clf.predict_proba(X_valid)
# print("---------- VotingClassifier Valid Eval ----------")
# eval(y_pred_voting_valid, y_valid, thre)


# y_pred_voting_valid = voting_clf.predict_proba(test_data)
# print("---------- VotingClassifier Test Eval ----------")
# eval(y_pred_voting_valid, target, thre)
# save_pred(y_pred_voting_valid, thre, '522023330025.txt')
