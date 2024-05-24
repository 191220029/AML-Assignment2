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
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import PolynomialFeatures

def eval(pred, y_true, mu):
    y_pred = np.where(pred[:, 1] > mu, 1, 0)
    f1 = f1_score(y_true, y_pred)
    print("f1:", f1)

df = pd.read_csv('data/train.csv', delimiter=',')
df.dropna()
df = df.drop(["RowNumber", "CustomerId", "Surname"], axis = 1)

# df_train = df.sample(frac=0.8,random_state=200)
# df_train.to_csv('data/df_train.csv')

df_train = pd.read_csv('data/df_train.csv')
df_train.dropna()
df_test = df.drop(df_train.index)
df_train['TenureByAge'] = df_train.Tenure/(df_train.Age)
df_train['CreditScoreGivenAge'] = df_train.CreditScore/(df_train.Age)
df_train['BalanceSalaryRatio'] = df_train.Balance/df_train.EstimatedSalary
continuous_vars = ['CreditScore',  'Age', 'Tenure', 'Balance','NumOfProducts', 'EstimatedSalary', 'BalanceSalaryRatio',
                   'TenureByAge','CreditScoreGivenAge']
cat_vars = ['HasCrCard', 'IsActiveMember','Geography', 'Gender']
df_train = df_train[['Exited'] + continuous_vars + cat_vars]
df_train.loc[df_train.HasCrCard == 0, 'HasCrCard'] = -1
df_train.loc[df_train.IsActiveMember == 0, 'IsActiveMember'] = -1

lst = ['Geography', 'Gender']
remove = list()
for i in lst:
    if pd.api.types.is_string_dtype(df_train[i]) or pd.api.types.is_object_dtype(df_train[i]):
        for j in df_train[i].unique():
            df_train[i + '_' + str(j)] = np.where(df_train[i] == j, 1, -1)
        remove.append(i)
df_train = df_train.drop(remove, axis=1)

minVec = df_train[continuous_vars].min().copy()
maxVec = df_train[continuous_vars].max().copy()
# df_train[continuous_vars] = (df_train[continuous_vars]-minVec)/(maxVec-minVec)

def DfPrepPipeline(df_predict,df_train_Cols,minVec,maxVec):
    # Add new features
    df_predict['BalanceSalaryRatio'] = df_predict.Balance/df_predict.EstimatedSalary
    df_predict['TenureByAge'] = df_predict.Tenure/(df_predict.Age - 17)
    df_predict['CreditScoreGivenAge'] = df_predict.CreditScore/(df_predict.Age - 17)

    # Reorder the columns
    continuous_vars = ['CreditScore','Age','Tenure','Balance','NumOfProducts','EstimatedSalary','BalanceSalaryRatio',
                   'TenureByAge','CreditScoreGivenAge']
    cat_vars = ['HasCrCard','IsActiveMember',"Geography", "Gender"] 
    df_predict = df_predict[['Exited'] + continuous_vars + cat_vars]
    # Change the 0 in categorical variables to -1
    df_predict.loc[df_predict.HasCrCard == 0, 'HasCrCard'] = -1
    df_predict.loc[df_predict.IsActiveMember == 0, 'IsActiveMember'] = -1
    # One hot encode the categorical variables
    lst = ["Geography", "Gender"]
    remove = list()
    for i in lst:
        for j in df_predict[i].unique():
            df_predict[i+'_'+j] = np.where(df_predict[i] == j,1,-1)
        remove.append(i)
    df_predict = df_predict.drop(remove, axis=1)

    # Ensure that all one hot encoded variables that appear in the train data appear in the subsequent data
    L = list(set(df_train_Cols) - set(df_predict.columns))
    for l in L:
        df_predict[str(l)] = -1        
    # MinMax scaling coontinuous variables based on min and max from the train data
    # df_predict[continuous_vars] = (df_predict[continuous_vars]-minVec)/(maxVec-minVec)
    # Ensure that The variables are ordered in the same way as was ordered in the train set
    df_predict = df_predict[df_train_Cols]
    return df_predict


df_test = DfPrepPipeline(df_test,df_train.columns,minVec,maxVec)
df_test = df_test.mask(np.isinf(df_test))
df_test = df_test.dropna()


voting_clf = VotingClassifier(estimators=[
    ('gbdt', GradientBoostingClassifier()),
    ('catboost', CatBoostClassifier(verbose=0))
], voting='soft')
# ada_clf = AdaBoostClassifier(n_estimators=100000, random_state=42, learning_rate=0.01)

# 训练和评估
voting_clf.fit(df_train.loc[:, df_train.columns != 'Exited'],df_train.Exited)
# ada_clf.fit(df_train.loc[:, df_train.columns != 'Exited'],df_train.Exited)

y_pred_voting_valid = voting_clf.predict_proba(df_test.loc[:, df_train.columns != 'Exited'])
print("---------- VotingClassifier Valid Eval ----------")
eval(y_pred_voting_valid, df_test.Exited, 0.4)
# y_pred_voting_valid = ada_clf.predict_proba(df_test.loc[:, df_train.columns != 'Exited'])
# print("---------- Adaboosting Valid Eval ----------")
# eval(y_pred_voting_valid, df_test.Exited, 0.4)

df_test=pd.read_csv("data/test_verify.csv")
df_test = DfPrepPipeline(df_test,df_train.columns,minVec,maxVec)
eval(voting_clf.predict_proba(df_test.loc[:, df_test.columns != 'Exited']), df_test.Exited, 0.4)

RF = RandomForestClassifier(bootstrap=True, class_weight=None, criterion='gini',max_depth=8, max_features=6, max_leaf_nodes=None,min_impurity_decrease=0.0,
                            min_samples_leaf=1, min_samples_split=3,min_weight_fraction_leaf=0.0, n_estimators=50, n_jobs=None,
                            oob_score=False, random_state=None, verbose=0,warm_start=False)
RF.fit(df_train.loc[:, df_train.columns != 'Exited'],df_train.Exited)
eval(RF.predict_proba(df_test.loc[:, df_test.columns != 'Exited']), df_test.Exited, 0.4)

poly2 = PolynomialFeatures(degree=2)
df_train.dropna()
df_train_pol2 = poly2.fit_transform(df_train.loc[:, df_train.columns != 'Exited'])
log_pol2 = LogisticRegression(C=10, class_weight=None, dual=False, fit_intercept=True,intercept_scaling=1, max_iter=300, multi_class='auto', n_jobs=None, 
                              penalty='l2', random_state=None, solver='liblinear',tol=0.0001, verbose=0, warm_start=False)
log_pol2.fit(df_train_pol2,df_train.Exited)
df_test_pol2 = poly2.fit_transform(df_test.loc[:, df_train.columns != 'Exited'])
eval(log_pol2.predict_proba(df_test_pol2), df_test.Exited, 0.4)


