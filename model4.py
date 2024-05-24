from sklearn.metrics import classification_report
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import PolynomialFeatures
import pandas as pd
import numpy as np
from sklearn.metrics import f1_score

df = pd.read_csv('data/train.csv', delimiter=',')
df.dropna()
df = df.drop(["RowNumber", "CustomerId", "Surname"], axis = 1)

df_train = df.sample(frac=0.8,random_state=200)
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
df_train[continuous_vars] = (df_train[continuous_vars]-minVec)/(maxVec-minVec)
df_train.dropna()

# data prep pipeline for test data
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
    df_predict[continuous_vars] = (df_predict[continuous_vars]-minVec)/(maxVec-minVec)
    # Ensure that The variables are ordered in the same way as was ordered in the train set
    df_predict = df_predict[df_train_Cols]
    return df_predict

def eval(pred,y_true):
    f1 = f1_score(y_true, pred)
    print("f1 : ",f1)
df_test = DfPrepPipeline(df_test,df_train.columns,minVec,maxVec)
df_test = df_test.mask(np.isinf(df_test))
df_test = df_test.dropna()

# Fit Random Forest classifier
# RF = RandomForestClassifier(bootstrap=True, class_weight=None, criterion='gini',max_depth=8, max_features=6, max_leaf_nodes=None,min_impurity_decrease=0.0,
#                             min_samples_leaf=1, min_samples_split=3,min_weight_fraction_leaf=0.0, n_estimators=50, n_jobs=None,
#                             oob_score=False, random_state=None, verbose=0,warm_start=False)
# RF.fit(df_train.loc[:, df_train.columns != 'Exited'],df_train.Exited)
# eval(RF.predict(df_test.loc[:, df_test.columns != 'Exited']), df_test.Exited)

# Fit logistic regression with pol 2 kernel
poly2 = PolynomialFeatures(degree=2)
df_train.dropna()
df_train_pol2 = poly2.fit_transform(df_train.loc[:, df_train.columns != 'Exited'])
log_pol2 = LogisticRegression(C=10, class_weight=None, dual=False, fit_intercept=True,intercept_scaling=1, max_iter=300, multi_class='warn', n_jobs=None, 
                              penalty='l2', random_state=None, solver='liblinear',tol=0.0001, verbose=0, warm_start=False)
log_pol2.fit(df_train_pol2,df_train.Exited)

df_test = pd.read_csv('data/test_verify.csv', delimiter=',')
df_test = DfPrepPipeline(df_test,df_train.columns,minVec,maxVec)
# eval(RF.predict(df_test.loc[:, df_test.columns != 'Exited']), df_test.Exited)
eval(log_pol2.predict(df_test.loc[:, df_test.columns != 'Exited']), df_test.Exited)
