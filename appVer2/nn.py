import sys
import pandas as pd
import numpy as np
import glob
import time
import pickle
import statistics
from matplotlib import pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.multioutput import MultiOutputClassifier
from sklearn.model_selection import LeaveOneGroupOut
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from imblearn.over_sampling import SMOTE
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from imblearn.under_sampling import RandomUnderSampler
from sklearn.datasets import load_boston
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import AdaBoostRegressor
from sklearn.ensemble import BaggingRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
import lightgbm as lgb
from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
from scipy import stats
import itertools
import shap
from statistics import mean

from sklearn.ensemble import RandomForestClassifier
from sklearn.multioutput import MultiOutputClassifier
from sklearn.model_selection import LeaveOneGroupOut
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
from sklearn import metrics
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from imblearn.over_sampling import SMOTE
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from imblearn.under_sampling import RandomUnderSampler
from sklearn.datasets import load_boston
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import AdaBoostRegressor
from sklearn.ensemble import BaggingRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.model_selection import BaseCrossValidator
from imblearn.under_sampling import RandomUnderSampler
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.metrics import roc_auc_score

import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

import optuna
from sklearn.model_selection import cross_val_score
from functools import partial

import joblib


np.set_printoptions(threshold=np.inf)
pd.set_option('display.max_rows', 100)
pd.set_option('display.max_columns', 100)

feature_all=[]
cm_all=[]
accuracy_all=[]
precision_all=[]
recall_all=[]
f1_all=[]
identification=[]

feature=[]
cm=[]
accuracy=[]
precision=[]
recall=[]
f1=[]
classification=[]


def data_get_drop_morning(feature_drop):
    file = []
    for i in range(1,6):
        for s in [1,2]:
            if i==4 and s==2:
                pass
            else:
                f = open('分析データ/' + 'ID' + str(i) + '/' + 'SEX' + str(s) + '/' + 'Morning20221118'  + '.pickle' ,'rb')
                file.append(pickle.load(f))
                f.close()
    
    file = [j.drop(feature_drop,axis=1) for j in file]

    df_list = [i.dropna(how='any').replace('nan',0) for i in file]
    df_list00 = [i[["M1","M2","M3"]].astype('float64').astype('int64') for i in df_list]
    df1_list=[i[['M1']].replace(0,1).replace(1,1).replace(2,1).replace(3,2).replace(4,3).replace(5,3) for i in df_list00]
    df2_list=[i[['M2']].replace(0,1).replace(1,1).replace(2,1).replace(3,2).replace(4,3).replace(5,3) for i in df_list00]
    df3_list=[i[['M3']].replace(0,1).replace(1,1).replace(2,2).replace(3,2).replace(4,2).replace(5,2) for i in df_list00]



    yN1_list = [i[["M1"]] for i in df1_list]
    yN2_list = [i[["M2"]] for i in df2_list]
    yN3_list = [i[["M3"]] for i in df3_list]
    X_list = [j.drop([ "M1","M2","M3"], axis=1) for j in df_list]
    X = np.concatenate(X_list)
    #print([i[['M1']] for i in df_list00])

    y1 = np.concatenate(yN1_list)
    y2 = np.concatenate(yN2_list)
    y3 = np.concatenate(yN3_list)
    X=X.astype(float)

    #print(y3)
    #アップサンプリングあり
    cols = X_list[0].columns
    #X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.7)
    groups = np.arange(len(y1))%3
    logo = LeaveOneGroupOut()
    #print(groups)
    return X,y1,y2,y3,cols,groups,logo




# optunaの目的関数を設定する
def objective(X_train, y_train,evaluation,trial):
    #criterion = trial.suggest_categorical('criterion', ['mse', 'mae'])
   # criterion = trial.suggest_categorical("criterion", ["gini", "entropy"])
    #bootstrap = trial.suggest_categorical('bootstrap',['True','False'])
    max_depth = trial.suggest_int('max_depth', 1, 1000)
    #max_features = trial.suggest_categorical('max_features', ['auto', 'sqrt','log2'])
    max_leaf_nodes = trial.suggest_int('max_leaf_nodes', 1,1000)
    n_estimators =  trial.suggest_int('n_estimators', 1, 1000)
    min_samples_split = trial.suggest_int('min_samples_split',2,5)
    min_samples_leaf = trial.suggest_int('min_samples_leaf',1,10)
    
    y_train=y_train.ravel()
    #print(y_train)
    regr = RandomForestClassifier(#bootstrap = bootstrap, criterion = criterion,
                                 max_depth = max_depth, #max_features = max_features,
                                 max_leaf_nodes = max_leaf_nodes,n_estimators = n_estimators,
                                 min_samples_split = min_samples_split,min_samples_leaf = min_samples_leaf)

    score = cross_val_score(regr, X_train, y_train, cv=3, scoring=evaluation)
    f1_macro = score.mean()
    #print(f1_macro)

    return f1_macro
#https://www.haya-programming.com/entry/2019/06/22/235058

def bagging(X_train, y_train,seed):
    f = partial(objective, X_train, y_train,evaluation)#optuna search cv
#    f = lambda  x:objective(X_train, y_train,evaluation,x)
    #optunaで学習
    if n_trials_num==0:
        optimised_rf = RandomForestClassifier()
    else:
        study = optuna.create_study()
        study.optimize(f, n_trials=n_trials_num)
        # チューニングしたハイパーパラメーターをフィット
        optimised_rf = RandomForestClassifier(#criterion = study.best_params['criterion'], bootstrap = study.best_params['bootstrap'], 
            max_depth = study.best_params['max_depth'], #max_features = study.best_params['max_features'],
            max_leaf_nodes = study.best_params['max_leaf_nodes'],n_estimators = study.best_params['n_estimators'],
            min_samples_split = study.best_params['min_samples_split'],min_samples_leaf = study.best_params['min_samples_leaf'],
            n_jobs=2)
    
    if down_mode==1:    
        sampler = RandomUnderSampler(random_state=seed, replacement=True)
        X_resampled, y_resampled = sampler.fit_resample(X_train, y_train)
    else:
        X_resampled = X_train 
        y_resampled = y_train
    
    model_bagging = optimised_rf.fit(X_resampled, y_resampled)
    return model_bagging

def RandomForestClassifier_optuna(up_mode, down_mode, bagging_trials_num, n_trials_num, evaluation,PCA_mode=0):
    for i,y in enumerate([y1]):#,y3,y4]):
        tmp_1 = []
        tmp_2 = []
        y_pred_sum=[]
        y_test_sum=[]
        
        for train_index, test_index in logo.split(X, y, groups):      
            X_train, X_test = X[train_index], X[test_index]
            y_train, y_test = y[train_index], y[test_index]
            #print(X_train)
            # モデルのインスタンスの作成
            #model = RandomForestClassifier()
            #model = lgb.LGBMClassifier() 
            
            #アップサンプリング
            if up_mode==1 and i==0:
                resampler = SMOTE(sampling_strategy={1:100, 2:50,3:100},k_neighbors=3,random_state=0)
                X_res, y_res = resampler.fit_resample(X_train, y_train)
            elif up_mode==1 and i==1:
                resampler = SMOTE(sampling_strategy={1:30, 2:35},k_neighbors=3,random_state=0)
                X_res, y_res = resampler.fit_resample(X_train, y_train)
            else:
                X_res = X_train
                y_res = y_train
            
            optimised_rf = []
            bagging_t = partial(bagging, X_res, y_res)

            for t in range(bagging_trials_num):
                optimised_rf.append(bagging_t(t))
        
        # 学習済みモデルの保存
        joblib.dump(optimised_rf, "nn.pkl", compress=True)

        filename = 'nn.sav'
        pickle.dump(optimised_rf, open(filename, 'wb'))

                
        y_preds = []
        for m in optimised_rf:
            y_preds.append(m.predict(X_test))
        #print(y_preds)
        y_preds_bagging=[]
        y_preds_bagging, count_2=stats.mode(y_preds, axis=0)
        
        y_pred_sum.extend(y_preds_bagging)
        y_test_sum.extend(y_test)

        # 予測精度

    y_pred_sum=list(itertools.chain.from_iterable(y_pred_sum))
    print(classification_report(y_test_sum, y_pred_sum))
        

feature_all=[]
cm_all=[]
accuracy_all=[]
precision_all=[]
recall_all=[]
f1_all=[]
identification=[]

feature=[]
cm=[]
accuracy=[]
precision=[]
recall=[]
f1=[]
classification=[]
#全ての特徴量　oputuna up_sample
#def RandomForestClassifier_optuna(mode = ノーマル、アップサンプリング，ダウンサンプリング, n_trials_num = 整数,feature_drop=特徴量):

up_mode= 1
down_mode = 1
bagging_trials_num = 10
n_trials_num = 0
evaluation = 'f1_macro'#accuracy, recall_weighted, f1_weighted

feature_drop=[ #'RRiM','RRiS','LP_all',
                                   #'LP_Bathing', 'LP_Cooking', 'LP_Eating', 'LP_Goingout', 'LP_Sleeping', 'LP_Other',
                                    #'SA_Bathing', 'SA_Cooking', 'SA_Eating', 'SA_Goingout', 'SA_Sleeping', 'SA_Other',
                                    #'LP_Bathing_2', 'LP_Cooking_2', 'LP_Eating_2', 'LP_Goingout_2', 'LP_Sleeping_2', 'LP_Other_2',#_2は起床後から4h
                                    #'SA_Bathing_2', 'SA_Cooking_2', 'SA_Eating_2', 'SA_Goingout_2', 'SA_Sleeping_2', 'SA_Other_2',
                                   
#                                 'Bathing_LPSAprefer_LP', 'Cooking_LPSAprefer_LP', 'Eating_LPSAprefer_LP', 'Goingout_LPSAprefer_LP','Sleeping_LPSAprefer_LP', 'Other_LPSAprefer_LP', 
#                                 'Bathing_LPSAprefer_SA', 'Cooking_LPSAprefer_SA', 'Eating_LPSAprefer_SA', 'Goingout_LPSAprefer_SA','Sleeping_LPSAprefer_SA', 'Other_LPSAprefer_SA',
#                                 'Bathing_LPSAprefer_LP_SA', 'Cooking_LPSAprefer_P_SA', 'Eating_LPSAprefer_LP_SA', 'Goingout_LPSAprefer_LP_SA','Sleeping_LPSAprefer_LP_SA', 'Other_LPSAprefer_LP_SA',
                                   # 'M1','M2','M3' , 
]                                  

X,y1,y2,y3,cols,groups,logo = data_get_drop_morning(feature_drop)
RandomForestClassifier_optuna(up_mode, down_mode, bagging_trials_num, n_trials_num, evaluation)