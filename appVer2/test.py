from flask import Flask, render_template, request, flash
from wtforms import Form, FloatField, SubmitField, validators, ValidationError
import numpy as np
import joblib
from scipy import stats
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
import datetime
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

import optuna
from sklearn.model_selection import cross_val_score
from functools import partial

import collections

features = ['RRiM','RRiS','LP_all',
'LP_Bathing', 'LP_Cooking', 'LP_Eating', 'LP_Goingout', 'LP_Sleeping', 'LP_Other',
'SA_Bathing', 'SA_Cooking', 'SA_Eating', 'SA_Goingout', 'SA_Sleeping', 'SA_Other',
#'LP_Bathing_2', 'LP_Cooking_2', 'LP_Eating_2', 'LP_Goingout_2', 'LP_Sleeping_2', 'LP_Other_2',#_2は起床後から4h
#'SA_Bathing_2', 'SA_Cooking_2', 'SA_Eating_2', 'SA_Goingout_2', 'SA_Sleeping_2', 'SA_Other_2',
'Bathing_LPSAprefer_LP', 'Cooking_LPSAprefer_LP', 'Eating_LPSAprefer_LP', 'Goingout_LPSAprefer_LP','Sleeping_LPSAprefer_LP', 'Other_LPSAprefer_LP', 
'Bathing_LPSAprefer_SA', 'Cooking_LPSAprefer_SA', 'Eating_LPSAprefer_SA', 'Goingout_LPSAprefer_SA','Sleeping_LPSAprefer_SA', 'Other_LPSAprefer_SA',
#'Bathing_LPSAprefer_LP_SA', 'Cooking_LPSAprefer_P_SA', 'Eating_LPSAprefer_LP_SA', 'Goingout_LPSAprefer_LP_SA','Sleeping_LPSAprefer_LP_SA', 'Other_LPSAprefer_LP_SA',
]                


def saveFile(xTest, pred, turn):#fileにdataframeを保存する用
    ID = 1
    SEX = 1    
    dtNow = datetime.datetime.now()
    xTestNP=[]
    xTestNP = makeNpArray(xTest)
    print('xTestNP',list(itertools.chain.from_iterable(xTestNP)), flush=True)

    #df = pd.DataFrame(xTestNP, index=[], columns=features)
   # df.to_csv('result /' + 'ID' + str(ID) + 'SEX' + str(SEX) + 'TURN' + str(turn) + 'DATE' + str(dtNow.year) + str(dtNow.month)  +  str(dtNow.day) + '.csv')

def makePrefer():#mixed_indicatorの設定（定数）
    lpPrefer = pd.DataFrame(
    {
        "Name": [
            "ID1SEX1", "ID2SEX1","ID3SEX1", "ID5SEX1",
            "ID1SEX2","ID2SEX2", "ID3SEX2","ID4SEX1","ID5SEX2"
            ],
        'Bathing_prefer':[0.7,1.14,2.04,0.12,1,1,4,1,3], 
        'Cooking_prefer':[0.01,1.86,0.01,1.3,0.67,1.22,0.54,1.42,1.15], 
        'Eating_prefer':[1.09,0.94,0.88,1.09,1.61,1.16,0.64,0.83,0.76],
        'Goingout_prefer':[1.05,0.96,1.19,0.8,0.8,1.13,1.53,0.99,0.54], 
        'Sleeping_prefer':[0.97,0.7,1.33,1,0.95,1.02,1.02,1.09,0.92], 
        'Other_prefer':[1.03,0.90,0.89,1.17,1.13,0.99,1.01,0.98,0.89],
        }
    )
    lpPrefer = lpPrefer.set_index('Name')
    lpPreferCol=lpPrefer.columns

    saPrefer = pd.DataFrame(
        {
            "Name": [
                "ID1SEX1", "ID2SEX1","ID3SEX1", "ID5SEX1",
                "ID1SEX2","ID2SEX2", "ID3SEX2","ID4SEX1","ID5SEX2"
                ],
            'Bathing_prefer':[0.25,0.60,2.33,0.83,0.4,0.37,2.49,0.53,1.2], 
            'Cooking_prefer':[0.01,0.01,3.37,0.61,1.84,0.53,1.34,0.37,0.92], 
            'Eating_prefer':[0.67,0.72,1.65,0.95,0.66,0.73,1.43,0.94,1.23],
            'Goingout_prefer':[0.93,1.12,0.84,1.11,0.71,1.10,0.86,1.26,1.07], 
            'Sleeping_prefer':[1.52,0.61,0.83,1.04,1.11,1.31,0.78,1.31,0.48], 
            'Other_prefer':[1.01,1.27,0.77,0.95,1.14,0.99,0.92,0.86,1.11],
            }
        )
    saPrefer = saPrefer.set_index('Name')
    saPreferCol=  saPrefer.columns
    lpsaPrefer = lpPrefer.copy()
    lpsaPrefer = lpPrefer * saPrefer
    lpsaPreferCol = lpsaPrefer.columns
    return lpsaPrefer


def defineLP():#LPの予測の際に対数を代入
        #'LP_all','LP_Bathing', 'LP_Cooking', 'LP_Eating', 'LP_Goingout', 'LP_Sleeping', 'LP_Other'
        lp = [1931.181986,1528.488601,1822.615563,1974.988845,1294.652644,2672.004235]
        lpAll = 1900
        
        return lp, lpAll

def deleteFeatures(features):#不必要なものがあれば削除するための関数
    featureDrop = [ #'RRiM','RRiS','LP_all',
                                    #'LP_Bathing', 'LP_Cooking', 'LP_Eating', 'LP_Goingout', 'LP_Sleeping', 'LP_Other',
                                    #'SA_Bathing', 'SA_Cooking', 'SA_Eating', 'SA_Goingout', 'SA_Sleeping', 'SA_Other',
                                    #'LP_Bathing_2', 'LP_Cooking_2', 'LP_Eating_2', 'LP_Goingout_2', 'LP_Sleeping_2', 'LP_Other_2',#_2は起床後から4h
                                    #'SA_Bathing_2', 'SA_Cooking_2', 'SA_Eating_2', 'SA_Goingout_2', 'SA_Sleeping_2', 'SA_Other_2',
#                                 'Bathing_LPSAprefer_LP', 'Cooking_LPSAprefer_LP', 'Eating_LPSAprefer_LP', 'Goingout_LPSAprefer_LP','Sleeping_LPSAprefer_LP', 'Other_LPSAprefer_LP', 
#                                 'Bathing_LPSAprefer_SA', 'Cooking_LPSAprefer_SA', 'Eating_LPSAprefer_SA', 'Goingout_LPSAprefer_SA','Sleeping_LPSAprefer_SA', 'Other_LPSAprefer_SA',
#                                 'Bathing_LPSAprefer_LP_SA', 'Cooking_LPSAprefer_P_SA', 'Eating_LPSAprefer_LP_SA', 'Goingout_LPSAprefer_LP_SA','Sleeping_LPSAprefer_LP_SA', 'Other_LPSAprefer_LP_SA',
]                

def makeFeatures(sa):#予測用の値を生成
    lpTemplate, lpAll = defineLP()
    lpsaPrefer = makePrefer()
    rriM, rriS = 1000,0.6
    lpsaPreferLP = (lpsaPrefer.loc['ID3SEX2'].values * lpTemplate)
    lpsaPreferSA = sa/(lpsaPrefer.loc['ID3SEX2'].values)
    featuresValue = [rriM,rriS,lpAll, lpTemplate, sa, lpsaPreferLP, lpsaPreferSA ]
    #featuresDay = np.concatenate([features,featuresValue])
    return featuresValue

def add_el(ar1:list, el1):#np.arrayを１列にするための関数（reshapeが使えなかったため作成）
    ar1.append(el1)
    return ar1
def makeNpArray(parameters):
    params = []
    for el in parameters:
        try:
            for e in el:
                params = add_el(params, e)
        except:
            params = add_el(params, el)
    params = np.array(params)
    params = params.reshape(1, -1)
    return params

# 学習済みモデルを読み込み利用します
def predictStress(parameters):# ニューラルネットワークのモデルを読み込み
    model = joblib.load('./nn.pkl')
    params = makeNpArray(parameters)
    yPreds = []
    for m in model:
        yPreds.append(m.predict(params))
    yPredsBagging=[]
    yPredsBagging, count_2=stats.mode(yPreds, axis=0)
    return yPredsBagging


# ラベルから体調の状態を取得します
def stressStatus(label):
    #print(label)
    if label == 1:
        return "体調は悪くなるでしょう"
    elif label == 2: 
        return "体調は維持されるでしょう"
    elif label == 3: 
        return "体調は良くなるでしょう"
    else: 
        return "Error"


app = Flask(__name__)
app.config.from_object(__name__)
app.config['SECRET_KEY'] = 'zJe09C5c3tMf5FnNL09C5d6SAzZoY'

# 公式サイト
# http://wtforms.simplecodes.com/docs/0.6/fields.html
# Flaskとwtformsを使い、index.html側で表示させるフォームを構築します。

goingoutLength = 4               
cookingLength  = 4
eatingLength = 4            
bathingLength  = 4     
sleepingLength = 4 
otherLength  = 4 

timeTocount = 60*4
x = [goingoutLength *timeTocount, cookingLength *timeTocount, eatingLength *timeTocount, bathingLength *timeTocount,sleepingLength *timeTocount,otherLength*timeTocount]

X_test= makeFeatures(x)

pred = predictStress(X_test)
Status = stressStatus(int(pred))

saveFile(X_test,pred,1)
#print(Status)
