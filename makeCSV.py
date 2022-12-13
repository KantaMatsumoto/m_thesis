from flask import Flask, render_template, request, flash, session, escape, redirect, url_for
from wtforms import (
    Form, BooleanField, IntegerField, PasswordField, StringField,
    SubmitField, TextAreaField, validators, ValidationError, FloatField)
import numpy as np
import joblib
from scipy import stats
import pandas as pd
import numpy as np
from statistics import mean
import datetime

featureTime = ['Date','Going-out','Cooking', 'Eating', 'Bathing',  'Other','Sleeping','pred', 'timeLeft','sleepStart', 'sleepEnd',
    'goingoutLengthPre','cookingLengthPre','eatingLengthPre','bathingLengthPre','otherLengthPre']
data = [[0,0,0,0,0,0,0,0,0, str(0)+':'+ str(0), str(0)+':'+ str(0),
    0,0,0,0,0]]
for id in range(13):
    df = pd.DataFrame(columns=featureTime)
    f = open('result/' + str(id) + "_1回目" + '.csv', 'w')
    df.to_csv(f,index=False)
    f.close()
    df = pd.DataFrame(columns=featureTime)
    f = open('result/' + str(id) + "_2回目" + '.csv', 'w')
    df.to_csv(f,index=False)
    f.close()

# featureTime = ['Date','Going-out','Cooking', 'Eating', 'Bathing',  'Other','Sleeping','pred', 'timeLeft','sleepStart', 'sleepEnd',
# 'goingoutLengthPre','cookingLengthPre','eatingLengthPre','bathingLengthPre','otherLengthPre']    
# data = [[1,2,3,4,5,6,88,1,2, str(4)+':'+ str(5), str(6)+':'+ str(6),
#     6,1,16,23,46],[1,2,3,4,5,6,88,1,2, str(4)+':'+ str(5), str(6)+':'+ str(6),
#     6,1,16,23,46]]
# df = pd.DataFrame(data,columns=featureTime)
# print(type(df))
# print(df)
# csv = pd.read_csv('result/' + str(0) + "_1回目" + '.csv')
# print(type(csv))
# print(csv)
# A = pd.concat([csv, df], axis=0)

# print('結果')
# print(A)