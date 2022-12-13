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


app = Flask(__name__, static_folder='./templates/') # Flask動かす時のおまじない。
#app.config.from_object(__name__)
app.config['SECRET_KEY'] = 'zJe09C5c3tMf5FnNL09C5d6SAzZoY'

features = ['RRiM','RRiS','LP_all',
'LP_Bathing', 'LP_Cooking', 'LP_Eating', 'LP_Goingout', 'LP_Sleeping', 'LP_Other',
'SA_Bathing', 'SA_Cooking', 'SA_Eating', 'SA_Goingout', 'SA_Sleeping', 'SA_Other',
#'LP_Bathing_2', 'LP_Cooking_2', 'LP_Eating_2', 'LP_Goingout_2', 'LP_Sleeping_2', 'LP_Other_2',#_2は起床後から4h
#'SA_Bathing_2', 'SA_Cooking_2', 'SA_Eating_2', 'SA_Goingout_2', 'SA_Sleeping_2', 'SA_Other_2',
'Bathing_LPSAprefer_LP', 'Cooking_LPSAprefer_LP', 'Eating_LPSAprefer_LP', 'Goingout_LPSAprefer_LP','Sleeping_LPSAprefer_LP', 'Other_LPSAprefer_LP', 
'Bathing_LPSAprefer_SA', 'Cooking_LPSAprefer_SA', 'Eating_LPSAprefer_SA', 'Goingout_LPSAprefer_SA','Sleeping_LPSAprefer_SA', 'Other_LPSAprefer_SA',
#'Bathing_LPSAprefer_LP_SA', 'Cooking_LPSAprefer_P_SA', 'Eating_LPSAprefer_LP_SA', 'Goingout_LPSAprefer_LP_SA','Sleeping_LPSAprefer_LP_SA', 'Other_LPSAprefer_LP_SA',
]      


def saveFile(id, day, goingoutLength,cookingLength,eatingLength,bathingLength,otherLength, pred, 
        sleepHourStart, sleepMinStart, sleepHourEnd, sleepMinEnd,sleepingLength, timeLeft,
        goingoutLengthPre,cookingLengthPre,eatingLengthPre,bathingLengthPre,otherLengthPre,fileState):#fileにdataframeを保存する用
    dtNow = datetime.datetime.now()
    try:
        for p in pred:
            for p2 in p:
                pred=p2
    except:
        pass
    featureTime = ['Date','Going-out','Cooking', 'Eating', 'Bathing',  'Other','Sleeping','pred', 'timeLeft','sleepStart', 'sleepEnd',
    'goingoutLengthPre','cookingLengthPre','eatingLengthPre','bathingLengthPre','otherLengthPre']    
    data = [[day,goingoutLength,cookingLength,eatingLength,bathingLength,otherLength,sleepingLength,pred,timeLeft, str(sleepHourStart)+':'+ str(sleepMinStart), str(sleepHourEnd)+':'+ str(sleepMinEnd),
        goingoutLengthPre,cookingLengthPre,eatingLengthPre,bathingLengthPre,otherLengthPre]]
    df = pd.DataFrame(data,columns=featureTime)
    if fileState ==1:
        csv = pd.read_csv('result/' + str(id) + "_1回目" + '.csv')
        A=pd.concat([csv, df], axis=0)
        A.to_csv('result/' + str(id) + "_1回目"+ '.csv',index=False)
    else:
        csv = pd.read_csv('result/' + str(id) + "_2回目" + '.csv')
        A=pd.concat([csv, df], axis=0)
        A.to_csv('result/' + str(id) + "_2回目" + '.csv',index=False)

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

def add_el(ar1, el1):#np.arrayを１列にするための関数（reshapeが使えなかったため作成）
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
        return "体調が悪くなるでしょう"
    elif label == 2: 
        return "体調が維持されるでしょう"
    elif label == 3: 
        return "体調が良くなるでしょう"
    else: 
        return "Error"

def getForm(sleepingLength, goingoutLength, cookingLength, eatingLength, bathingLength, otherLength,numberTimes):
    timeToCount = 60*4
    sleepingLength = float(sleepingLength)
    Time = [sleepingLength, goingoutLength, cookingLength, eatingLength, bathingLength, otherLength]
    timeCount = [goingoutLength * timeToCount, cookingLength * timeToCount, eatingLength * timeToCount, bathingLength * timeToCount, sleepingLength * timeToCount, otherLength * timeToCount]#特徴量が15秒に1回周期のため。
    xTest= makeFeatures(timeCount)
    pred = predictStress(xTest)
    if pred==1:
        if numberTimes == 2:
            pred = 2
        if numberTimes >= 3:
            pred = 3
    elif pred==2:
        if numberTimes >= 2:
            pred = 3
    else:
        pass
    if sleepingLength < 6:
        pred = 1
    status = stressStatus(int(pred))
    return Time,xTest,pred,status

def getSleepForm():
    id = int(request.form["id"])        
    sleepHourStart  = int(request.form["sleepHourStart"])
    sleepMinStart = int(request.form["sleepMinStart"])            
    sleepHourEnd  = int(request.form["sleepHourEnd"])            
    sleepMinEnd = int(request.form["sleepMinEnd"])
    goingoutHour = float(request.form["goingoutHour"])                       
    cookingHour  = float(request.form["cookingHour"])
    eatingHour = float(request.form["eatingHour"])            
    bathingHour  = float(request.form["bathingHour"])            
    otherHour  = float(request.form["otherHour"])
    goingoutMin = float(request.form["goingoutMin"])                       
    cookingMin  = float(request.form["cookingMin"])
    eatingMin = float(request.form["eatingMin"])            
    bathingMin  = float(request.form["bathingMin"])            
    otherMin  = float(request.form["otherMin"])
    goingoutLength = goingoutHour+goingoutMin/60
    cookingLength = cookingHour+cookingMin/60
    eatingLength = eatingHour+eatingMin/60
    bathingLength = bathingHour+bathingMin/60
    otherLength = otherHour+otherMin/60    
    return id,sleepHourStart,sleepMinStart,sleepHourEnd,sleepMinEnd,goingoutLength,cookingLength,eatingLength,bathingLength,otherLength

class sleepIdForm(Form):
    id = IntegerField("IDを入力してください",
                    [validators.InputRequired("この項目は入力必須です"),
                    validators.NumberRange(min=0, max=50)])
    sleepHourStart  = IntegerField("入眠時間",
                    [validators.InputRequired("この項目は入力必須です"),
                    validators.NumberRange(min=0, max=24)])
    sleepMinStart = IntegerField("分",
                    [validators.InputRequired("この項目は入力必須です"),
                    validators.NumberRange(min=0, max=59)])
    sleepHourEnd  = IntegerField("起床時間",
                    [validators.InputRequired("この項目は入力必須です"),
                    validators.NumberRange(min=0, max=24)])
    sleepMinEnd = IntegerField("分",
                    [validators.InputRequired("この項目は入力必須です"),
                    validators.NumberRange(min=0, max=59)])
    goingoutHour = IntegerField("外出時間",
                    [validators.InputRequired("この項目は入力必須です"),
                    validators.NumberRange(min=0, max=12)])
    goingoutMin = IntegerField("分",
                    [validators.InputRequired("この項目は入力必須です"),
                    validators.NumberRange(min=0, max=59)])
    cookingHour  = IntegerField("料理時間",
                    [validators.InputRequired("この項目は入力必須です"),
                    validators.NumberRange(min=0, max=10)])
    cookingMin = IntegerField("分",
                    [validators.InputRequired("この項目は入力必須です"),
                    validators.NumberRange(min=0, max=59)])
    eatingHour = IntegerField("食事時間",
                    [validators.InputRequired("この項目は入力必須です"),
                    validators.NumberRange(min=0, max=10)])
    eatingMin = IntegerField("分",
                    [validators.InputRequired("この項目は入力必須です"),
                    validators.NumberRange(min=0, max=59)])
    bathingHour  = IntegerField("入浴時間",
                    [validators.InputRequired("この項目は入力必須です"),
                    validators.NumberRange(min=0, max=10)])
    bathingMin = IntegerField("分",
                    [validators.InputRequired("この項目は入力必須です"),
                    validators.NumberRange(min=0, max=59)])
    otherHour  = IntegerField("その他時間",
                    [validators.InputRequired("この項目は入力必須です"),
                    validators.NumberRange(min=0, max=12)])
    otherMin = IntegerField("分",
                    [validators.InputRequired("この項目は入力必須です"),
                    validators.NumberRange(min=0, max=59)])
    accept = BooleanField("内容確認：",[validators.InputRequired("この項目は入力必須です")])
    # html側で表示するsubmitボタンの表示
    submitID = SubmitField("IDと睡眠時間を入力完了")

class activityFormIndex(Form):
    goingoutHour = IntegerField("外出時間",
                    [validators.InputRequired("この項目は入力必須です"),
                    validators.NumberRange(min=0, max=12)])
    goingoutMin = IntegerField("分",
                    [validators.InputRequired("この項目は入力必須です"),
                    validators.NumberRange(min=0, max=59)])
    cookingHour  = IntegerField("料理時間",
                    [validators.InputRequired("この項目は入力必須です"),
                    validators.NumberRange(min=0, max=10)])
    cookingMin = IntegerField("分",
                    [validators.InputRequired("この項目は入力必須です"),
                    validators.NumberRange(min=0, max=59)])
    eatingHour = IntegerField("食事時間",
                    [validators.InputRequired("この項目は入力必須です"),
                    validators.NumberRange(min=0, max=10)])
    eatingMin = IntegerField("分",
                    [validators.InputRequired("この項目は入力必須です"),
                    validators.NumberRange(min=0, max=59)])
    bathingHour  = IntegerField("入浴時間",
                    [validators.InputRequired("この項目は入力必須です"),
                    validators.NumberRange(min=0, max=10)])
    bathingMin = IntegerField("分",
                    [validators.InputRequired("この項目は入力必須です"),
                    validators.NumberRange(min=0, max=59)])
    otherHour  = IntegerField("その他時間",
                    [validators.InputRequired("この項目は入力必須です"),
                    validators.NumberRange(min=0, max=12)])
    otherMin = IntegerField("分",
                    [validators.InputRequired("この項目は入力必須です"),
                    validators.NumberRange(min=0, max=59)])
    # html側で表示するsubmitボタンの表示
    submit = SubmitField("送信")

class activityFormIndex2(Form):
    numberTimes = IntegerField("回数",
                    [validators.InputRequired("この項目は入力必須です"),
                    validators.NumberRange(min=0, max=50)])
    goingoutHour = IntegerField("外出時間",
                    [validators.InputRequired("この項目は入力必須です"),
                    validators.NumberRange(min=0, max=12)])
    goingoutMin = IntegerField("分",
                    [validators.InputRequired("この項目は入力必須です"),
                    validators.NumberRange(min=0, max=59)])
    cookingHour  = IntegerField("料理時間",
                    [validators.InputRequired("この項目は入力必須です"),
                    validators.NumberRange(min=0, max=10)])
    cookingMin = IntegerField("分",
                    [validators.InputRequired("この項目は入力必須です"),
                    validators.NumberRange(min=0, max=59)])
    eatingHour = IntegerField("食事時間",
                    [validators.InputRequired("この項目は入力必須です"),
                    validators.NumberRange(min=0, max=10)])
    eatingMin = IntegerField("分",
                    [validators.InputRequired("この項目は入力必須です"),
                    validators.NumberRange(min=0, max=59)])
    bathingHour  = IntegerField("入浴時間",
                    [validators.InputRequired("この項目は入力必須です"),
                    validators.NumberRange(min=0, max=10)])
    bathingMin = IntegerField("分",
                    [validators.InputRequired("この項目は入力必須です"),
                    validators.NumberRange(min=0, max=59)])
    otherHour  = IntegerField("その他時間",
                    [validators.InputRequired("この項目は入力必須です"),
                    validators.NumberRange(min=0, max=12)])
    otherMin = IntegerField("分",
                    [validators.InputRequired("この項目は入力必須です"),
                    validators.NumberRange(min=0, max=59)])
    sleepHourStart  = IntegerField("入眠時間",
                    [validators.Optional(),
                    validators.NumberRange(min=0, max=24)])
    sleepMinStart = IntegerField("分",
                    [validators.Optional(),
                    validators.NumberRange(min=0, max=59)])
    sleepHourEnd  = IntegerField("起床時間",
                    [validators.Optional(),
                    validators.NumberRange(min=0, max=24)])
    sleepMinEnd = IntegerField("分",
                    [validators.Optional(),
                    validators.NumberRange(min=0, max=59)])
    # html側で表示するsubmitボタンの表示
    submit = SubmitField("送信")

@app.route('/', methods = ['GET', 'POST'])# どのページで実行する関数か設定
def root():
    idForm = sleepIdForm(request.form)
    if request.method == 'POST':
        if idForm.validate() == False:
            flash("全て入力する必要があります。")
            return render_template('id.html', idForm=idForm)
        else:
            id,sleepHourStart,sleepMinStart,sleepHourEnd,sleepMinEnd,goingoutLengthPre,cookingLengthPre,eatingLengthPre,bathingLengthPre,otherLengthPre = getSleepForm()
            dtNow = datetime.datetime.now() + datetime.timedelta(hours=9)
            day = dtNow.strftime('%Y年%m月%d日 %H:%M:%S')
            if sleepHourStart >= 18:
                sleepingLength = (24 - (sleepHourStart + sleepMinStart/60)) + (sleepHourEnd + sleepMinEnd/60)
                #18時睡眠の場合、今日の残り睡眠時間（24:00-18:00）＋明日の睡眠時間
                timeLeft=(sleepHourStart + sleepMinStart/60)-(int(dtNow.hour)+int(dtNow.hour)/60)
            else:
                sleepingLength = (sleepHourEnd + sleepMinEnd/60) - (sleepHourStart + sleepMinStart/60)
                #2時睡眠の場合、今日の睡眠時間-明日の起床時間
                timeLeft=(24-(int(dtNow.hour)+int(dtNow.hour)/60))+(sleepHourStart + sleepMinStart/60)
                #今日の残り時間（24:00-今の時間）+次の日の睡眠開始時間
            return redirect(url_for('firstIndex',id=id,day=day, sleepHourStart=sleepHourStart, sleepMinStart=sleepMinStart,
                sleepHourEnd=sleepHourEnd,sleepMinEnd=sleepMinEnd,sleepingLength=sleepingLength,timeLeft=timeLeft, goingoutLengthPre=goingoutLengthPre,
                cookingLengthPre=cookingLengthPre,eatingLengthPre=eatingLengthPre,bathingLengthPre=bathingLengthPre,otherLengthPre=otherLengthPre,dummy=1))
    elif request.method == 'GET':
        return render_template('id.html', idForm=idForm)

@app.route('/firstIndex/<id>/<day>/<sleepHourStart>/<sleepMinStart>/<sleepHourEnd>/<sleepMinEnd>/<timeLeft>/<sleepingLength>/<goingoutLengthPre>/<cookingLengthPre>/<eatingLengthPre>/<bathingLengthPre>/<otherLengthPre>/<dummy>', methods = ['GET', 'POST'])# どのページで実行する関数か設定
def firstIndex(id,day,sleepHourStart, sleepMinStart,sleepHourEnd,sleepMinEnd,sleepingLength,timeLeft,goingoutLengthPre,cookingLengthPre,eatingLengthPre,bathingLengthPre,otherLengthPre,dummy):
    form = activityFormIndex(request.form)
    if request.method == 'POST':
        if form.validate() == False:
            flash("全て入力する必要があります。")
            return render_template('firstIndex.html', form=form, id=id,day=day)
        else:
            goingoutHour = float(request.form["goingoutHour"])                       
            cookingHour  = float(request.form["cookingHour"])
            eatingHour = float(request.form["eatingHour"])            
            bathingHour  = float(request.form["bathingHour"])            
            otherHour  = float(request.form["otherHour"])
            goingoutMin = float(request.form["goingoutMin"])                       
            cookingMin  = float(request.form["cookingMin"])
            eatingMin = float(request.form["eatingMin"])            
            bathingMin  = float(request.form["bathingMin"])            
            otherMin  = float(request.form["otherMin"])
            goingoutLength = goingoutHour+goingoutMin/60
            cookingLength = cookingHour+cookingMin/60
            eatingLength = eatingHour+eatingMin/60
            bathingLength = bathingHour+bathingMin/60
            otherLength = otherHour+otherMin/60
            Time,xTest,pred,sendStatus = getForm(sleepingLength,goingoutLength,cookingLength,eatingLength,bathingLength,otherLength,numberTimes=0)
            saveFile(id, day, goingoutLength,cookingLength,eatingLength,bathingLength,otherLength, pred, 
                    sleepHourStart, sleepMinStart, sleepHourEnd, sleepMinEnd,sleepingLength, timeLeft,
                    goingoutLengthPre,cookingLengthPre,eatingLengthPre,bathingLengthPre,otherLengthPre,fileState=1)
        return redirect(url_for('result', goingoutLength=goingoutLength,cookingLength=cookingLength,eatingLength=eatingLength,
                bathingLength=bathingLength,otherLength=otherLength, pred=int(pred), sendStatus=sendStatus ,id=id,day=day, 
                sleepHourStart=sleepHourStart, sleepMinStart=sleepMinStart, sleepHourEnd=sleepHourEnd, sleepMinEnd=sleepMinEnd,sleepingLength=sleepingLength, timeLeft=timeLeft,
                goingoutLengthPre=goingoutLengthPre,cookingLengthPre=cookingLengthPre,eatingLengthPre=eatingLengthPre,bathingLengthPre=bathingLengthPre,otherLengthPre=otherLengthPre))
    elif request.method == 'GET':
        return render_template('firstIndex.html', form=form, id=id,day=day, sleepHourStart=sleepHourStart, 
                sleepMinStart=sleepMinStart, sleepHourEnd=sleepHourEnd, sleepMinEnd=sleepMinEnd,sleepingLength=sleepingLength,timeLeft=timeLeft,
                goingoutLengthPre=goingoutLengthPre,cookingLengthPre=cookingLengthPre,eatingLengthPre=eatingLengthPre,bathingLengthPre=bathingLengthPre,otherLengthPre=otherLengthPre)#id=id,sleepHourStart=sleepHourStart,sleepMinStart=sleepMinStart,sleepHourEnd=sleepHourEnd,sleepMinEnd=sleepMinEnd,)

@app.route('/secondIndex/<id>/<day>/<sleepHourStart>/<sleepMinStart>/<sleepHourEnd>/<sleepMinEnd>/<sleepingLength>/<timeLeft>/<goingoutLengthPre>/<cookingLengthPre>/<eatingLengthPre>/<bathingLengthPre>/<otherLengthPre>/<dummy>', methods = ['GET', 'POST'])
def secondIndex(id,day,sleepHourStart, sleepMinStart,sleepHourEnd,sleepMinEnd,sleepingLength,timeLeft,goingoutLengthPre,cookingLengthPre,eatingLengthPre,bathingLengthPre,otherLengthPre,dummy):    
    form = activityFormIndex2(request.form)
    if request.method == 'POST':
        if form.validate() == False:
            flash("全て入力する必要があります。")
            return render_template('secondIndex.html', form=form, id=id,day=day)
        else:
            goingoutHour = float(request.form["goingoutHour"])                       
            cookingHour  = float(request.form["cookingHour"])
            eatingHour = float(request.form["eatingHour"])            
            bathingHour  = float(request.form["bathingHour"])            
            otherHour  = float(request.form["otherHour"])
            goingoutMin = float(request.form["goingoutMin"])                       
            cookingMin  = float(request.form["cookingMin"])
            eatingMin = float(request.form["eatingMin"])            
            bathingMin  = float(request.form["bathingMin"])            
            otherMin  = float(request.form["otherMin"])

            goingoutLength = goingoutHour+goingoutMin/60
            cookingLength = cookingHour+cookingMin/60
            eatingLength = eatingHour+eatingMin/60
            bathingLength = bathingHour+bathingMin/60
            otherLength = otherHour+otherMin/60


            sleepHourStart  = int(request.form["sleepHourStart"])
            sleepMinStart = int(request.form["sleepMinStart"])            
            sleepHourEnd  = int(request.form["sleepHourEnd"])            
            sleepMinEnd = int(request.form["sleepMinEnd"])

            if sleepHourStart >= 18:
                sleepingLength = (24 - (sleepHourStart + sleepMinStart/60)) + (sleepHourEnd + sleepMinEnd/60)
            else:
                sleepingLength = (sleepHourEnd + sleepMinEnd/60) - (sleepHourStart + sleepMinStart/60)
                #2時睡眠の場合、今日の睡眠時間-明日の起床時間
            numberTimes=int(request.form["numberTimes"])    
            Time,xTest,pred,sendStatus = getForm(sleepingLength,goingoutLength,cookingLength,eatingLength,bathingLength,otherLength,numberTimes)
            
            return redirect(url_for('result', goingoutLength=goingoutLength,cookingLength=cookingLength,eatingLength=eatingLength,
            bathingLength=bathingLength,otherLength=otherLength, pred=int(pred), sendStatus=sendStatus ,id=id,day=day, 
            sleepHourStart=sleepHourStart, sleepMinStart=sleepMinStart, sleepHourEnd=sleepHourEnd, sleepMinEnd=sleepMinEnd,sleepingLength=sleepingLength, timeLeft=timeLeft,
            goingoutLengthPre=goingoutLengthPre,cookingLengthPre=cookingLengthPre,eatingLengthPre=eatingLengthPre,bathingLengthPre=bathingLengthPre,otherLengthPre=otherLengthPre))
    elif request.method == 'GET':
        return render_template('secondIndex.html', form=form, id=id,day=day, sleepHourStart=sleepHourStart, 
                sleepMinStart=sleepMinStart, sleepHourEnd=sleepHourEnd, sleepMinEnd=sleepMinEnd,sleepingLength=sleepingLength,timeLeft=timeLeft, 
                goingoutLengthPre=goingoutLengthPre,cookingLengthPre=cookingLengthPre,eatingLengthPre=eatingLengthPre,bathingLengthPre=bathingLengthPre,otherLengthPre=otherLengthPre)#id=id,sleepHourStart=sleepHourStart,sleepMinStart=sleepMinStart,sleepHourEnd=sleepHourEnd,sleepMinEnd=sleepMinEnd,)

@app.route('/result/<goingoutLength>/<cookingLength>/<eatingLength>/<bathingLength>/<otherLength>/<pred>/<sendStatus>/<id>/<day>/<sleepHourStart>/<sleepMinStart>/<sleepHourEnd>/<sleepMinEnd>/<sleepingLength>/<timeLeft>/<goingoutLengthPre>/<cookingLengthPre>/<eatingLengthPre>/<bathingLengthPre>/<otherLengthPre>', methods = ['GET', 'POST'])
def result(goingoutLength,cookingLength,eatingLength,bathingLength,otherLength, pred, sendStatus, id,day, sleepHourStart, sleepMinStart, sleepHourEnd, sleepMinEnd,sleepingLength, timeLeft,goingoutLengthPre,cookingLengthPre,eatingLengthPre,bathingLengthPre,otherLengthPre):
    pred=int(pred)
    return render_template('result.html',goingoutLength=goingoutLength,cookingLength=cookingLength,eatingLength=eatingLength,
            bathingLength=bathingLength,otherLength=otherLength, pred=int(pred), sendStatus=sendStatus ,id=id,day=day,
            sleepHourStart=sleepHourStart, sleepMinStart=sleepMinStart, sleepHourEnd=sleepHourEnd, sleepMinEnd=sleepMinEnd,sleepingLength=sleepingLength, timeLeft=timeLeft,
            goingoutLengthPre=goingoutLengthPre,cookingLengthPre=cookingLengthPre,eatingLengthPre=eatingLengthPre,bathingLengthPre=bathingLengthPre,otherLengthPre=otherLengthPre)

@app.route('/end/<goingoutLength>/<cookingLength>/<eatingLength>/<bathingLength>/<otherLength>/<pred>/<sendStatus>/<id>/<day>/<sleepHourStart>/<sleepMinStart>/<sleepHourEnd>/<sleepMinEnd>/<sleepingLength>/<timeLeft>/<goingoutLengthPre>/<cookingLengthPre>/<eatingLengthPre>/<bathingLengthPre>/<otherLengthPre>', methods = ['GET', 'POST'])
def end(goingoutLength,cookingLength,eatingLength,bathingLength,otherLength, pred, sendStatus, id,day, sleepHourStart, sleepMinStart, sleepHourEnd, sleepMinEnd,sleepingLength, timeLeft,goingoutLengthPre,cookingLengthPre,eatingLengthPre,bathingLengthPre,otherLengthPre):
    saveFile(id, day, goingoutLength,cookingLength,eatingLength,bathingLength,otherLength, pred, 
            sleepHourStart, sleepMinStart, sleepHourEnd, sleepMinEnd,sleepingLength, timeLeft,
            goingoutLengthPre,cookingLengthPre,eatingLengthPre,bathingLengthPre,otherLengthPre,fileState=2)
    return render_template('end.html',goingoutLength=goingoutLength,cookingLength=cookingLength,eatingLength=eatingLength,
            bathingLength=bathingLength,otherLength=otherLength, pred=int(pred), sendStatus=sendStatus ,id=id,day=day,
            sleepHourStart=sleepHourStart, sleepMinStart=sleepMinStart, sleepHourEnd=sleepHourEnd, sleepMinEnd=sleepMinEnd,sleepingLength=sleepingLength, timeLeft=timeLeft,
            goingoutLengthPre=goingoutLengthPre,cookingLengthPre=cookingLengthPre,eatingLengthPre=eatingLengthPre,bathingLengthPre=bathingLengthPre,otherLengthPre=otherLengthPre)

if __name__ == "__main__": # 実行されたら
    app.run(debug=True, host='0.0.0.0', port=8888, threaded=True)# デバッグモード、localhost:8888 のマルチスレッドで実行