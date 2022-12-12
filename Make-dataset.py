import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import csv
import pprint

from matplotlib import ticker
import seaborn as sns
import math
import datetime
from pandas import DataFrame, Series
from datetime import timedelta
from datetime import datetime as dt
#Behavior estimation
import pickle
import pathlib

# エリア別統計量のDataFrameを作成
# Create DataFrame of area statistics
def area_statistics(df, label):
    df_statistics = pd.DataFrame(index=[], columns=['Bathing', 'Cooking', 'Eating', 'Goingout', 'Sleeping', 'Other'])

    # Living
    df_statistics['Bathing'] = df[df['area'] == 'Bathing'][label].describe()
    # Rehabilitation
    df_statistics['Cooking'] = df[df['area'] == 'Cooking'][label].describe()
    # Recreation
    df_statistics['Eating'] = df[df['area'] == 'Eating'][label].describe()
    # Bed
    df_statistics['Goingout'] = df[df['area'] == 'Goingout'][label].describe()
    # Toilet
    df_statistics['Sleeping'] = df[df['area'] == 'Sleeping'][label].describe()
    # Clerk
    #df_statistics['Clerk'] = df[df['area'] == 'Clerk'][label].describe()
    # Other
    df_statistics['Other'] = df[df['area'] == 'Other'][label].describe()

    return df_statistics


'''
RRIのローレンツプロット（全体）を出力
(DataFrame, ラベル名)
'''
def output_LP(df, display=True):
    df_next = df.copy(deep=True)
    df_next['rri_next'] = df['rri'].shift(-1)

    sns.set()
    sns.set_context("poster")
    sns.set_style('whitegrid')
    # fig = plt.figure(figsize=(14,14))
    # ax = fig.add_subplot(111)

    fig = plt.figure(figsize=(14,14))
    ax = fig.add_subplot(111)
    sns.regplot(x=df_next['rri'], y=df_next['rri_next'] ,ax=ax, fit_reg=False)
    # ax = plt.gca()

    ax.set_xlim([0, 2000])
    ax.set_ylim([0, 2000])
    ax.set_xlabel("RRI(n) [ms]"  )
    ax.set_ylabel("RRI(n+1) [ms]")

    #plt.setp(ax.get_xticklabels(), rotation=45)
    #plt.setp(ax.get_yticklabels(), rotation=0)

    #plt.savefig('SUMMARY' + '/' + Q1 + '/' + 'RRI_LP' + str(po) + '.png', bbox_inches='tight', pad_inches=0)
    #plt.close()

def lorenz_plot_area(rri, next_rri):
    nn = len(rri) #データの長さ
    x_data = rri #x軸となるデータ
    y_data = next_rri #y軸になるデータ（x軸のデータを1点ずらしたデータ）
    rotation_x_data = [0]; #回転後ｘ軸データ
    rotation_y_data = [0]; #回転後y軸データ

    #すべてのデータを-45度回転させる．###########################################
    sita = -1*(math.pi/4)

    for n in range(0,nn-1): #データ数が1点減るので、-1してる
        xx = x_data[n] * math.cos(sita) -1 * y_data[n] * math.sin(sita)
        rotation_x_data = np.hstack([rotation_x_data, xx])
        yy = x_data[n] * math.sin(sita) + y_data[n] * math.cos(sita)
        rotation_y_data = np.hstack([rotation_y_data, yy])
    ########################################################################
        
    #配列の初期値を0にしていたので、そこを捨てる
    rotation_x_data = rotation_x_data[1:nn+1]
    rotation_y_data = rotation_y_data[1:nn+1]

    std_x = np.std(rotation_x_data) #X軸の標準偏差
    std_y = np.std(rotation_y_data) #Y軸の標準偏差

    area = (std_x * std_y * math.pi)/4 #楕円として面積を求める

    return area

# ローレンツプロット面積を計算し，エリアごとでグラフ化する
def output_LP_scale(df):
    df_next = df.copy(deep=True)
    df_next['rri_next'] = df['rri'].shift(-1)
    #print(df_next)
    lorenz_plot_scale = pd.DataFrame(index=[], columns=['area', 'scale'])
    for area_name in df_rri_statistics.columns:
        try:
            #bath（要素）ない時エラーでるexceptで０にする
            df_whs3_one_area = df_next.groupby('area').get_group(area_name)
            df_whs3_one_area = df_whs3_one_area.dropna().reset_index()
            s = lorenz_plot_area(df_whs3_one_area['rri'], df_whs3_one_area['rri_next'])
            #print(s)
            
        except:
            s = None
        lorenz_plot_scale = lorenz_plot_scale.append({'area': area_name, 'scale': s}, ignore_index=True)
        
    #print(area_name)
    #print(lorenz_plot_scale)
    

    #fig = plt.figure(figsize=(16,10))
    #ax = fig.add_subplot(111)
    #sns.barplot(x="area", y="scale", data=lorenz_plot_scale)
    #ax.set_ylim([0, 10000])
    #print(lorenz_plot_scale)
    #plt.savefig('分析データ/' + 'ID' + str(ID) + '/'+ 'SEX' + str(SEX) + '/'  + '/' + 'summary' + '/' + 'output_LP_scale_' + path + '.png', bbox_inches='tight', pad_inches=0)
   # plt.savefig('SUMMARY' + '/' + Q1 + '/'+ 'output_LP_scale_' + str(po) + '.png', bbox_inches='tight', pad_inches=0)
    #plt.close()
    return lorenz_plot_scale

def make_df_question(MONTH,DAY):#集計開始２４h
    RRiBeS = RRiBe.loc[data_w:data_w2]
    RRiBeA = np.array(RRiBeS)[:,:1]
    #print(RRiBeA)
    #RRiBeB = np.argmax(RRiBeA,axis=1).reshape(-1,1)
    #print(RRiBeB)
    RRiBeC = pd.DataFrame(RRiBeA,columns=['rri'])
    RRiBeC.index.name = 'time'
    #print(RRiBeC)

    RRiBeF = np.array(RRiBeS)[:,1:]
    #print(RRiBeF)
    RRiBeG = np.argmax(RRiBeF,axis=1).reshape(-1,1)
    RRiBeH = pd.DataFrame(RRiBeG,columns=['area'])
    RRiBeH.index.name = 'time'
    RRiBeI=RRiBeH.replace(0, 'Bathing')
    RRiBeI=RRiBeI.replace(1, 'Cooking')
    RRiBeI=RRiBeI.replace(2, 'Eating')
    RRiBeI=RRiBeI.replace(3, 'Goingout')
    RRiBeI=RRiBeI.replace(4, 'Sleeping')
    RRiBeI=RRiBeI.replace(5, 'Other')
    #print(RRiBeH)

    df_whs3_area = pd.merge(RRiBeC, RRiBeI,on='time')
    df_whs3_area_rri = df_whs3_area.dropna(how='any')
    return df_whs3_area_rri

def make_df_question_2(MONTH,DAY):#0~4h
    RRiBeS = RRiBe.loc[data_w :data_w+ datetime.timedelta(hours=4)]
    #print(RRiBeS)
    RRiBeA = np.array(RRiBeS)[:,:1]
    #print(RRiBeA)
    #RRiBeB = np.argmax(RRiBeA,axis=1).reshape(-1,1)
    #print(RRiBeB)
    RRiBeC = pd.DataFrame(RRiBeA,columns=['rri'])
    RRiBeC.index.name = 'time'
    #print(RRiBeC)

    RRiBeF = np.array(RRiBeS)[:,1:]
    #print(RRiBeF)
    RRiBeG = np.argmax(RRiBeF,axis=1).reshape(-1,1)
    RRiBeH = pd.DataFrame(RRiBeG,columns=['area'])
    RRiBeH.index.name = 'time'
    RRiBeH=RRiBeH.replace(0, 'Bathing')
    RRiBeH=RRiBeH.replace(1, 'Cooking')
    RRiBeH=RRiBeH.replace(2, 'Eating')
    RRiBeH=RRiBeH.replace(3, 'Goingout')
    RRiBeH=RRiBeH.replace(4, 'Sleeping')
    RRiBeH=RRiBeH.replace(5, 'Other')
    #print(RRiBeH)

    df_whs3_area_2 = pd.merge(RRiBeC, RRiBeH,on='time')
    df_whs3_area_rri_2 = df_whs3_area_2.dropna(how='any')
    return df_whs3_area_rri_2

def makePrefer():
    LPprefer = pd.DataFrame(
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
    LPprefer = LPprefer.set_index('Name')
    LPprefer_col=LPprefer.columns

    SAprefer = pd.DataFrame(
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
    SAprefer = SAprefer.set_index('Name')
    SAprefer_col=SAprefer.columns

    LPSAprefer=LPprefer.copy()
    LPSAprefer=LPprefer*SAprefer
    LPSAprefer_col = LPSAprefer.columns
    return LPSAprefer


#### 3段階Morning
df_rri_statistics = pd.DataFrame(index=[],columns=['Bathing', 'Cooking', 'Eating', 'Goingout', 'Sleeping', 'Other'])
df_whs3_area_rri_question_1 = pd.DataFrame(index=[],columns=['rri','area'])


for ID in range(1,6):
    for SEX in range (1,3):
        if (ID==4) and (SEX==2):
            pass
        else:
            try:
                RRi = pd.read_csv('RRi/ID0' + str(ID) + '/0' + str(SEX) + '_' + str(ID) + '_RRI.csv',
                                  index_col='time', parse_dates=True)
                RRimeanT1 = RRi.resample('15S').mean() #変更01
                RRimeanT = RRimeanT1
                #(RRimeanT1 - RRimeanT1.mean()) / (RRimeanT1.std(ddof=0))
                #RRimeanT = (RRimeanT1 - RRimeanT1.min()) / (RRimeanT1.max()-RRimeanT1.min())

                f = open('行動/yprefeatured_0' + str(SEX) + '_' + str(ID) + '.pickle','rb',)
                Be = pickle.load(f)
                BefirstT = Be.resample('15S').first()
                RRiBe = pd.merge(RRimeanT, BefirstT,on='time')
                question = pd.read_csv('アンケート/questionnaire_ID_0' + str(ID) + '_' + str(SEX) + '.csv',
                                       index_col='date', parse_dates=True)

                date_e = RRiBe.index[-1]
                date_s = RRiBe.index[0]
                date_s = date_s + datetime.timedelta(days=1)  
                date_s = date_s.strftime('%Y/%m/%d')
                data_sleep = datetime.datetime.strptime(date_s + ' 01:00', "%Y/%m/%d %H:%M")

                breakcount=0

                while True:#開始位置 start点 morning1:00 Night18:00
                    #Morningの時は ==0 Night==1
                    breakcount = breakcount + 1
                    if breakcount>10000:
                        break
                    try:
                        if RRiBe["Sleeping"][data_sleep] == 0:
                            data_w = data_sleep + datetime.timedelta(days=1)  

                            data_w_1 = data_w + datetime.timedelta(minutes=10)
                            data_w_2 = data_w + datetime.timedelta(minutes=30)
                            if RRiBe["Sleeping"][data_w_1]  == 0 and RRiBe["Sleeping"][data_w_2]  == 0:
                                break
                            else:
                                data_sleep = data_sleep + datetime.timedelta(minutes=10)
                        else:
                            data_w = None
                            data_sleep = data_sleep + datetime.timedelta(minutes=10)
                    except:
                        data_sleep = data_sleep + datetime.timedelta(days=1) 



                data_sleep2 = (datetime.datetime.strptime(date_s + ' 1:00', "%Y/%m/%d %H:%M"))
                data_sleep2 = data_sleep2 + datetime.timedelta(days=2)  
                while True:#終わり位置  morning18:00 Night1:00
                    breakcount = 0
                    breakcount = breakcount + 1
                    if breakcount>10000:
                        break
                    try:
                        #Morningの時は ==0 Night==1
                        if RRiBe["Sleeping"][data_sleep2] == 0:
                            data_w2 = data_sleep2
                            data_w2_1 = data_w2 + datetime.timedelta(minutes=10)
                            data_w2_2 = data_w2 + datetime.timedelta(minutes=30)
                            if RRiBe["Sleeping"][data_w2_1] == 0 and RRiBe["Sleeping"][data_w2_2] == 0:
                                break
                            else:
                                data_sleep2 = data_sleep2 + datetime.timedelta(minutes=10)
                        else:
                            data_w2 = None
                            data_sleep2 = data_sleep2 + datetime.timedelta(minutes=10)
                    except:
                        print('break1')

                td_1d = datetime.timedelta(days=1)

                MONTH=6
                DAY_S=int(data_w2.strftime('%d'))
                DAY_E=31


                ####被験者ごとはここ
                LP1=np.array([['DAY','RRiM','RRiS','LP_all',
                            'LP_Bathing', 'LP_Cooking', 'LP_Eating', 'LP_Goingout', 'LP_Sleeping', 'LP_Other',
                                'SA_Bathing', 'SA_Cooking', 'SA_Eating', 'SA_Goingout', 'SA_Sleeping', 'SA_Other',
                                #'LP_Bathing_2', 'LP_Cooking_2', 'LP_Eating_2', 'LP_Goingout_2', 'LP_Sleeping_2', 'LP_Other_2',#_2は起床後から4h
                                #'SA_Bathing_2', 'SA_Cooking_2', 'SA_Eating_2', 'SA_Goingout_2', 'SA_Sleeping_2', 'SA_Other_2',
                                'Bathing_LPSAprefer_LP', 'Cooking_LPSAprefer_LP', 'Eating_LPSAprefer_LP', 'Goingout_LPSAprefer_LP','Sleeping_LPSAprefer_LP', 'Other_LPSAprefer_LP', 
                                'Bathing_LPSAprefer_SA', 'Cooking_LPSAprefer_SA', 'Eating_LPSAprefer_SA', 'Goingout_LPSAprefer_SA','Sleeping_LPSAprefer_SA', 'Other_LPSAprefer_SA',                              
                            'M1','M2','M3']])
                            #"N1","N2","N3","N4"]])
                try:
                    LPSAprefer = makePrefer()
    
                    for Mon in range(2):
                        for DAY in range (DAY_S,DAY_E): 
                            df_whs3_area_rri_a = make_df_question(MONTH,DAY)  
                            df_whs3_area_rri_a_2 = make_df_question_2(MONTH,DAY) 

                            data_a = data_w2.strftime('%Y/%m/%d')
                            data_a1 = datetime.datetime.strptime(data_a, "%Y/%m/%d")

                            RRimeanT_LP = RRimeanT.copy(deep=True)
                            if 'RRI' in RRimeanT_LP.columns:
                                RRimeanT_LP = RRimeanT_LP.rename(columns={'RRI': 'rri'}).fillna(method='ffill')
                            else:
                                RRimeanT_LP = RRimeanT_LP.rename(columns={'Heart Rate': 'rri'}).fillna(method='ffill')
                            RRimeanT_LP_next = RRimeanT_LP.rename(columns={'rri':'rri_next'}).fillna(method='ffill').shift(-1)
                            #print(RRimeanT_LP)

                            LP_all = lorenz_plot_area(RRimeanT_LP['rri'].loc[data_w:data_w2] ,RRimeanT_LP_next['rri_next'].loc[data_w:data_w2])
                            LP_2 =  lorenz_plot_area(RRimeanT_LP['rri'].loc[data_w:data_w + datetime.timedelta(hours=4)] ,RRimeanT_LP_next['rri_next'].loc[data_w:data_w + datetime.timedelta(hours=4)])

                            point_a = question.loc[data_a1]##############
                            pointM1 = point_a['M1']
                            pointM2 = point_a['M2']
                            pointM3 = point_a['M3']


                            data_b = data_w2 - datetime.timedelta(days=1)
                            data_b2 = data_b.strftime('%Y/%m/%d')
                            data_b1 = datetime.datetime.strptime(data_b2, "%Y/%m/%d")
                            point_b = question.loc[data_b1]

                            pointN1 = point_b['N1']
                            pointN2 = point_b['N2']
                            pointN3 = point_b['N3']
                            pointN4 = point_b['N4']

                            RRiM = RRimeanT.loc[data_w:data_w2].mean()
                            RRiS = RRimeanT.loc[data_w:data_w2].std(ddof=0)
                            RRiM_2 = RRimeanT.loc[data_w:data_w + datetime.timedelta(hours=4)].mean()
                            RRiS_2 = RRimeanT.loc[data_w :data_w + datetime.timedelta(hours=4)].std(ddof=0)


                            df = df_whs3_area_rri_a
                            df2 = [np.count_nonzero(df=='Bathing'),
                                            np.count_nonzero(df=='Cooking'),np.count_nonzero(df=='Eating'),
                                                            np.count_nonzero(df=='Goingout'),
                                            np.count_nonzero(df=='Sleeping'),np.count_nonzero(df=='Other')]
                            df_2 = df_whs3_area_rri_a_2
                            df2_2 = [np.count_nonzero(df_2=='Bathing'),
                                            np.count_nonzero(df_2=='Cooking'),np.count_nonzero(df_2=='Eating'),
                                                            np.count_nonzero(df_2=='Goingout'),
                                            np.count_nonzero(df_2=='Sleeping'),np.count_nonzero(df_2=='Other')]
                            
                            ######ここ！！
                            LP = np.concatenate([[data_a],RRiM,RRiS,[LP_all], 
                                        output_LP_scale(df)['scale'].values,df2,
                                        output_LP_scale(df)['scale'].values/LPSAprefer.loc['ID'+str(ID)+'SEX'+str(SEX)],
                                        LPSAprefer.loc['ID'+str(ID)+'SEX'+str(SEX)]*df2,
                                        [pointM1,pointM2,pointM3]])

                            LP1 = np.concatenate([LP1,[LP]])
                            #前の日の終わり位置から始める。次の日の終わり位置を探す。
                            data_w = data_w2

                            data_b = data_w2.strftime('%Y/%m/%d')

                            # Morning次の日の朝１時から始める　Night次の日の夜１８時から始める
                            data_sleep2 = (datetime.datetime.strptime(data_b + ' 01:00', "%Y/%m/%d %H:%M"))
                            data_sleep2 =  data_sleep2 + datetime.timedelta(days=1)
                            breakcount = 0
                            breakcount = 0
                            while True:#終わり位置　Morning ==0 Night ==1
                                breakcount = breakcount + 1
                                if breakcount>10000:
                                    break
                                try:
                                    if RRiBe["Sleeping"][data_sleep2] == 0:
                                        data_w2 = data_sleep2
                                        data_w2_1 = data_w2 + datetime.timedelta(minutes=10)
                                        data_w2_2 = data_w2 + datetime.timedelta(minutes=30)
                                        if RRiBe["Sleeping"][data_w2_1] == 0 and RRiBe["Sleeping"][data_w2_2] == 0:
                                            break
                                        else:
                                            data_sleep2 = data_sleep2 + datetime.timedelta(minutes=10)
                                    else:
                                        data_w2 = None
                                        data_sleep2 = data_sleep2 + datetime.timedelta(minutes=10)
                                except:
                                    print('break2')

                        MONTH=7    
                        DAY_S=1
                        DAY_E=int(date_e.strftime('%d'))
                except:
                    print('break1')
            except:
                print('break3')
            
            
            LP2 = pd.DataFrame(LP1[1:],columns=LP1[0])
            LP2 = LP2.set_index('DAY')
            print('LP2')
            print(LP2)

            dtNow = datetime.datetime.now()
            f = open('分析データ/' + 'ID' + str(ID) + '/' + 'SEX' + str(SEX) + '/' + 'Morning' + str(dtNow.year) + str(dtNow.month)  +  str(dtNow.day)  + '.pickle' ,'wb')
            pickle.dump(LP2,f)
            f.close()



            # <p style="text-align: center;">
            #    ID{{ id }}<br>
            #    入眠時間 {{ sleepHourStart }}時 {{ sleepMinStart }}分（24時間表記）<br>            
            #    起床時間 {{ sleepHourEnd }}時 {{ sleepMinEnd }}分（24時間表記）<br>
            # </p>

            # id=id,
            #     sleepHourStart=sleepHourStart,sleepMinStart=sleepMinStart,sleepHourEnd=sleepHourEnd,sleepMinEnd=sleepMinEnd