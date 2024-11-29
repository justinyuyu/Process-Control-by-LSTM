# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import numpy as np
import pandas as pd
from keras.models import load_model
import matplotlib.pyplot as plt

df_train = pd.read_excel("SSED (time series).xlsx",index_col="Time", parse_dates=True)
df_test  = pd.read_excel("SSED (time series) test.xlsx",index_col="Time", parse_dates=True)

xMe_model  = load_model('my_model_LSTM-Me.h5')
xTol_model = load_model('my_model_LSTM-TOL.h5')
xEt_model  = load_model('my_model_LSTM-Et.h5')

F    = df_test.iloc[:,0:1].values
xF   = df_test.iloc[:,1:2].values
RR   = df_test.iloc[:,2:3].values
Fm   = df_test.iloc[:,3:4].values
QR   = df_test.iloc[:,4:5].values
Fs   = df_test.iloc[:,5:6].values
zMe  = df_test.iloc[:,6:7].values
zTol = df_test.iloc[:,7:8].values
zEt  = df_test.iloc[:,8:9].values

def get_range(data):
    var_range=[]
    for i in range(data.shape[1]):
        val_max = data.iloc[:,i:i+1].values.max(axis=0)[0]
        val_min = data.iloc[:,i:i+1].values.min(axis=0)[0]
        var_range.append([val_max, val_min])
    return var_range

var_range = get_range(df_train)  # a list of arrays

def minmaxsc(data_test,var_range):
    data_std=(data_test-var_range[1])/(var_range[0]-var_range[1])    
    return data_std

def inv_minmaxsc(data_std, var_range):
    data_std=var_range[1]+(var_range[0]-var_range[1])*data_std       
    return data_std


def create_dataset(ds, look_back=1):
    x_data = []
    for i in range(len(ds)-look_back):
        x_data.append(ds[i:(i+look_back),0])        
    return np.array(x_data)

F    = minmaxsc(F,var_range[0])
xF   = minmaxsc(xF,var_range[1])
RR   = minmaxsc(RR,var_range[2])
Fm   = minmaxsc(Fm,var_range[3])
QR   = minmaxsc(QR,var_range[4])
Fs   = minmaxsc(Fs,var_range[5])
zMe  = minmaxsc(zMe,var_range[6])
zTol = minmaxsc(zTol,var_range[7])
zEt  = minmaxsc(zEt,var_range[8])

look_back = 5
F_x    = create_dataset(F, look_back)
xF_x   = create_dataset(xF, look_back)
RR_x   = create_dataset(RR, look_back)
Fm_x   = create_dataset(Fm, look_back)
QR_x   = create_dataset(QR, look_back)
Fs_x   = create_dataset(Fs, look_back)
zMe_x  = create_dataset(zMe, look_back)
zTol_x = create_dataset(zTol, look_back)
zEt_x  = create_dataset(zEt, look_back)

input_x = np.stack((F_x,xF_x,RR_x,Fm_x,QR_x,Fs_x,zMe_x,zTol_x,zEt_x),axis=2)

j=5
i=1
change = np.stack((F[0],xF[0]*1.5,RR[0],Fm[0],QR[0],Fs[0]))
zMe_y  = []
zTol_y = []
zEt_y  = []

while j < 200:
    zMe_new   = xMe_model.predict(input_x)
    zTol_new  = xTol_model.predict(input_x) 
    zEt_new   = xEt_model.predict(input_x)   
    zMe_y  = np.append(zMe_y,zMe_new)
    zTol_y = np.append(zTol_y,zTol_new)
    zEt_y  = np.append(zEt_y,zEt_new)
    
    if i == 1:
        m=0
        while m<=j:
            zMe_y  = np.append(zMe_y,zMe_new)
            zTol_y = np.append(zTol_y,zTol_new)
            zEt_y  = np.append(zEt_y,zEt_new)
            m +=1
            
    if j>=10:
        F_x  = np.append(F_x, [change[0]], axis=1)
        xF_x = np.append(xF_x, [change[1]], axis=1)
        
        E1 = zMe_y[-1]-zMe_y[0]
        E2 = zMe_y[-2]-zMe_y[0]
        Kc = 2.5
        TI = 3
        RR_new_PI = RR_x[:,-1]+Kc*((E1-E2)+0.1/TI*E1)       
        RR_x = np.append(RR_x, [RR_new_PI], axis=1)
        
        E111 = zEt_y[-1]-zEt_y[0]
        E222 = zEt_y[-2]-zEt_y[0]
        Kc = -0.25
        TI = 60000
        Fm_new_PI = Fm_x[:,-1]+Kc*((E111-E222)+0.1/TI*E111)       
        Fm_x = np.append(Fm_x, [Fm_new_PI], axis=1)
        
        E11 = zTol_y[-1]-zTol_y[0]
        E22 = zTol_y[-2]-zTol_y[0]
        Kc = 0.1
        TI = 0.25
        QR_new_PI = QR_x[:,-1]+Kc*((E11-E22)+0.1/TI*E11)       
        QR_x = np.append(QR_x, [QR_new_PI], axis=1)
        
        Fs_x = np.append(Fs_x, [change[5]], axis=1)
        zMe_x = np.append(zMe_x, zMe_new, axis=1)
        zTol_x = np.append(zTol_x, zTol_new, axis=1)
        zEt_x = np.append(zEt_x, zEt_new, axis=1)

        F_x    = np.delete(F_x, 0,axis=1)
        xF_x   = np.delete(xF_x, 0,axis=1)
        RR_x   = np.delete(RR_x, 0,axis=1)
        Fm_x   = np.delete(Fm_x, 0,axis=1)
        QR_x   = np.delete(QR_x, 0,axis=1)
        Fs_x   = np.delete(Fs_x, 0,axis=1)
        zMe_x  = np.delete(zMe_x, 0,axis=1)
        zTol_x = np.delete(zTol_x, 0,axis=1)
        zEt_x  = np.delete(zEt_x, 0,axis=1)
        
    F      = np.append(F, [F_x[:,-1]], axis=0)
    xF     = np.append(xF, [xF_x[:,-1]], axis=0)
    RR     = np.append(RR, [RR_x[:,-1]], axis=0)
    Fm     = np.append(Fm, [Fm_x[:,-1]], axis=0)
    QR     = np.append(QR, [QR_x[:,-1]], axis=0)
    Fs     = np.append(Fs, [Fs_x[:,-1]], axis=0)
    print(np.shape(zMe_y), np.shape(F))
    
    input_x = np.stack((F_x,xF_x,RR_x,Fm_x,QR_x,Fs_x,zMe_x,zTol_x,zEt_x),axis=2)
        
    j+=1
    i+=1

F_new       = inv_minmaxsc(F, var_range[0])
# xF_new       = inv_minmaxsc(xF, var_range[1])
RR_new       = inv_minmaxsc(RR, var_range[2])
Fm_new       = inv_minmaxsc(Fm, var_range[3])
QR_new       = inv_minmaxsc(QR, var_range[4])
zMe_y_new   = inv_minmaxsc(zMe_y, var_range[6])
zTol_y_new  = inv_minmaxsc(zTol_y, var_range[7])
zEt_y_new   = inv_minmaxsc(zEt_y, var_range[8])

x=np.linspace(0,200,len(zMe_y_new))

fig, ax =plt.subplots(3,2)
# ax[0,0].plot(x*0.1,xF_new)
ax[0,0].plot(x*0.1,RR_new)
ax[0,1].plot(x*0.1, zMe_y_new)
ax[1,0].plot(x*0.1, zTol_y_new)
ax[1,1].plot(x*0.1,zEt_y_new)
ax[2,0].plot(x*0.1,QR_new)
ax[2,1].plot(x*0.1,Fm_new)


ax[0,0].set_title('RR')
ax[0,1].set_title('xMeOH')
ax[1,0].set_title('xTol')
ax[1,1].set_title('xEt3N')
ax[2,0].set_title('QR')
ax[2,1].set_title('Fm')

fig.tight_layout()

