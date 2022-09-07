# -*- coding: utf-8 -*-

import matplotlib.pyplot as plt
from scipy.io import loadmat
import numpy as np
from sklearn.metrics import r2_score

def quota(real,pred,name):
    mape = np.mean(np.abs((pred - real) / real))
    # rmse
    rmse = np.sqrt(np.mean(np.square(pred - real)))
    # mae
    mae = np.mean(np.abs(pred - real))
    # R2
    r2 = r2_score(real, pred)
    print(name,'的mape:', mape, ' rmse:', rmse, ' mae:', mae, ' R2:', r2)


data0=loadmat('结果/MLP_result.mat')['true']
data1=loadmat('结果/MLP_result.mat')['pred']
data2=loadmat('结果/lstm_result.mat')['pred']
data3=loadmat('结果/ssa_lstm_result.mat')['pred']

quota(data0,data1,'MLP')
quota(data0,data2,'LSTM')
quota(data0,data3,'SSA-LSTM')
plt.rcParams['xtick.direction'] = 'in'
plt.rcParams['ytick.direction'] = 'in'
plt.figure()
plt.subplot(2,2,1)
plt.plot(data0,c='r', label='real')
plt.plot(data1,c='b',label='pred')
plt.ylabel('MLP')
plt.legend()

plt.subplot(2,2,2)
plt.plot(data0,c='r', label='real')
plt.plot(data2,c='b',label='pred')
plt.ylabel('LSTM')
plt.legend()

plt.subplot(2,2,3)
plt.plot(data0,c='r', label='real')
plt.plot(data3,c='b',label='pred')
plt.legend()
plt.xlabel('time/h')
plt.ylabel('SSA-LSTM')

# In[7] 画图
plt.subplot(2,2,4)
plt.plot(data0,'-',label='real')
plt.plot(data1,'-*',label='MLP')
plt.plot(data2,'-*',label='LSTM')
plt.plot(data3,'-*',label='SSA-LSTM')
plt.grid()
plt.legend()
plt.xlabel('time/h')
plt.ylabel('Compare')
plt.show()
