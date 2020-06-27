import statsmodels.api as sm
import numpy as np
import matplotlib.pyplot as plt
from date_functions import *
import time
from get_WMAE import *
import pywt
from scipy.optimize import basinhopping
from sklearn import linear_model
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LassoCV, LassoLarsCV, LassoLarsIC, LassoLars


def remove_mean(window_size,first_day_index,lprices,lloads):
    window_size_h = window_size*24
    first_hour_index = first_day_index*24
    qs = lprices[first_hour_index:first_hour_index+window_size_h]
    qs_mean=np.mean(qs)
    qs = np.reshape(qs,(-1,24)) #qs : (window_size,24)

    qs=qs-qs_mean

    zt = lloads[first_hour_index:first_hour_index+window_size_h+24]
    zt = np.reshape(zt,(-1,24)) #zt : (window_size+1,24)
    
    qsmin = np.min(qs, axis = 1, keepdims = True)
    qsmax = np.max(qs, axis = 1, keepdims = True)

    return qs, qsmin, qsmax, zt, qs_mean

def decomposition_HP(lprices, lloads, first_day_index, window_size,lambd):
    window_size_h = window_size*24
    first_hour_index=first_day_index*24
    qs = lprices[first_hour_index:first_hour_index+window_size_h]
    zt = lloads[first_hour_index:first_hour_index+window_size_h+24]

    #removing LTSC
    qs, Ts = sm.tsa.filters.hpfilter(qs, lambd)
    zt, _ = sm.tsa.filters.hpfilter(zt, lambd)
    
    qs = np.reshape(qs,(-1,24)) #qs: (window_size,24)
    zt = np.reshape(zt,(-1,24)) #zt: (window_size+1,24)
    
    qsmin = np.min(qs, axis = 1, keepdims = True)
    qsmax = np.max(qs, axis = 1, keepdims = True)

    #Ts[-24:] used as T_hat for day+1
    return qs, qsmin, qsmax, zt, Ts[-24:]

def decomposition_wavelet(lprices, lloads, first_day_index, window_size, level):
    window_size_h = window_size*24
    first_hour_index=first_day_index*24
    qs = lprices[first_hour_index:first_hour_index+window_size_h]
    zt = lloads[first_hour_index:first_hour_index+window_size_h+24]
    #removing LTSC
    mode='symmetric'
    wavelet='db24'
    coeffs = pywt.wavedec(qs, wavelet, level=14, mode=mode)
    Ts = pywt.waverec(coeffs[:15-level] + [None]*level, wavelet, mode)
    Ts = Ts[:len(qs)]
    qs = qs - Ts

    coeffs = pywt.wavedec(zt, wavelet, level=14, mode=mode)
    zt_LTSC = pywt.waverec(coeffs[:15-level] + [None]*level, wavelet, mode)
    zt_LTSC = zt_LTSC[:len(zt)]
    zt = zt -zt_LTSC
    
    qs = np.reshape(qs,(-1,24)) #qs: (window_size,24)
    zt = np.reshape(zt,(-1,24)) #zt: (window_size+1,24)
    
    qsmin = np.min(qs, axis = 1, keepdims = True)
    qsmax = np.max(qs, axis = 1, keepdims = True)
    
    #Ts[-24:] used as T_hat for day+1
    return qs, qsmin, qsmax, zt, Ts[-24:]

def get_calibartion_dataset(qs, qsmin, qsmax, zt, window_size, h, dummies):
    #qs: (window_size, 24)
    #qsmin: (window_size,1)
    #zt: (window_size+1,24)
    #dummies: [D1, ..., D7]: (window_size+1,7)
    ## returns ##
    #estimation set: X, Y
    #prediction set: Xr
    print(qs.shape)
    X=np.zeros((window_size-7,105))
    Xr=np.zeros((1,105))
    for i in range(24):
        X[:,i]=qs[6:-1,i]       # 0-23: 24 prices -1 day
        X[:,24+i]=qs[5:-2,i]    # 24-47: 24 prices -2 day
        X[:,48+i]=qs[0:-7,i]    # 48-71: 24 prices -7 day
        X[:,72+i]=zt[7:-1,i]    # 72-95: 24 loads 
        Xr[0,i]=qs[-1,i]
        Xr[0,24+i]=qs[-2,i]
        Xr[0,48+i]=qs[-7,i]
        Xr[0,72+i]=zt[-1,i]
    
    
    X[:,96:103]=dummies[7:-1,:] # 96-102: dummies [D1,D2,...,D7] 
    X[:,103]=qsmin[6:-1,0]
    X[:,104]=qsmax[6:-1,0]

    Xr[0,96:103]=dummies[-1,:]
    Xr[0,103]=qsmin[-1,0]
    Xr[0,104]=qsmax[-1,0]

    Y=qs[7:,h]
    
    return X, Y, Xr 

def get_dummies(dates):
    #changes dates (given as str 'YYYYmmdd') to dummies Mon: D1, Tue: D2, Wed: D3,..., Sun: D7
    dates_asint = dates.astype(int)
    dates_asint = np.reshape(dates_asint,(-1,24)) 
    dayofweek=np.zeros((dates_asint.shape[0],1))
    for i in range(dates_asint.shape[0]):
        dayofweek[i,0] = get_dayofweek(dates_asint[i,0])
    dummies=np.zeros((dates_asint.shape[0],7))
    for i in range(7):
        dummies[:, i] = (dayofweek == i).astype(int)[:,0]

    return dummies

def get_dummies_inwindow(dummies, window_size, first_day_index):
    dummies = dummies[first_day_index:first_day_index+window_size+1,:] 
    #D1, ..., D7: (window_size +1,7)
    return dummies

def get_estimated_parameters_LSM(X,Y):
    #XT=X.T
    sol,_,_,_=np.linalg.lstsq(X,Y)
    return sol#np.linalg.inv(np.dot(XT,X)).dot(XT).dot(Y)

def make_prediction(X,params,T): 
    return np.exp(np.dot(X,params)+T)

def loss_function(params, X, Y):
    Yhat = np.dot(X,params)
    return np.mean(np.abs(Y - Yhat))

def run_model(dataset, window_size, first_eday, last_eday, param,file_lambd,file_betas):
    #param: lambd for HP, level for wavelet
    raw_data = np.genfromtxt(f'DATA/{dataset}.txt')
    qs_real = np.reshape(raw_data[:,2],(-1,24))
    num_days=qs_real.shape[0]

    qs_predictions = np.zeros(qs_real.shape)
    dummies_all=get_dummies(raw_data[:,0])
    lprices = np.log(raw_data[:,2])
    lloads = np.log(raw_data[:,4])

    for day in range(window_size+1,num_days+1):
        first_day_index=day-(window_size+1)
        dummies = get_dummies_inwindow(dummies_all, window_size, first_day_index)
        #qs, qsmin, qsmax, zt, Ts_hat=decomposition_wavelet(lprices, lloads, first_day_index, window_size,param)
        #qs, qsmin, qsmax, zt, Ts_hat=decomposition_HP(lprices, lloads, first_day_index, window_size,param)
        qs, qsmin, qsmax, zt, Ts_hat=remove_mean(window_size, first_day_index, lprices, lloads)
        for hour in range(24):
            X, Y, Xr = get_calibartion_dataset(qs, qsmin, qsmax, zt, window_size, hour, dummies)

            model_aic = LassoLarsIC(criterion='aic',fit_intercept=False)
            fitted_model=model_aic.fit(X, Y)
            params=fitted_model.coef_
            est_lambd=fitted_model.alpha_
            file_lambd.write(str(est_lambd) + "\n")
            np.savetxt(file_betas, params.reshape(1, params.shape[0]))
            #print(params)
            c_prediction=make_prediction(Xr,params,Ts_hat) #if wavelet or HP filters are used, change Ts_hat -> Ts_hat[hour]
            qs_predictions[first_day_index+window_size,hour]=c_prediction
            print(f'(day,hour):\t({day},{hour}):\t{c_prediction}')

    date = raw_data[0,0].astype(int)
    first_day = get_datetime(date)
    WMAE, ave_num = get_WMAE(qs_real, qs_predictions, first_day, first_eday, last_eday)
    return qs_real, qs_predictions, WMAE

dataset = 'NPdata_2013-2016'#'GEFCOM_hourly'#
window_size = 360
first_eday = datetime(2013, 12, 27)#datetime(2011, 12, 27)#
last_eday = datetime(2015, 12, 24)#datetime(2013, 12, 16)#

param = np.array([1])
#param = np.array([9e-4,8e-4,7e-4,6e-4,5e-4,4e-4,3e-4,2e-4])
#param = np.array([1,0.5,1e-1,5e-2,1e-2,5e-3,1e-3,5e-4,1e-4,5e-5,1e-5,5e-6,1e-6])
#param = np.array([10**8, 5*10**8, 10**9, 5*10**9, 10**10, 5*10**10, 10**11, 5*10**11])
WMAEs=np.zeros((len(param),2))

for i in range(len(param)):
    print(param[i])
    WMAEs[i,0]=param[i]
    file_lambd = open(f'{dataset}_l4_lasso_mean_aic_lambdas.txt',"w+")
    file_betas = open(f'{dataset}_l4_lasso_mean_aic_betas.txt',"w+")
    _,_,WMAEs[i,1]=run_model(dataset, window_size, first_eday, last_eday, param[i],file_lambd,file_betas)
    #print(WMAEs)
header = "lambda, WMAE\n"
np.savetxt(f'{dataset}_l4_lasso_mean_aic.csv', WMAEs, delimiter=',', header=header)
file_lambd.close()
file_betas.close()


