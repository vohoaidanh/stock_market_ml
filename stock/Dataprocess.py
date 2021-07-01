import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import math

def load_stock(filename=r'E:\StocksData\HPG.csv'):
    '''
    

    Parameters
    ----------
    filename : TYPE, optional
        DESCRIPTION. The default is r'E:\StocksData\HPG.csv'.

    Returns: 
    -------
    TYPE : DataFrame
        DESCRIPTION: Bảng dữ liệu của cổ phiếu gồm các cột 'date','Open','High','Low','Close','Volume'

    '''
    return pd.read_csv(filename,names=['date','Open','High','Low','Close','Volume'])


def make_ma(data:pd.DataFrame,length=5,column='Close'): 
    '''
    Parameters
    ----------
    data : pd.DataFrame
        DESCRIPTION.
    length : TYPE, optional
        DESCRIPTION. The default is 5.
    column : TYPE, optional
        DESCRIPTION. The default is 'Close'.

    Returns
    -------
    TYPE
        DESCRIPTION.

    '''

    return data[column].rolling(window=length).mean()


def create_dataset2(dataset,look_back=1):
    """
    
    Parameters
    ----------
    dataset : TYPE
        DESCRIPTION.
    look_back : TYPE, optional
        DESCRIPTION. The default is 1.

    Returns
    -------
    ma5 : TYPE
        DESCRIPTION.
    ma20 : TYPE
        DESCRIPTION.
    ma50 : TYPE
        DESCRIPTION.

    """
    dataX,dataY = [],[]    
    ma50 = make_ma(dataset,length=50).dropna()
    minlen = len(ma50)
    
    ma20 = make_ma(dataset,length=20)
    ma20 = ma20[-minlen:]
    ma5 = make_ma(dataset,length=5)
    ma5 = ma5[-minlen:]
    return (ma5,ma20,ma50)


class Dataset():
    
    def __init__(self,data=''):
        self.data = data
    
    def fit(self,data):
        self.data = data
    
    def create_moving_average(self,column_name='Close',*avg):
        result_list = []
        __avg = avg
        if (len(__avg) < 1):
            __avg = 1
        for ma in avg:
            result_list.append(make_ma(self.data,length=ma,column=column_name).rename(column_name+'_ma_'+str(ma)))
        return pd.concat(result_list,axis=1)
        
    def create_dataset1(self,data:pd.DataFrame,look_back=1):
        dataX, dataY = [],[]
        
        _dataset = data.dropna()  
        _dataset1 = _dataset.values
        
        for i in range(len(_dataset1)-look_back-1):
            a = _dataset1[i:(i+look_back),0:]
            y = _dataset1[(i+look_back):(i+look_back+1),0:1]
            dataX.append(a)
            dataY.append(y)
        return np.asarray(dataX), np.asarray(dataY)
    
    def create_dataset(self):
        data = self.data
        DROP = 50
        _open = data['Open'].to_numpy()[DROP:]
        _close = data['Close'].to_numpy()[DROP:]
        _high = data['High'].to_numpy()[DROP:]
        _low = data['Low'].to_numpy()[DROP:]
        _volume = data['Volume'].to_numpy()[DROP:]
        _ma5 = make_ma(self.data,5).to_numpy()[DROP:]
        _ma20 = make_ma(self.data,20).to_numpy()[DROP:]
        _ma50 = make_ma(self.data,50).to_numpy()[DROP:]
        _vma20 = make_ma(self.data,20,'Volume').to_numpy()[DROP:]
        
        return _open,_close,_high,_low,_ma5,_ma20,_ma50,_vma20,_volume

    
class DMinMaxScaler():
    def __init__(self):
        pass
    
    def fit(self,data):
        self.data = data
    
    def transform(self):
        x_min = np.min(self.data)
        x_max = np.max(self.data)
        data_scaled = (self.data-x_min)/(x_max-x_min)
        return data_scaled

        
        

# =============================================================================
# 
# hpgData = loadStockData(filename=r'E:\StocksData\GIL.csv')
# hpgData1 = hpgData[hpgData['Volume'] != 0]
# 
# hpgData.loc[1:100]
# hpgData['Open'][98:101]
# hpgData['Open'].loc[98:100]
# 
# ma5 = make_MA(hpgData1,5,'Close')
# ma20 = make_MA(hpgData1,20,'Close')
# ma100 = make_MA(hpgData1,100,'Close')
# 
# START = -60
# LEN = 59
# 
# 
# ma5[START:START+LEN].plot(color='green')
# ma20[START:START+LEN].plot(color='red')
# ma100[START:START+LEN].plot(color='black')
# plt.legend(['MA5','MA20','MA100'])
# 
# 
# train_size = int(len(ma5)*0.67)
# test_size = len(ma5)-train_size
# 
# =============================================================================
