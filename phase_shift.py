import cv2

import numpy as np
import matplotlib.pyplot as plt

def sample_test1(img):
    #return img[159, 73:393]
    return img[159, 73:193]
    
def sample_test2():
    signal = np.sin(np.linspace(0, 60 * np.pi, 1200))
    return signal
    
def divide_4signals(signal):
    s1 = signal[0::4]
    s2 = signal[1::4]
    s3 = signal[2::4]
    s4 = signal[3::4]
    return s1, s2, s3, s4
    
    
def divide_signals(signal, div_num):
    signals = np.zeros((div_num, len(signal)/ div_num))
    for i in range(div_num):
        signals[i,:] = signal[i::div_num]
    
    return signals
    
def upsample(s, div_num = 4, offset = 0):
    
    signal_len = len(s)

    dst = np.zeros((signal_len * div_num))
    
    for i in range(offset):
        dst[i] = s[0]
    
    src_id = 0
    count = 0
    for i in range(0, signal_len - 1):
        dst[4 * i + offset] = s[i]
        dst[4 * i + offset + 1] = s[i] * ( 1 - 1.0 / div_num ) + s[i+1] * ( 1.0 / div_num )
        dst[4 * i + offset + 2] = s[i] * ( 1 - 2.0 / div_num ) + s[i+1] * ( 2.0 / div_num )
        dst[4 * i + offset + 3] = s[i] * ( 1 - 3.0 / div_num ) + s[i+1] * ( 3.0 / div_num )
        
    for i in range(offset):
        dst[4 * (signal_len - 1) + i] = s[-1]
            
    return dst

def upsample2(s, div_num, offset):
    
    signal_len = s.shape[0]

    dst = np.zeros((signal_len * div_num))
    
    for i in range(offset):
        dst[i] = s[0]
    

    for i in range(0, signal_len - 1):
        for d in range(div_num):
        
            dst[div_num * i + offset + d] = s[i] * ( 1 - float(d) / div_num ) + s[i+1] * ( float(d) / div_num )
        
    for i in range(offset):
        dst[div_num * (signal_len - 1) + i] = s[-1]
            
    return dst

def calcTheta(s1, s2, s3, s4):
    Nr = 4
    
    denom = 0 # 分母
    numer = 0 # 分子
    
    denom += s1 * np.cos(0)
    numer += s1 * np.sin(0)    
    
    denom += s2 * np.cos(1 * 2 * np.pi / Nr)
    numer += s2 * np.sin(1 * 2 * np.pi / Nr)
    
    denom += s3 * np.cos(2 * 2 * np.pi / Nr)
    numer += s3 * np.sin(2 * 2 * np.pi / Nr)
    
    denom += s4 * np.cos(3 * 2 * np.pi / Nr)
    numer += s4 * np.sin(3 * 2 * np.pi / Nr)
    
    theta = np.arctan2( numer, denom )
    
    return theta
    

def calcThetas( signals ):
    Nr = signals.shape[0]
    
    denom = 0 # 分母
    numer = 0 # 分子
    
    for k, signal in enumerate(signals):    
        denom += signal * np.cos(k * 2 * np.pi / Nr)
        numer += signal * np.sin(k * 2 * np.pi / Nr)  
    
    theta = np.arctan2( numer, denom )
    
    return theta
    
    
    

def test1():
    print('aa')
    
    img = cv2.imread('pattern1.png', 0)

    signal = sample_test1(img)    
    
    # graph plot
    plt.plot(signal)
    plt.show()
    
    #%%
    s1, s2, s3, s4 = divide_4signals(signal)
    
    #%%アップサンプル後
    S1 = upsample(s1, 4, 0)
    S2 = upsample(s2, 4, 1)
    S3 = upsample(s3, 4, 2)
    S4 = upsample(s4, 4, 3)
    
    plt.plot(S1, color='r')
    plt.plot(S2, color='g')
    plt.plot(S3, color='b')
    plt.plot(S4, color='y')
    plt.show()
    
    thetaList = np.zeros((len(signal)))
    
    for i in range(len(thetaList)):
        thetaList[i] = calcTheta(S1[i], S2[i], S3[i], S4[i])
        
    plt.plot(thetaList)
    print('aaa')
    
    
#%%
def test2():
    print('bb')
    
    img = cv2.imread('pattern1.png', 0)

    #signal = sample_test1(img)
    signal = sample_test2()
    signal_len = len(signal)
    
    # graph plot
    plt.plot(signal)
    plt.show()
    
    #%%
    div_num = 30
    signals = divide_signals(signal, div_num)
    #signals = signals.T
    
    plot_num = min(div_num , 3)
    for d in range(plot_num):
        plt.plot(signals[d])
    plt.show()
    
    #%%アップサンプル後
    upSignals = np.zeros((div_num, signal_len))
    for d in range(div_num):
        upSignals[d] = upsample2(signals[d], div_num, d)
    
    
    plt.plot(upSignals[0], color='r')
    plt.plot(upSignals[1], color='g')
    plt.plot(upSignals[2], color='b')
    plt.plot(upSignals[3], color='y')
    plt.show()
    
    thetaList = np.zeros((len(signal)))
    
    for i in range(len(thetaList)):        
        thetaList[i] = calcThetas(upSignals[:, i])
        
    plt.plot(thetaList)
    plt.show()
    print('aaa')
    
#%%
if __name__ == '__main__':
    #test1()
    test2()