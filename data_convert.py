# 将时域数据通过小波变换转化为频域数据
import pandas as pd
import numpy as np
import os
# 小波变换
from pywt import wavedec,waverec
# 进行平稳性检验
from statsmodels.tsa.stattools import adfuller
from tqdm import trange
'''
test:
    load the data
'''
data_path = 'processed'
freq_path = 'frequence'

#SMD
smd_path = os.listdir(os.path.join(data_path,'SMD'))
os.makedirs('frequence/SMD',exist_ok=True)
# MSL
msl_path = os.listdir(os.path.join(data_path,'MSL'))
os.makedirs('frequence/MSL',exist_ok=True)
# NAB
nab_path = os.listdir(os.path.join(data_path,'NAB'))
os.makedirs('frequence/NAB',exist_ok=True)
# SMAP
smap_path = os.listdir(os.path.join(data_path,'SMAP'))
os.makedirs('frequence/SMAP',exist_ok=True)
# MBA
mba_path = os.listdir(os.path.join(data_path,'MBA'))
os.makedirs('frequence/MBA',exist_ok=True)
# SWaT
swat_path = os.listdir(os.path.join(data_path,'SWaT'))
os.makedirs('frequence/SWaT',exist_ok=True)
# WADI
wadi_path = os.listdir(os.path.join(data_path,'WADI'))
os.makedirs('frequence/WADI',exist_ok=True)

'''
ADF检验是一种基于单位根检验的方法，
用于检验一个时间序列是否具有单位根，
即是否具有非平稳性。
检验的原假设是存在单位根，即非平稳性，
备择假设是不存在单位根，即平稳性。
通过ADF检验可以得到一个统计量和一个p值，
如果p值小于显著性水平（通常为0.05或0.01），
则可以拒绝原假设，认为序列是平稳的。
'''
def ADF_test(data):
    for i in trange(data.shape[1]):
        result = adfuller(data[:,i])
        # 输出检验结果
        # print('p值：', result[1])
        if result[1] < 0.05:
            pass
        else:
            print(f'变量{i}为非平稳的，p值为{result[1]:.4f}')
        # print('Lags Used：', result[2])
        # print('Number of Observations Used：', result[3])
        # print('Critical Values：')
        # for key, value in result[4].items():
        #     print('\t%s: %.3f' % (key, value))

# def convert_to_freq():
#     for smd in smd_path:
#         if 'train' in smd:
#             # print(f'convert {smd} to frequnce sequence...')
#             smd_train = np.load(os.path.join(data_path,'SMD',smd))
#             shape = smd_train.shape[1]
#             smd_train = smd_train.flatten()
#             # print(smd_train.shape)
#             # 转为频率矩阵，在之前需要判断数据是否平稳
#             # 进行ADF检验，一次对一个变量进行检验
#             # ADF_test(smd_train)
#             # 非平稳序列使用小波变换转为频域数据
#             wavelet = 'db4'
#             level = 4
#             coeffs_train = wavedec(smd_train, wavelet, level=level)
#             # 对小波变换后重建成频域数据
#             reconstructed = waverec(coeffs_train, wavelet)
#             temp = smd.split('.')[0]
#             reconstructed = np.array(reconstructed,dtype=object).reshape(-1,shape)
#             np.save(os.path.join(freq_path,f'SMD/{temp}_freq'),reconstructed)
#         if 'test' in smd:
#             # print(f'convert {smd} to frequnce sequence...')
#             smd_test = np.load(os.path.join(data_path,'SMD',smd))
#             shape = smd_test.shape[1]
#             smd_test = smd_test.flatten()
#             # ADF_test(smd_test)
#             wavelet = 'db4'
#             level = 4
#             coeffs_test = wavedec(smd_test, wavelet, level=level)
#             # 对小波变换后重建成频域数据
#             reconstructed = waverec(coeffs_test, wavelet)
#             temp = smd.split('.')[0]
#             reconstructed = np.array(reconstructed,dtype=object).reshape(-1,shape)
#             np.save(os.path.join(freq_path,f'SMD/{temp}_freq'),reconstructed)

#     for msl in msl_path:
#         # MSL数据集会出现小波变换重建后维度与原数据不匹配的问题
#         if 'train' in msl:
#             # print(f'convert {msl} to frequnce sequence...')
#             msl_train = np.load(os.path.join(data_path,'MSL',msl))
#             shape = msl_train.shape[1]
#             msl_train = msl_train.flatten()
#             wavelet = 'db4'
#             level = 4
#             coeffs_train = wavedec(msl_train, wavelet, level=level)
#             # 对小波变换后重建成频域数据
#             reconstructed = waverec(coeffs_train, wavelet)
#             temp = msl.split('.')[0]
#             # print(msl_train.shape[0],reconstructed.shape)
#             reconstructed = reconstructed[:msl_train.shape[0]]
#             reconstructed = np.array(reconstructed,dtype=object).reshape(-1,shape)
#             np.save(os.path.join(freq_path,f'MSL/{temp}_freq'),reconstructed)
#         if 'test' in msl:
#             # print(f'convert {msl} to frequnce sequence...')
#             msl_test = np.load(os.path.join(data_path,'MSL',msl))
#             shape = msl_test.shape[1]
#             msl_test = msl_test.flatten()
#             wavelet = 'db4'
#             level = 4
#             coeffs_test = wavedec(msl_test, wavelet, level=level)
#             # 对小波变换后重建成频域数据
#             reconstructed = waverec(coeffs_test, wavelet)
#             temp = msl.split('.')[0]
#             reconstructed = reconstructed[:msl_test.shape[0]]
#             reconstructed = np.array(reconstructed,dtype=object).reshape(-1,shape)
#             np.save(os.path.join(freq_path,f'MSL/{temp}_freq'),reconstructed)

#     for nab in nab_path:
#         if 'train' in nab:
#             # print(f'convert {nab} to frequnce sequence...')
#             nab_train = np.load(os.path.join(data_path,'NAB',nab))
#             shape = nab_train.shape[1]
#             nab_train = nab_train.flatten()
#             wavelet = 'db4'
#             level = 4
#             coeffs_train = wavedec(nab_train, wavelet, level=level)
#             # 对小波变换后重建成频域数据
#             reconstructed = waverec(coeffs_train, wavelet)
#             temp = nab.split('.')[0]
#             reconstructed = np.array(reconstructed,dtype=object).reshape(-1,shape)
#             np.save(os.path.join(freq_path,f'NAB/{temp}_freq'),reconstructed)
#         if 'test' in nab:
#             # print(f'convert {nab} to frequnce sequence...')
#             nab_test = np.load(os.path.join(data_path,'NAB',nab))
#             shape = nab_test.shape[1]
#             nab_test = nab_test.flatten()
#             wavelet = 'db4'
#             level = 4
#             coeffs_test = wavedec(nab_test, wavelet, level=level)
#             # 对小波变换后重建成频域数据
#             reconstructed = waverec(coeffs_test, wavelet)
#             temp = nab.split('.')[0]
#             reconstructed = np.array(reconstructed,dtype=object).reshape(-1,shape)
#             np.save(os.path.join(freq_path,f'NAB/{temp}_freq'),reconstructed)

#     for smap in smap_path:
#         if 'train' in smap:
#             # print(f'convert {smap} to frequnce sequence...')
#             smap_train = np.load(os.path.join(data_path,'SMAP',smap))
#             shape = smap_train.shape[1]
#             smap_train = smap_train.flatten()
#             wavelet = 'db4'
#             level = 4
#             coeffs_train = wavedec(smap_train, wavelet, level=level)
#             # 对小波变换后重建成频域数据
#             reconstructed = waverec(coeffs_train, wavelet)
#             temp = smap.split('.')[0]
#             reconstructed = reconstructed[:smap_train.shape[0]]
#             reconstructed = np.array(reconstructed,dtype=object).reshape(-1,shape)
#             np.save(os.path.join(freq_path,f'SMAP/{temp}_freq'),reconstructed)
#         if 'test' in smap:
#             # print(f'convert {smap} to frequnce sequence...')
#             smap_test = np.load(os.path.join(data_path,'SMAP',smap))
#             shape = smap_test.shape[1]
#             smap_test = smap_test.flatten()
#             wavelet = 'db4'
#             level = 4
#             coeffs_test = wavedec(smap_test, wavelet, level=level)
#             # 对小波变换后重建成频域数据
#             reconstructed = waverec(coeffs_test, wavelet)
#             temp = smap.split('.')[0]
#             reconstructed = reconstructed[:smap_test.shape[0]]
#             reconstructed = np.array(reconstructed,dtype=object).reshape(-1,shape)
#             np.save(os.path.join(freq_path,f'SMAP/{temp}_freq'),reconstructed)

#     for mba in mba_path:
#         if 'train' in mba:
#             # print(f'convert {mba} to frequnce sequence...')
#             mba_train = np.load(os.path.join(data_path,'MBA',mba))
#             shape = mba_train.shape[1]
#             mba_train = mba_train.flatten()
#             wavelet = 'db4'
#             level = 4
#             coeffs_train = wavedec(mba_train, wavelet, level=level)
#             # 对小波变换后重建成频域数据
#             reconstructed = waverec(coeffs_train, wavelet)
#             temp = mba.split('.')[0]
#             reconstructed = np.array(reconstructed,dtype=object).reshape(-1,shape)
#             np.save(os.path.join(freq_path,f'MBA/{temp}_freq'),reconstructed)
#         if 'test' in mba:
#             # print(f'convert {mba} to frequnce sequence...')
#             mba_test = np.load(os.path.join(data_path,'MBA',mba))
#             shape = mba_test.shape[1]
#             mba_test = mba_test.flatten()
#             wavelet = 'db4'
#             level = 4
#             coeffs_test = wavedec(mba_test, wavelet, level=level)
#             # 对小波变换后重建成频域数据
#             reconstructed = waverec(coeffs_test, wavelet)
#             temp = mba.split('.')[0]
#             reconstructed = np.array(reconstructed,dtype=object).reshape(-1,shape)
#             np.save(os.path.join(freq_path,f'MBA/{temp}_freq'),reconstructed)

#     for swat in swat_path:
#         if 'train' in swat:
#             # print(f'convert {mba} to frequnce sequence...')
#             swat_train = np.load(os.path.join(data_path,'SWaT',swat))
#             shape = swat_train.shape[1]
#             swat_train = swat_train.flatten()
#             wavelet = 'db4'
#             level = 4
#             coeffs_train = wavedec(swat_train, wavelet, level=level)
#             # 对小波变换后重建成频域数据
#             reconstructed = waverec(coeffs_train, wavelet)
#             temp = swat.split('.')[0]
#             reconstructed = np.array(reconstructed,dtype=object).reshape(-1,shape)
#             np.save(os.path.join(freq_path,f'SWaT/{temp}_freq'),reconstructed)
#         if 'test' in swat:
#             # print(f'convert {mba} to frequnce sequence...')
#             swat_test = np.load(os.path.join(data_path,'SWaT',swat))
#             shape = swat_test.shape[1]
#             swat_test = swat_test.flatten()
#             wavelet = 'db4'
#             level = 4
#             coeffs_test = wavedec(swat_test, wavelet, level=level)
#             # 对小波变换后重建成频域数据
#             reconstructed = waverec(coeffs_test, wavelet)
#             temp = swat.split('.')[0]
#             reconstructed = np.array(reconstructed,dtype=object).reshape(-1,shape)
#             np.save(os.path.join(freq_path,f'SWaT/{temp}_freq'),reconstructed)

#     for wadi in wadi_path:
#         if 'train' in wadi:
#             wadi_train = np.load(os.path.join(data_path,'WADI',wadi))
#             shape = wadi_train.shape[1]
#             wadi_train = wadi_train.flatten()
#             wavelet = 'db4'
#             level = 4
#             coeffs_train = wavedec(wadi_train, wavelet, level=level)
#             # 对小波变换后重建成频域数据
#             reconstructed = waverec(coeffs_train, wavelet)
#             temp = wadi.split('.')[0]
#             reconstructed = np.array(reconstructed,dtype=object).reshape(-1,shape)
#             np.save(os.path.join(freq_path,f'WADI/{temp}_freq'),reconstructed)
#         if 'test' in wadi:
#             wadi_test = np.load(os.path.join(data_path,'WADI',wadi))
#             shape = wadi_test.shape[1]
#             wadi_test = wadi_test.flatten()
#             wavelet = 'db4'
#             level = 4
#             coeffs_test = wavedec(wadi_test, wavelet, level=level)
#             # 对小波变换后重建成频域数据
#             reconstructed = waverec(coeffs_test, wavelet)
#             # 不知道为什么WADI小波重建后会比原始数据多一
#             reconstructed = reconstructed[:wadi_test.shape[0]]
#             temp = wadi.split('.')[0]
#             reconstructed = np.array(reconstructed,dtype=object).reshape(-1,shape)
#             np.save(os.path.join(freq_path,f'WADI/{temp}_freq'),reconstructed)

'''
对时域数据通过短时傅里叶变换stft转换为频域
'''
from scipy.signal import stft

def convert_to_freq():
    # 设置STFT参数
    fs = 1000  # 采样率

    for smd in smd_path:
        nperseg = 74  # 每个窗口的长度
        noverlap = 37  # 窗口之间的重叠量
        if 'train' in smd:
            smd_train = np.load(os.path.join(data_path,'SMD',smd))
            smd_train = smd_train.flatten()
            # 执行STFT
            frequencies, times, Z = stft(
                smd_train, fs=fs, nperseg=nperseg, noverlap=noverlap)
            Z = Z.transpose()
            temp = smd.split('.')[0]
            real_Z = np.real(Z)
            np.save(os.path.join(freq_path,f'SMD/{temp}_freq'),real_Z)
        if 'test' in smd:
            smd_test = np.load(os.path.join(data_path,'SMD',smd))
            smd_test = smd_test.flatten()
            # 执行STFT
            frequencies, times, Z = stft(
                smd_test, fs=fs, nperseg=nperseg, noverlap=noverlap)
            Z = Z.transpose()
            temp = smd.split('.')[0]
            real_Z = np.real(Z)
            np.save(os.path.join(freq_path,f'SMD/{temp}_freq'),real_Z)
    for msl in msl_path:
        nperseg = 108  # 每个窗口的长度
        noverlap = 54  # 窗口之间的重叠量
        if 'train' in msl:
            msl_train = np.load(os.path.join(data_path,'MSL',msl))
            msl_train = msl_train.flatten()
            # 执行STFT
            frequencies, times, Z = stft(
                msl_train, fs=fs, nperseg=nperseg, noverlap=noverlap)
            Z = Z.transpose()
            temp = msl.split('.')[0]
            real_Z = np.real(Z)
            np.save(os.path.join(freq_path,f'MSL/{temp}_freq'),real_Z)
        if 'test' in msl:
            msl_test = np.load(os.path.join(data_path,'MSL',msl))
            msl_test = msl_test.flatten()
            # 执行STFT
            frequencies, times, Z = stft(
                msl_test, fs=fs, nperseg=nperseg, noverlap=noverlap)
            Z = Z.transpose()
            temp = msl.split('.')[0]
            real_Z = np.real(Z)
            np.save(os.path.join(freq_path,f'MSL/{temp}_freq'),real_Z)
    for nab in nab_path:
        nperseg = 1  # 每个窗口的长度
        noverlap = 0  # 窗口之间的重叠量
        if 'train' in nab:
            nab_train = np.load(os.path.join(data_path,'NAB',nab))
            nab_train = nab_train.flatten()
            # 执行STFT
            frequencies, times, Z = stft(
                nab_train, fs=fs, nperseg=nperseg, noverlap=noverlap)
            Z = Z.transpose()
            temp = nab.split('.')[0]
            real_Z = np.real(Z)
            np.save(os.path.join(freq_path,f'NAB/{temp}_freq'),real_Z)
        if 'test' in nab:
            nab_test = np.load(os.path.join(data_path,'NAB',nab))
            nab_test = nab_test.flatten()
            # 执行STFT
            frequencies, times, Z = stft(
                nab_test, fs=fs, nperseg=nperseg, noverlap=noverlap)
            Z = Z.transpose()
            temp = nab.split('.')[0]
            real_Z = np.real(Z)
            np.save(os.path.join(freq_path,f'NAB/{temp}_freq'),real_Z)
    for smap in smap_path:
        nperseg = 48  # 每个窗口的长度
        noverlap = 24  # 窗口之间的重叠量
        if 'train' in smap:
            smap_train = np.load(os.path.join(data_path,'SMAP',smap))
            smap_train = smap_train.flatten()
            # 执行STFT
            frequencies, times, Z = stft(
                smap_train, fs=fs, nperseg=nperseg, noverlap=noverlap)
            Z = Z.transpose()
            temp = smap.split('.')[0]
            real_Z = np.real(Z)
            np.save(os.path.join(freq_path,f'SMAP/{temp}_freq'),real_Z)
        if 'test' in smap:
            smap_test = np.load(os.path.join(data_path,'SMAP',smap))
            smap_test = smap_test.flatten()
            # 执行STFT
            frequencies, times, Z = stft(
                smap_test, fs=fs, nperseg=nperseg, noverlap=noverlap)
            Z = Z.transpose()
            temp = smap.split('.')[0]
            real_Z = np.real(Z)
            np.save(os.path.join(freq_path,f'SMAP/{temp}_freq'),real_Z)
    for mba in mba_path:
        nperseg = 2  # 每个窗口的长度
        noverlap = 1  # 窗口之间的重叠量
        if 'train' in mba:
            mba_train = np.load(os.path.join(data_path,'MBA',mba))
            mba_train = mba_train.flatten()
            # 执行STFT
            frequencies, times, Z = stft(
                mba_train, fs=fs, nperseg=nperseg, noverlap=noverlap)
            Z = Z.transpose()
            temp = mba.split('.')[0]
            real_Z = np.real(Z)
            np.save(os.path.join(freq_path,f'MBA/{temp}_freq'),real_Z)
        if 'test' in mba:
            mba_test = np.load(os.path.join(data_path,'MBA',mba))
            mba_test = mba_test.flatten()
            # 执行STFT
            frequencies, times, Z = stft(
                mba_test, fs=fs, nperseg=nperseg, noverlap=noverlap)
            Z = Z.transpose()
            temp = mba.split('.')[0]
            real_Z = np.real(Z)
            np.save(os.path.join(freq_path,f'MBA/{temp}_freq'),real_Z)
    for swat in swat_path:
        nperseg = 1  # 每个窗口的长度
        noverlap = 0  # 窗口之间的重叠量
        if 'train' in swat:
            swat_train = np.load(os.path.join(data_path,'SWaT',swat))
            swat_train = swat_train.flatten()
            # 执行STFT
            frequencies, times, Z = stft(
                swat_train, fs=fs, nperseg=nperseg, noverlap=noverlap)
            Z = Z.transpose()
            temp = swat.split('.')[0]
            real_Z = np.real(Z)
            np.save(os.path.join(freq_path,f'SWaT/{temp}_freq'),real_Z)
        if 'test' in swat:
            swat_test = np.load(os.path.join(data_path,'SWaT',swat))
            swat_test = swat_test.flatten()
            # 执行STFT
            frequencies, times, Z = stft(
                swat_test, fs=fs, nperseg=nperseg, noverlap=noverlap)
            Z = Z.transpose()
            temp = swat.split('.')[0]
            real_Z = np.real(Z)
            np.save(os.path.join(freq_path,f'SWaT/{temp}_freq'),real_Z)
    for wadi in wadi_path:
        nperseg = 252  # 每个窗口的长度
        noverlap = 126  # 窗口之间的重叠量
        if 'train' in wadi:
            wadi_train = np.load(os.path.join(data_path,'WADI',wadi))
            wadi_train = wadi_train.flatten()
            # 执行STFT
            frequencies, times, Z = stft(
                wadi_train, fs=fs, nperseg=nperseg, noverlap=noverlap)
            Z = Z.transpose()
            temp = wadi.split('.')[0]
            real_Z = np.real(Z)
            np.save(os.path.join(freq_path,f'WADI/{temp}_freq'),real_Z)
        if 'test' in wadi:
            wadi_test = np.load(os.path.join(data_path,'WADI',wadi))
            wadi_test = wadi_test.flatten()
            # 执行STFT
            frequencies, times, Z = stft(
                wadi_test, fs=fs, nperseg=nperseg, noverlap=noverlap)
            Z = Z.transpose()
            temp = wadi.split('.')[0]
            real_Z = np.real(Z)
            np.save(os.path.join(freq_path,f'WADI/{temp}_freq'),real_Z)
