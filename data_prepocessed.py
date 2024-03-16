'''
多变量时序数据的读取与处理
数据集：
    * NAB
    * SMAP
    * MSL
    * SMD
'''
import numpy as np
import pandas as pd
import os
import sys
import json

# global constant
datasets = ['NAB', 'SMAP', 'SMD', 'MBA', 'SWaT',  'WADI', 'MSL']
# data folder
output_folder = 'processed'
data_folder = 'datasets'

def convertNumpy(df):
    x = df[df.columns[3:]].values[::10, :]
    return (x - x.min(0)) / (x.ptp(0) + 1e-4)

def load_save(category,filename,dataset,dataset_folder):
    # 从文本文件加载数据
    temp_data = np.genfromtxt(fname=os.path.join(dataset_folder,category,filename),
                              dtype=np.float64,
                              delimiter=',')
    # print('data saving '+dataset+'_'+category+'_'+filename,', data shape:',temp_data.shape)
    # save data
    np.save(os.path.join(output_folder,f'SMD/{dataset}_{category}.npy'),temp_data)
    return temp_data.shape

def save_label(category,filename,dataset,dataset_folder,data_shape):
    temp = np.zeros(data_shape)
    with open(os.path.join(dataset_folder,'interpretation_label',filename),'r') as f:
        ls = f.readlines()
    for line in ls:
        # 数据格式 857-860:12,13
        # 19375-19416:1,2,3,4,6,7,16,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35,36
        pos,tag = line.split(':')[0],line.split(':')[1].split(',')
        start,end,index = int(pos.split('-')[0]), int(pos.split('-')[1]), [int(i) - 1 for i in tag]
        temp[start-1:end-1,index] = 1
    # print('label saving '+dataset+'_'+category+'_'+filename,', label shape:',temp.shape)
    np.save(os.path.join(output_folder,f'SMD/{dataset}_{category}.npy'),temp)

# 添加白噪声
def wgn(a,snr):
    min_a, max_a = np.min(a, axis=0), np.max(a, axis=0)
    x = (a - min_a) / (max_a - min_a + 0.0001)
    batch_size, len_x = x.shape
    Ps = np.sum(np.power(x, 2)) / len_x
    Pn = Ps / (np.power(10, snr / 10))
    noise = np.random.randn(len_x) * np.sqrt(Pn)
    return noise/100

# 数据标准化
def normalize(a,min_a=None,max_a=None):
    if min_a is None:
        min_a,max_a = np.min(a,axis=0),np.max(a,axis=0)
    return ((a - min_a) / (max_a - min_a + 0.0001)) + wgn(a, 50), min_a, max_a


def load_data(dataset):
    # prepocessed data storage
    folder = os.path.join(output_folder,dataset)
    os.makedirs(folder,exist_ok=True)
    '''
    SMD is made up by the following parts:
        train: The former half part of the dataset.
        test: The latter half part of the dataset.
        test_label: The label of the test set. It denotes whether a point is an anomaly.
        interpretation_label: The lists of dimensions contribute to each anomaly.
    '''
    if dataset == 'SMD':
        print('prepocessing SMD...')
        dataset_folder = 'datasets/SMD'
        file_list = os.listdir(os.path.join(dataset_folder,'train'))
        for file in file_list:
            if file.endswith('.txt'):
                train_data = load_save('train',file,file.strip('.txt'),dataset_folder)
                test_data = load_save('test',file,file.strip('.txt'),dataset_folder)
                save_label('labels',file,file.strip('.txt'),dataset_folder,test_data)
    elif dataset == 'NAB':
        print('prepocessing NAB...')
        dataset_folder = 'datasets/NAB'
        file_list = os.listdir(dataset_folder)
        with open(dataset_folder+'/labels.json')as f:
            labeldict = json.load(f)
        for file in file_list:
            if not file.endswith('.csv'):continue
            
            df = pd.read_csv(os.path.join(dataset_folder,file))
            # 数据格式：
            #   timestamp          ,value
            #   2013-07-04 00:00:00,69.88083514
            values = df.values[:,-1]
            labels = np.zeros_like(values,dtype=np.float64)

            for timestamp in labeldict['realKnownCause/' + file]:
                # 对label进行数据清理
                # json文件里标注的时间点代表异常点
                tstamp = timestamp.replace('.000000', '')
                #  当where内只有一个参数时，那个参数表示条件，当条件成立时，where返回的是每个符合condition条件元素的坐标,返回的是以元组的形式
                index = np.where(((df['timestamp'] == tstamp).values + 0) == 1)[0][0]
                # print(index)
                labels[index - 4:index + 4] = 1

            min_t,max_t = np.min(values),np.max(values)
            # 标准化数据
            values = (values - min_t) / (max_t - min_t)
            train, test = values.astype(float), values.astype(float)
            train, test, labels = train.reshape(-1, 1), test.reshape(-1, 1), labels.reshape(-1, 1)
            fn = file.replace('.csv','')
            
            for f in ['train','test','labels']:
                np.save(os.path.join(folder,f'{fn}_{f}.npy'),eval(f))
                # print(f'{fn}_{f}.npy saved.')
    elif dataset in ['SMAP','MSL']:
        dataset_folder = 'datasets/SMAP_MSL'
        print('prepocessing SMAP...')
        file = os.path.join(dataset_folder,'labeled_anomalies.csv')
        values = pd.read_csv(file)
        # 数据集格式
        # chan_id,spacecraft,anomaly_sequences,class,num_values
        # P-1,SMAP,"[[2149, 2349], [4536, 4844], [3539, 3779]]","[contextual, contextual, contextual]",8505
        # spacecraft 数据集名称
        values = values[values['spacecraft'] == dataset]
        filename = values['chan_id'].values.tolist()
        for fn in filename:
            train = np.load(f'{dataset_folder}/train/{fn}.npy')
            test = np.load(f'{dataset_folder}/test/{fn}.npy')
            '''
            将训练和测试数据标准化，
            并添加信噪比为50dB的白噪声作为数据增强，
            最后划分为滑动时间窗口
            '''
            train,min_a,max_a = normalize(train)
            test,_,_ = normalize(test, min_a, max_a)
            np.save(f'{folder}/{fn}_train.npy', train)
            np.save(f'{folder}/{fn}_test.npy', test)
            labels = np.zeros(test.shape)
            indices = values[values['chan_id'] == fn]['anomaly_sequences'].values[0]
            indices = indices.replace('[','').replace(']','').split(',')
            indices = [int(i) for i in indices]
            for i in range(0, len(indices), 2):
                labels[indices[i]:indices[i + 1], :] = 1
            np.save(f'{folder}/{fn}_labels.npy', labels)
            # print(f'{folder}/{fn}_labels.npy saved.')
    elif dataset == 'MBA':
        print('prepocessing MBA...')
        dataset_folder = 'datasets/MBA'
        ls = pd.read_excel(os.path.join(dataset_folder, 'labels.xlsx'))
        train = pd.read_excel(os.path.join(dataset_folder, 'train.xlsx'))
        test = pd.read_excel(os.path.join(dataset_folder, 'test.xlsx'))
        train, test = train.values[1:, 1:].astype(float), test.values[1:, 1:].astype(float)
        train, min_a, max_a = normalize(train)
        test, _, _ = normalize(test, min_a, max_a)
        ls = ls.values[:, 1].astype(int)
        labels = np.zeros_like(test)
        for i in range(-20, 20):
            labels[ls + i, :] = 1
        for file in ['train', 'test', 'labels']:
            np.save(os.path.join(folder, f'{file}.npy'), eval(file))
    elif dataset == 'SWaT':
        print('prepocessing SWaT...')
        dataset_folder = 'datasets/SWaT'
        file = os.path.join(dataset_folder, 'series.json')
        df_train = pd.read_json(file, lines=True)[['val']][3000:6000]
        df_test = pd.read_json(file, lines=True)[['val']][7000:12000]
        train, min_a, max_a = normalize(df_train.values)
        test, _, _ = normalize(df_test.values, min_a, max_a)
        labels = pd.read_json(file, lines=True)[['noti']][7000:12000] + 0
        for file in ['train', 'test', 'labels']:
            np.save(os.path.join(folder, f'{file}.npy'), eval(file))
    elif dataset == 'WADI':
        print('prepocessing WADI...')
        dataset_folder = 'datasets/WADI'
        ls = pd.read_csv(os.path.join(dataset_folder, 'WADI_attacklabels.csv'))
        train = pd.read_csv(os.path.join(dataset_folder, 'WADI_14days.csv'), skiprows=1000, nrows=2e5)
        test = pd.read_csv(os.path.join(dataset_folder, 'WADI_attackdata.csv'))
        train.dropna(how='all', inplace=True)
        test.dropna(how='all', inplace=True)
        train.fillna(0, inplace=True)
        test.fillna(0, inplace=True)
        test['Time'] = test['Time'].astype(str)
        test['Time'] = pd.to_datetime(test['Date'] + ' ' + test['Time'])
        labels = test.copy(deep=True)
        for i in test.columns.tolist()[3:]: labels[i] = 0
        for i in ['Start Time', 'End Time']:
            ls[i] = ls[i].astype(str)
            ls[i] = pd.to_datetime(ls['Date'] + ' ' + ls[i])
        for index, row in ls.iterrows():
            to_match = row['Affected'].split(', ')
            matched = []
            for i in test.columns.tolist()[3:]:
                for tm in to_match:
                    if tm in i:
                        matched.append(i)
                        break
            st, et = str(row['Start Time']), str(row['End Time'])
            labels.loc[(labels['Time'] >= st) & (labels['Time'] <= et), matched] = 1
        train, test, labels = convertNumpy(train), convertNumpy(test), convertNumpy(labels)
        # print(train.shape, test.shape, labels.shape)
        for file in ['train', 'test', 'labels']:
            np.save(os.path.join(folder, f'{file}.npy'), eval(file))
    

if __name__ == '__main__':
    print('Data loading...')
    for data in datasets:
        load_data(data)
        print(f'{data} prepocessed.')
    print('Preprocess finished.')
