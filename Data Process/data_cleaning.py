import pandas as pd
import numpy as np
from collections import OrderedDict
import matplotlib.pyplot as plt
import matplotlib.colors
import matplotlib.dates as mdates
from mpl_toolkits import mplot3d
import seaborn as sns
from datetime import datetime

def count2pcuflow(date):# 輸出流量資料
    pce = 1.4 #大型車之小客車當量(根據公路容量手冊建議)
    psg = read_data("D:/VD_data/%s/%s_psg_vd2.csv"%(date,date))
    lag = read_data("D:/VD_data/%s/%s_lag_vd2.csv"%(date,date))
    tr = read_data("D:/VD_data/%s/%s_tr_vd2.csv"%(date,date))
    flow = psg.add(lag, fill_value=0)
    flow = flow.add(tr, fill_value=0)
    flow.to_csv('D:/VD_data/%s/%s_flow_vd2.csv'%(date,date), sep=',', encoding = 'big5')
    pcu = psg.add(lag*pce, fill_value=0)
    pcu = flow.add(tr*pce, fill_value=0)
    pcu.to_csv('D:/VD_data/%s/%s_pcu_vd2.csv'%(date,date), sep=',', encoding = 'big5')

def read_data(file,section='none'):#讀取交通資料，可選取主線或匝道
    '''time = [int(i/60)*100+i%60 for i in range(0,1439)]
    for i in range(0,10): time[i] = '000' + str(time[i])
    for i in range(10,60): time[i] = '00' + str(time[i])
    for i in range(60,600): time[i] = '0' + str(time[i])
    for i in range(600,1439): time[i] = str(time[i])'''
    data = pd.read_csv(file, index_col=0, encoding = 'big5')

    if section == 'mainline':
        mainline_id = pd.read_csv("D:/VD_data/vd_static_mainline.csv")
        mainline_id = [mainline_id.iloc[i,0] for i in range(mainline_id.shape[0])]
        data = data[mainline_id]
        vd = data.columns.tolist()
        for i in range(len(vd)):
            vd[i] = vd[i].split('-')[3]
        data.columns = vd
        return data
    elif section == 'ramp':
        ramp_id = pd.read_csv("D:/VD_data/vd_static_ramp.csv", encoding = 'big5')
        ramp_name = [ramp_id.iloc[i,1] for i in range(ramp_id.shape[0])]
        ramp_id = [ramp_id.iloc[i,0] for i in range(ramp_id.shape[0])]
        data = data[ramp_id]
        data.columns = ramp_name
        return data
    else:
        return data

def get_index_1(idx):
    if idx.empty:
        return False
    else:
        temp = {}
        for c in idx.index:
            if len(idx.loc[c])!=0:
                temp[c] = [i for i in idx.loc[c]]
        return temp

def get_index_2(data,mask):
    if mask.empty:
        return False
    else:
        temp = {}
        idx, idy = np.where(pd.isnull(mask))
        result = np.column_stack((data.index[idx], data.columns[idy]))
        #print(np.unique(result[:,1]))
        for c in np.unique(result[:,1]):
            temp[c] = [result[i,0] for i in range(result.shape[0]) if result[i,1] == c]
        return temp

def continuous_data(data,speed,flow,occ):#檢查連續相同數據筆數是否超過6筆
    free_flow = get_index_2(data,data.mask((flow == 0) & (speed == 0) & (occ == 0), np.nan))
    temp_dict = {} #輸出之字典
    for c in data.columns:
        temp = [] #存放連續出現同數值之index
        count = 1
        for r in data.index:
            if r != data.index.to_list()[-1]: #非最後一筆資料
                if data.loc[r,c] == data.loc[r+1,c]: #前筆資料等於下一筆資料計數加一
                    count += 1
                else: #若上下筆資料不同則統計連續之序號
                    if count >= 6: #超過6筆相同資料才計入
                        temp = temp + [idx for idx in range(r-count+1,r+1)]
                    count = 1
        temp = list(set(temp) - set(free_flow[c]))#移除自由車流下狀況之異常值
        temp_dict[c] = temp
    return temp_dict

def detect_error_data(speed,flow,occ):    
    error_code={}
    #先對佔有率進行轉換，將速度流量不為0但佔有率為0之資料依據平均車長重新估計
    error_code['303'] = get_index_2(occ,occ.mask((occ == 0) & (flow != 0) & (speed != 0), np.nan))
    for k, v in error_code['303'].items():
        if isinstance(v, dict):
            for kk, vv in v.items():
                for i in vv:
                    occ.loc[i,kk] = 0.6*flow.loc[i,kk]/speed.loc[i,kk]
    error_code['206'] = flow_concervation(flow)
    
    #uper limit speed
    idx = speed.apply(lambda x:np.array((x>200)).nonzero()[0][:].tolist())
    error_code['101'] = get_index_1(idx)
    #lower limit speed
    idx = speed.apply(lambda x:np.array((x<0)).nonzero()[0][:].tolist())
    error_code['102'] = get_index_1(idx)
    #speed=0,flow!=0,occ!=0
    error_code['103'] = get_index_2(speed,speed.mask((speed == 0) & (flow != 0) & (occ != 0), np.nan))
    #speed!=0,flow=0,occ=0
    error_code['104'] = get_index_2(speed,speed.mask((speed != 0) & (flow == 0) & (occ == 0), np.nan))
    
    error_code['105'] = error_code['206']
    
    #uper limit flow
    idx = flow.apply(lambda x:np.array((x>6000)).nonzero()[0][:].tolist())
    error_code['201'] = get_index_1(idx)
    #lower limit flow
    idx = flow.apply(lambda x:np.array((x<0)).nonzero()[0][:].tolist())
    error_code['202'] = get_index_1(idx)
    #speed!=0,flow=0,occ!=0
    error_code['203'] = get_index_2(flow,flow.mask((flow == 0) & (speed != 0) & (occ != 0), np.nan))
    #speed=0,flow!=0,occ=0
    error_code['204'] = get_index_2(flow,flow.mask((flow != 0) & (speed == 0) & (occ == 0), np.nan))
    #檢查連續相同數據筆數是否超過6筆
    error_code['205'] = continuous_data(flow,speed,flow,occ)
    
    #uper limit occ
    idx = occ.apply(lambda x:np.array((x>100)).nonzero()[0][:].tolist())
    error_code['301'] = get_index_1(idx)
    #lowe limit occ
    idx = occ.apply(lambda x:np.array((x<0)).nonzero()[0][:].tolist())
    error_code['302'] = get_index_1(idx)
    #speed!=0,flow!=0,occ=0
    error_code['303'] = get_index_2(occ,occ.mask((occ == 0) & (flow != 0) & (speed != 0), np.nan))
    #speed=0,flow=0,occ!=0
    error_code['304'] = get_index_2(occ,occ.mask((occ != 0) & (flow == 0) & (speed == 0), np.nan))
    #
    error_code['305'] = error_code['206']
    
    for k, v in error_code.items():
        err_count = 0
        err_vd = 0
        if isinstance(v, dict):#檢查是否為字典
            for kk, vv in v.items():
                err_vd += 1
                err_count += len(vv)
        print('Error key : %s\terror vd count: %i\terror data count: %i'%(k,err_vd,err_count))
    return error_code

def flow_concervation(data):#檢驗主線偵測器密度變化
    temp_dict = {} #輸出之字典
    mainline_id = pd.read_csv("D:/VD_data/vd_static_mainline.csv")
    mainline_id = [mainline_id.iloc[i,0] for i in range(mainline_id.shape[0])]
    data = data[mainline_id]
    vd = data.columns.tolist()
    for i in range(len(vd)):
        vd[i] = vd[i].split('-')[3]
    for r in data.index:
        for c in range(len(data.columns.to_list())):
            if c != 0:
                v = data.columns.to_list()[c]
                dis = float(vd[c]) - float(vd[c-1])
                q_var = (data.loc[r,][c] - data.loc[r,][c-1])/60
                if q_var < dis*(-8.82)*2*1.1 or q_var > dis*(8.82)*2*1.1:
                    if v in temp_dict:
                        temp = temp_dict[v]
                    else:
                        temp_dict[v] = []
                        temp = temp_dict[v]
                    temp.append(r)
    return temp_dict

def read_error_code(dictionary,dtype):
    #dict{error code:{vd: time index}} error dictionary 輸入格式
    #dict{vd:time index} 輸出格式
    temp_dict = {}
    error_count = 0
    error_rate = 0.0
    if dtype == 'speed':
        div = 1
    elif dtype == 'flow':
        div = 2
    elif dtype == 'occ':
        div =3
    else:
        return print('error in read_error_code dtype!!')
    for k, v in dictionary.items():
        if int(int(k)/100) == div:
            #print("processing error number: %s"%k)
            #print('\n========================')
            if isinstance(v, dict):
                for kk, vv in v.items():
                    if kk in temp_dict:
                        temp = temp_dict[kk]
                    else:
                        temp_dict[kk] = []
                        temp = temp_dict[kk]
                    temp = temp + vv
                    temp = list(set(temp))
                    temp.sort()
                    temp_dict[kk] = temp
                    error_count = len(temp)
                    error_rate = error_count/934.70
                    #print("processing vd number: %s"%kk)
    print('Unique vd count: %i\t error rate:%.4f %%'%(len(temp_dict),error_rate))
    return temp_dict, error_rate

#處理負值數據
def imputation_data(data,dictionary,dtype):#移動平均方修正錯誤資料
    new_dictionary, rate = read_error_code(dictionary,dtype)
    for k, v in new_dictionary.items():
        #print('vd name: %s'%k)
        for i in v:
            if i >= 6:
                #print('origin value: %i'%data.loc[:,k][i])
                #print('MA: ',np.average(data.loc[:,k][i-6:i]))
                data.loc[i,k] = np.average(data.loc[i-6:i,k])
    return data

def preprocess_data(flow,speed,occ,psg,lag,tr):
    error_code = detect_error_data(speed,flow,occ)
    flow = imputation_data(flow,error_code,'flow')
    psg = imputation_data(psg,error_code,'flow')
    lag = imputation_data(lag,error_code,'flow')
    tr = imputation_data(tr,error_code,'flow')
    speed = imputation_data(speed,error_code,'speed')
    occ = imputation_data(occ,error_code,'occ')
    temp_dict, error_rate1 = read_error_code(error_code,'flow')
    temp_dict, error_rate2 = read_error_code(error_code,'speed')
    temp_dict, error_rate3 = read_error_code(error_code,'occ')
    avg_rate = np.mean([error_rate1,error_rate2,error_rate3])
    return flow,speed,occ,psg,lag,tr,avg_rate

def history(start_date,end_date,dtype):
    #輸入日期格式"YYYY-MM-DD"
    temp = pd.DataFrame()
    output = pd.DataFrame()
    day = []
    for i in pd.date_range(start_date,end_date):
        if i.weekday() == 6:
            s = datetime.strftime(i,'%Y-%m-%d').replace('-','')
            day.append(s)
    for d in day:
        if dtype == 'flow':
            data = read_data("D:/VD_data/%s/%s_flow_vd2.csv"%(d,d))
        if dtype == 'speed':
            data = read_data("D:/VD_data/%s/%s_speed_vd2.csv"%(d,d))
        if dtype == 'occ':
            data = read_data("D:/VD_data/%s/%s_occ_vd2.csv"%(d,d))
        temp = pd.concat([temp, data])
    mean = temp.groupby(level=0).mean()
    std = temp.groupby(level=0).std()
    return mean, std

def time_series_plot(Type,location,speed,flow,occ,start=0,end=287):
    #處理選定時間範圍
    if start != 0:
        start = int(start/100)*60+start%100
    if end != 287:
        end = int(end/100)*60+end%100
    x = flow.index.to_list()[start:end]
    xticks = np.arange(flow.index.min(), flow.index.max(), 12)
    xlabels = range(xticks.size)
    if Type == 'all':
        #fig, (ax1,ax2,ax3) = plt.subplots(3,1,figsize=(30, 10), sharex=True)
        fig, (ax1,ax2,ax3) = plt.subplots(3,1,figsize=(30, 10))
        #plot flow
        y = flow.loc[:,location][start:end]
        ax1.set_title('VD'+str(location), fontsize=24, fontweight ='bold')
        ax1.set_xlabel('Time', fontsize=16, fontweight ='bold')
        ax1.set_ylabel('Flow', fontsize=16, fontweight ='bold')
        ax1.set_ylim([0,3500])
        plt.xticks(xticks, xlabels,fontsize=16)
        ax1.plot(x,y)
        plt.yticks(fontsize=16)
        #plot speed
        y = speed.loc[:,location][start:end]
        ax2.set_xlabel('Time', fontsize=16, fontweight ='bold')
        ax2.set_ylabel('Speed', fontsize=16, fontweight ='bold')
        ax2.set_ylim([0,150])
        plt.xticks(xticks, xlabels,fontsize=16)
        ax2.plot(x,y)
        plt.yticks(fontsize=16)
        #plot occ
        y = occ.loc[:,location][start:end]
        ax3.set_xlabel('Time', fontsize=12, fontweight ='bold')
        ax3.set_ylabel('Occupancy', fontsize=12, fontweight ='bold')
        ax3.set_ylim([0,150])
        plt.xticks(xticks, xlabels,fontsize=16)
        ax3.plot(x,y)
        plt.yticks(fontsize=16)
        plt.show()
        plt.clf()
    elif Type == 'f':
        fig,ax = plt.subplots(1,1,figsize=(30, 10))
        #plot speed
        y = flow.loc[:,location][start:end]
        ax.set_title('VD'+str(location), fontsize=24, fontweight ='bold')
        ax.set_xlabel('Time', fontsize=16, fontweight ='bold')
        ax.set_ylabel('Flow', fontsize=16, fontweight ='bold')
        ax.set_ylim([0,3500])
        plt.xticks(xticks, xlabels)
        ax.plot(x,y,color='tab:blue')
        plt.yticks(fontsize=16)
        
        y = speed.loc[:,location][start:end]
        ax1 = ax.twinx()
        ax1.plot(x,y,color='tab:orange')
        ax1.set_ylabel('Speed', fontsize=12, fontweight ='bold')
        ax1.set_ylim([0,120])
        plt.yticks(fontsize=16)
    elif Type == 's':
        fig,ax = plt.subplots(1,1,figsize=(30, 10))
        #plot speed
        y = speed.loc[:,location][start:end]
        ax.set_title('VD'+str(location), fontsize=24, fontweight ='bold')
        ax.set_xlabel('Time', fontsize=16, fontweight ='bold')
        ax.set_ylabel('Flow', fontsize=16, fontweight ='bold')
        ax.set_ylim([0,3500])
        plt.xticks(xticks, xlabels)
        ax.plot(x,y)
        plt.yticks(fontsize=16)
    elif Type == 'o':
        fig,ax = plt.subplots(3,1,figsize=(30, 10))
        #plot speed
        y = occ.loc[:,location][start:end]
        ax.set_title('VD'+str(location), fontsize=24, fontweight ='bold')
        ax.set_xlabel('Time', fontsize=16, fontweight ='bold')
        ax.set_ylabel('Flow', fontsize=16, fontweight ='bold')
        ax.set_ylim([0,3500])
        plt.xticks(xticks, xlabels)
        ax.plot(x,y)
        plt.yticks(fontsize=16)


#基礎特性構圖
def fundamental_d(Type,location,speed,flow,occ,start=0,end=287):
    #處理選定時間範圍
    if start != 0:
        start = int(start/100)*12+start%100
    if end != 287:
        end = int(end/100)*12+end%100
    plot_flow = flow[(flow != 0) & (speed != 0)]
    plot_speed = speed[(flow != 0) & (speed != 0)]
    plot_occ = occ[(flow != 0) & (speed != 0)]
    data = pd.concat([plot_speed.loc[:,location][start:end],plot_flow.loc[:,location][start:end],plot_occ.loc[:,location][start:end]],axis=1)
    data.columns = ['speed', 'flow','occ']
    
    #speed-flow diagram
    if Type == 'sf':
        x = data.loc[:,'flow']
        y = data.loc[:,'speed']
        label = [str(int(i/60))+':'+str(i%60) for i in data.index.to_list()]
        #set plot elements
        fig, ax = plt.subplots(figsize=(20, 15))
        plt.title('VD '+location+'K Speed-Flow Diagram', fontsize=32, fontweight ='bold')
        plt.xlabel('Flow', fontsize=16, fontweight ='bold')
        plt.ylabel('Speed', fontsize=16, fontweight ='bold')
        ax.set_ylim([0,100])
        ax.set_xlim([0,3000])
        #plot lines
        #ax.plot(x, y, color='gray')
        #plot points
        ax.scatter(x, y, s=20, zorder=2 ,color='g')
        #plot label
        '''for x_pos, y_pos, label in zip(x, y, label):
            ax.annotate(label,             # The label for this point
                        xy=(x_pos, y_pos), # Position of the corresponding point
                        xytext=(7, 0),     # Offset text by 7 points to the right
                        textcoords='offset points', # tell it to use offset points
                        ha='left',         # Horizontally aligned to the left
                        va='center',       # Vertical alignment is centered
                        fontweight ='bold',
                        fontsize=16)'''

def plot_heat(data, title):
    #set plot elements
    fig, ax = plt.subplots(figsize=(20, 15))
    plt.title(title +' Diagram', fontsize=32, fontweight ='bold')
    plt.xlabel('Mileage', fontsize=16, fontweight ='bold')
    plt.ylabel('Time', fontsize=16, fontweight ='bold')
    #create color palette
    colors = ['purple','red','orange','yellow','green']
    cm = matplotlib.colors.ListedColormap(colors)
    #set color threshould
    nm = matplotlib.colors.BoundaryNorm([0,20,40,60,80,150], cm.N)
    
    #plot pcolormesh
    #psm = plt.pcolormesh(data, cmap=plt.cm.gist_rainbow)
    psm = plt.pcolormesh(data, cmap=cm,norm=nm)
    
    '''
    xticks function
    ticks should be position of indexes of the labels
    labels argument takes the list of label values
    rotation takes how the label should be presented in the plot'''
    plt.xticks(ticks = range(0,data.shape[1]), labels = data.columns, rotation=45)
    ticks = np.arange(data.index.min(), data.index.max(), 60)
    labels = range(ticks.size)
    plt.yticks(ticks, labels)
    fig.colorbar(psm, shrink=0.5, aspect=5)


def plot_3d(data,title):
    # Transform it to a long format
    df = data.unstack().reset_index()
    df.columns=["X","Y","Z"]
    # Make the plot
    #figure(num=None, figsize=(8, 6), dpi=80, facecolor='w', edgecolor='k')
    fig = plt.figure(figsize=(20, 15))
    ax = fig.gca(projection='3d')
    ax.plot_trisurf(df['Y'], df['X'], df['Z'], cmap=plt.cm.viridis, linewidth=0.2)
    # to Add a color bar which maps values to colors.
    surf=ax.plot_trisurf(df['Y'], df['X'], df['Z'], cmap=plt.cm.viridis, linewidth=0.2)
    fig.colorbar( surf, shrink=0.5, aspect=5)

    # Rotate it
    ax.view_init(45, 45)

    # Other palette
    ax.plot_trisurf(df['Y'], df['X'], df['Z'], cmap=plt.cm.jet, linewidth=0.01)
    plt.xlabel('Time', fontsize=16, fontweight ='bold')
    plt.ylabel('Mileage', fontsize=16, fontweight ='bold')
    plt.title(title +' Diagram', fontsize=32, fontweight ='bold')


