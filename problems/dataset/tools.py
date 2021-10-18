

import numpy as np
from path import Path
import pandas as pd
import matplotlib.pyplot as plt
from utils.utils import deg2vec


#[Time1,Time2,Place,Tem1,Tem,SH,Hum,Wv,Wd,RF,CF,BLH,Ap,-,-,LWR,SWR,-,SO2,NO2,PM10,PM2.5,O3,CO]


data_root = Path("/home/roit/datasets/huawei2021/pb-data2")

data_post_root = Path("/home/roit/datasets/huawei2021/pb-data2/post")


ctmnt_names = ['SO2','NO2','PM10','PM2.5','O3','CO']
post_weather_names = ['Tem','Hum','Ap','Wv','Wx','Wy']
prediction_names = ['Tem','SH','Hum','Wv','Wd','RF','CF','BLH','Ap','LWR','SWR','SO2','NO2','PM10','PM2.5','O3','CO']
reports = Path('../data_reports')


stems = [
'A_h_CW_prediction',#0 监测点A 污染物 气象数据 预报
'A_h_CW_detection', #1
'A_d_C_detection',  #2

'B_h_CW_prediction',
'B_h_CW_detection',
'B_d_C_detection',

'C_h_CW_prediction',
'C_h_CW_detection',
'C_d_C_detection',

'A1_h_CW_prediction',
'A1_h_CW_detection',
'A1_d_CW_detection',

'A2_h_CW_prediction',
'A2_h_CW_detection',
'A2_d_CW_detection',

'A3_h_CW_prediction',
'A3_h_CW_detection',
'A3_d_CW_detection'
]

data_dtype = {
    # 'Time': datetime,
    'SO2':np.float,
    'CO':np.float,
    'NO2':np.float,
    'PM2.5':np.float,
    'PM10': np.float,
    'O3': np.float,
    'Hum': np.float,
    'Tem':np.float,
    'Ap':np.float,
    'Wv':np.float,
    'Wd':np.float

         }


def data_load(stem):
    assert stem in stems or stem[:-5] in stems
    if (data_post_root/'{}.csv'.format(stem)).exists():
        df = pd.read_csv(data_post_root / '{}.csv'.format(stem), dtype=data_dtype)
    else:
        df = pd.read_csv(data_root/'{}.csv'.format(stem),dtype=data_dtype)
    return df





def norm(df_ret):
    for item in df_ret.columns:
        if item == 'Time':
            continue
        mean = df_ret[item].mean()
        std =  df_ret[item].std()
        df_ret[item] = (df_ret[item]-mean)/std
    return df_ret

def data_prep(df):

    #线性插值空缺点 1
    df_ret = df.interpolate(method='cubic',axis=0)
    df_ret = df_ret.loc[range(len(df_ret) - 1)]

    # # 离群点处理 2
    # for name in ctmnt_names:
    #     q1, q3 = df_ret[name].quantile([0.25, 0.75])
    #
    #     iqr = q3 - q1
    #
    #     ant_out_lier_mask = (df_ret[name] > q3 + iqr * 1.5) | (df_ret[name] < q1 + iqr * 1.5)
    #    df_ret =df.str.replace('—','nan')

    #     df_ret[name][~ant_out_lier_mask] =np.nan

    df_ret = df_ret.interpolate(method='cubic',axis=0)

    # 风向处理 3
    if 'Wd' in df_ret.columns:
        drec_x,drec_y = deg2vec(df_ret['Wd'])
        df_ret['Wx'] = drec_x
        df_ret['Wy'] = drec_y
        del df_ret['Wd']

    # 归一化处理
    # df_ret = norm(df_ret)
    #

    print(np.where(np.isnan(np.array(df_ret[ctmnt_names]))))
    return df_ret

def main():

    df = pd.read_csv(data_root/'B_h_CW_prediction.csv')
    df =df.replace('—','nan')
    df = df.astype(dtype=dtype)


    df = data_prep(df)

    df.to_csv('./B_h_CW_prediction.csv',index=0)

    print('ok')
if __name__ == '__main__':
    main()

