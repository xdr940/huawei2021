


from dataset import *
from utils.evaluation import AQI_np2




def main():
    df = pd.read_csv(data_root/'A_d_C_detection.csv',dtype=dtype)
    df2 = df[497:501]
    AQI_RET={
        'SO2':[],
        'CO':[],
        'O3':[],
        'NO2':[],
        'PM10':[],
        'PM2.5':[]
    }
    for ctmnt_name in AQI_RET.keys():
        AQI_RET[ctmnt_name] = AQI_np2(ctmnt_name,df2[ctmnt_name])
    AQI_RET_df = pd.DataFrame(AQI_RET)
    AQI_RET_df.to_csv("../data_reports/p1.csv")
    print(AQI_RET_df)


if __name__ == '__main__':
    main()