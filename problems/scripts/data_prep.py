
from dataset import *

def main():

    df = pd.read_csv(data_root/'C_h_CW_detection.csv')
    df =df.replace('—','nan')
    df = df.astype(dtype=data_dtype)


    df = data_prep(df)

    df.to_csv(reports/'C_h_CW_detection.csv',index=0)

    print('ok')
if __name__ == '__main__':
    main()