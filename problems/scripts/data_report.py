import pandas as pd
import pandas_profiling
import numpy as np

from path import Path
'''
A_h_CW_prediction.csv
A_h_CW_detection.csv
A_d_C_detection.csv

B_h_CW_prediction.csv
B_h_CW_detection.csv
B_d_C_detection.csv

C_h_CW_prediction.csv
C_h_CW_detection.csv
C_d_C_detection.csv

A1_h_CW_prediction.csv
A1_h_CW_detection.csv
A1_d_CW_detection.csv

A2_h_CW_prediction.csv
A2_h_CW_detection.csv
A2_d_CW_detection.csv

A3_h_CW_prediction.csv
A3_h_CW_detection.csv
A3_d_CW_detection.csv




'''

def report_to(name='tmp',df=None):
    profile = pandas_profiling.ProfileReport(df)
    profile.to_file("../data_reports/{}.html".format(name))

def main():
    input_file = 'A_h_CW_prediction'
    data_root = Path('/home/roit/datasets/huawei2021/pb-data2')

    df = pd.read_csv(data_root/(input_file+'.csv'))[['SSHF','SLHF','SSR','SWR']]

    profile = pandas_profiling.ProfileReport(df)
    profile.to_file("../data_reports/{}_LITE.html".format(input_file))
if __name__ == '__main__':
    main()



