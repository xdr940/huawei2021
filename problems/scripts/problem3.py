import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
from path import Path
import numpy as np
# sns.set_theme(style="ticks")
from dataset import *
from utils.evaluation import AQI_np2

rnn_pre_a = 'a-p.csv'
rnn_pre_b = 'b-p.csv'
rnn_pre_c = 'c-p.csv'

lstm_pre_a = 'aa-p.csv'
lstm_pre_b = 'bb-p.csv'
lstm_pre_c = 'cc-p.csv'


def bos_plot():
    # Initialize the figure with a logarithmic x axis
    f, ax = plt.subplots(figsize=(7, 6))
    ax.set_xscale("log")

    # Load the example planets dataset
    planets = sns.load_dataset("planets")

    # Plot the orbital period with horizontal boxes
    sns.boxplot(x="distance", y="method", data=planets,
                whis=[0, 100], width=.6, palette="vlag")

    # Add in points to show each observation
    sns.stripplot(x="distance", y="method", data=planets,
                  size=4, color=".3", linewidth=0)

    # Tweak the visual presentation
    ax.xaxis.grid(True)
    ax.set(ylabel="")
    sns.despine(trim=True, left=True)
    plt.show()

def err_dis():
    rnn_train = 'run-ctmnt_10-17-22_20_train-tag-err_abs_rel.csv'
    rnn_val = 'run-ctmnt_10-17-22_20_val-tag-err_abs_rel.csv'

    lstm_train = 'run-ctmnt_10-17-22_36_train-tag-err_abs_rel.csv'
    lstm_val = 'run-ctmnt_10-17-22_36_val-tag-err_abs_rel.csv'

    rnn_train = pd.read_csv(reports/rnn_train)['Value']
    rnn_val = pd.read_csv(reports/rnn_val)['Value']
    lstm_train = pd.read_csv(reports/lstm_train)['Value']
    lstm_val = pd.read_csv(reports/lstm_val)['Value']
    length = len(np.array(rnn_train))
    rnn_train= np.array(rnn_train)
    rth,_=np.histogram(rnn_train,bins=100)

    rnn_val = np.array(rnn_val)[:length]
    rvh, _ = np.histogram(rnn_val, bins=100)

    lstm_train = np.array(lstm_train)[:length]
    lth, _ = np.histogram(lstm_train, bins=100)
    rnn_pre_a = 'a-p.csv'

    lstm_val = np.array(lstm_val)[:length]
    lvh, _ = np.histogram(lstm_val, bins=100)

    plt.title('AbsRel Distribution')

    plt.plot(rth,'c')
    plt.plot(rvh,'r')
    plt.plot(lth,'g')
    plt.plot(lvh,'grey')

    plt.grid()
    plt.xlabel('abs_rel')
    plt.ylabel('sum')
    plt.legend(['rnn-train','rnn-val','lstm-train','lstm-val'])

    plt.show()
    print('ok')

    pass


def add_pre_values_DRAW():



#A ----------------------------
    df = data_load('A_h_CW_detection')

    rnn_pre_a_df = pd.read_csv(reports/rnn_pre_a)
    lstm_pre_a_df = pd.read_csv(reports/lstm_pre_a)
    length = len(df)
    L = len(rnn_pre_a_df)
    x = np.linspace(length,length+L-1,L)


    plt.plot(np.array(df['SO2']), 'r')
    plt.plot(x,np.array(rnn_pre_a_df['SO2']), 'g-*')
    plt.plot(x,np.array(lstm_pre_a_df['SO2']), 'b-.')
    plt.title('SO2/A')
    plt.legend(['Sensor Data','rnn','lstm'])


#b--------------------------------

    # df = data_load('B_h_CW_detection')
    #
    # rnn_pre_b_df = pd.read_csv(reports / rnn_pre_b)
    # lstm_pre_b_df = pd.read_csv(reports / lstm_pre_b)
    # length = len(df)
    # L = len(rnn_pre_b_df)
    # x = np.linspace(length, length + L - 1, L)
    #
    # plt.plot(np.array(df['SO2']), 'r')
    # plt.plot(x, np.array(rnn_pre_b_df['SO2']), 'g-*')
    # plt.plot(x, np.array(lstm_pre_b_df['SO2']), 'b-.')
    # plt.title('SO2/{}'.format("B"))
    #
    # plt.legend(['Sensor Data', 'rnn', 'lstm'])


#c---------------------------------

    # df = data_load('C_h_CW_detection')
    #
    # rnn_pre_c_df = pd.read_csv(reports / rnn_pre_c)
    # lstm_pre_c_df = pd.read_csv(reports / lstm_pre_c)
    # length = len(df)
    # L = len(rnn_pre_c_df)
    # x = np.linspace(length, length + L - 1, L)
    #
    # plt.plot(np.array(df['SO2']), 'r')
    # plt.plot(x, np.array(rnn_pre_c_df['SO2']), 'g-*')
    # plt.plot(x, np.array(lstm_pre_c_df['SO2']), 'b-.')
    # plt.title('SO2/{}'.format("C"))
    # plt.legend(['Sensor Data', 'rnn', 'lstm'])


    plt.show()
    print('ok')

    pass


def AQI_cacu():
    p = reports/rnn_pre_a

    loc_stems = [
        rnn_pre_a,
        rnn_pre_b,
        rnn_pre_c
    ]
    for loc in loc_stems:
        LOC_AQIS = []
        loc_feats=[]
        p = reports/loc
        df = pd.read_csv(p)
        for name in ctmnt_names:
            LOC_AQIS.append( AQI_np2(name,df[name]))
            loc_feats.append(np.array(df[name]))
        LOC_AQIS=np.array(LOC_AQIS)
        loc_feats = np.array(loc_feats)

        loc_feats = loc_feats.reshape([6,3,24])
        loc_feats = loc_feats.mean(axis=2)
        iaqi = np.argmax(LOC_AQIS,axis=0).reshape(3,24) #找行最多出现次数的
        print(LOC_AQIS.reshape([6, 3, 24]).mean(axis=2))#找列最大的
        AQI = LOC_AQIS.reshape([6, 3, 24]).mean(axis=2).max(axis=0)# 取整

        print(iaqi)


if __name__ == '__main__':
    AQI_cacu()