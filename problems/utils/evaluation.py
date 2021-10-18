
import numpy as np

MAX_LENGTH = 8
IAQI = [0, 50, 100, 150, 200, 300, 400, 500]

O3_8_avg = [0, 100, 160, 215, 265, 800, -1, -1]
SO2_24_avg = [0, 50, 150, 475, 800, 1600, 2100, 2620]
CO_24_avg = [0, 2, 4, 14, 24, 36, 48, 60]
NO2_24_avg = [0, 40, 80, 180, 280, 565, 750, 940]
PM10_24_avg = [0, 50, 150, 250, 350, 420, 500, 600]
PM25_24_avg = [0, 35, 75, 115, 150, 250, 350, 500]

ctmnt_tab = {
    'O3': O3_8_avg,
    'SO2': SO2_24_avg,
    'CO': CO_24_avg,
    'NO2': NO2_24_avg,
    'PM10': PM10_24_avg,
    'PM2.5': PM25_24_avg
}

def AQI(name,ctmnt):
    '''

    :param ctmnt: 污染物, numpy array
    :return:
    '''

    BP = ctmnt_tab[name]
    assert  name in ctmnt_tab.keys()



    index = 0

    for index in range(1,MAX_LENGTH):
        if ctmnt> BP[index-1] and ctmnt <BP[index]:
            break
    Lo = index-1
    Hi = index
    ret = (IAQI[Hi] - IAQI[Lo])/(BP[Hi] - BP[Lo]) * (ctmnt - BP[Lo]) + IAQI[Lo]
    # return ret
    return np.ceil(ret)

def AQI_np2(name,ctmnt_np):

    ret = np.zeros_like(ctmnt_np)


    for idx,item in enumerate(ctmnt_np):
        ret[idx] = AQI(name,item)


    return ret





def AQI_np(name,ctmnt):
    pass
    #TODO


# if __name__ == '__main__':
#     a= np.array([185,12])
#     AQI('O3',a)
#
#     AQI('SO2',12)
#     AQI('NO2',66)
#     AQI('CO',0.8)
#     AQI('O3',210)
#     AQI('PM10',83)
#     AQI('PM2.5',39)