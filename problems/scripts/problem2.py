from dataset.data_prep import *
from dataset.dataloader import data_load,data_root,data_post_root
from scripts.data_report import report_to
import numpy as np


def Delta(df,delta=1):
    ret_df ={}

    for name in df.columns:
        if name =='Time':
            continue
        elif name in ctmnt_names:
            ret_df[name] = np.array(df[name][delta:]) -np.array( df[name][:-delta])
    ret_df = pd.DataFrame(ret_df)
    return ret_df

def main():
    data_stem = 'A_h_CW_detection'
    data_path = data_root/'{}.csv'.format(data_stem)
    data_path_post = data_post_root/'{}_post.csv'.format(data_stem)


    if data_path_post.exists():
        df_ret = data_load('{}_post'.format(data_stem))

    else:
        df = data_load(data_stem)
        df_ret = data_prep(df)

    weather = df_ret[post_weather_names]
    ctmnt = df_ret[ctmnt_names]

    delta_ctmnt = Delta(ctmnt)

    delta_ctmnt_norm = norm(delta_ctmnt)
    weather.head(len(delta_ctmnt_norm))

    combine_df = pd.concat([delta_ctmnt_norm,weather],axis=1)

    report_to('tmp',combine_df)







    # kmeans_model = KMeans(n_clusters=3,random_state=1).fit(weather)
    # df['kmeans'] = kmeans_model.labels_
    print(df_ret.describe())
    pass

if __name__ == '__main__':
    main()