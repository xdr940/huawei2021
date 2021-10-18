

from path import Path
from random import random
import argparse
import pandas as pd
import random
import numpy as np



def writelines(list,path):
    lenth = len(list)
    with open(path,'w') as f:
        for i in range(lenth):
            if i == lenth-1:
                f.writelines(str(list[i]))
            else:
                f.writelines(str(list[i])+'\n')

def readlines(filename):
    """Read all the lines in a text file and return as a list
    """
    with open(filename, 'r') as f:
        lines = f.read().splitlines()
    return lines



def parse_args():
    parser = argparse.ArgumentParser(
        description='custom dataset split for training ,validation and test')

    parser.add_argument('--dataset_path', type=str,default='/home/roit/datasets/huawei2021/pb-data2/post/A_h_CW_detection_post.csv',help='csv file')
    parser.add_argument("--num",
                        # default=1000,
                        default=None
                        )
    parser.add_argument('--seq_len',default=5)


    parser.add_argument("--proportion",default=[0.8,0.1,0.1],help="train, val, test")
    parser.add_argument("--rand_seed",default=12346)
    parser.add_argument("--out_dir",default='../splits/CTMNT')

    return parser.parse_args()
def main(args):
    '''

    :param args:
    :return:none , output is  a dir includes 3 .txt files
    '''
    [train_,val_,test_] = args.proportion
    out_num = args.num
    if train_+val_+test_-1.>0.01:#delta
        print('erro')
        return



    seq_len = args.seq_len


    out_dir = Path(args.out_dir)
    out_dir.mkdir_p()
    train_txt_p = out_dir/'train.txt'
    val_txt_p = out_dir/'val.txt'
    test_txt_p = out_dir/'test.txt'


    dataset_path = Path(args.dataset_path)

    df = pd.read_csv(dataset_path)
    dataset_length = len(df)
    item_list = list(np.array(np.linspace(seq_len,dataset_length-1,dataset_length-seq_len),dtype=np.int))


    random.seed(args.rand_seed)
    # random.shuffle(item_list)



    length = len(item_list)
    train_bound = int(length * args.proportion[0])
    val_bound = int(length * args.proportion[1]) + train_bound
    test_bound = int(length * args.proportion[2]) + val_bound

    print(" train items:{}\n val items:{}\n test items:{}".format(len(item_list[:train_bound]), len(item_list[train_bound:val_bound]), len(item_list[val_bound:test_bound])))
    writelines(item_list[:train_bound],train_txt_p)
    writelines(item_list[train_bound:val_bound],val_txt_p)
    writelines(item_list[val_bound:test_bound],test_txt_p)













if  __name__ == '__main__':
    options = parse_args()
    main(options)