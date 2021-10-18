from __future__ import absolute_import, division, print_function

import cv2
import numpy as np

import torch
from torch.utils.data import DataLoader
from datasets.mc_dataset import relpath_split

from networks.encoders import getEncoder
from networks.depth_decoder import getDepthDecoder

from networks.layers import disp2depth,disp_to_depth
from utils.official import readlines
import datasets
import networks
from tqdm import  tqdm
from path import Path
from utils.yaml_wrapper import YamlHandler
from utils.official import compute_errors
from utils.assist import reframe
import matplotlib.pyplot as plt

cv2.setNumThreads(0)  # This speeds up evaluation 5x on our unix systems (OpenCV 3.3.1)



def batch_post_process_disparity(l_disp, r_disp):
    """Apply the disparity post-processing method as introduced in Monodepthv1
    """
    _, h, w = l_disp.shape
    m_disp = 0.5 * (l_disp + r_disp)
    l, _ = np.meshgrid(np.linspace(0, 1, w), np.linspace(0, 1, h))
    l_mask = (1.0 - np.clip(20 * (l - 0.05), 0, 1))[None, ...]
    r_mask = l_mask[:, :, ::-1]
    return r_mask * l_disp + l_mask * r_disp + (1.0 - l_mask - r_mask) * m_disp


def dict_update(dict):

    keys = list(dict.keys()).copy()
    for key in keys:
        if 'encoder.' in key:
            new_key = key.replace('encoder.','')
            dict[new_key] =  dict.pop(key)

    return dict


def model_init(model_path,mode):
    encoder_path = model_path['encoder']
    decoder_path = model_path['depth']

    #model init
    encoder = getEncoder(model_mode=mode)
    depth_decoder = getDepthDecoder(model_mode='default',mode='test')

    #encoder dict updt
    encoder_dict = torch.load(encoder_path)
    encoder_dict = dict_update(encoder_dict)
    decoder_dict = torch.load(decoder_path)


    #load encoder dict
    model_dict = encoder.state_dict()
    model_dict_ = {k: v for k, v in encoder_dict.items() if k in model_dict}
    encoder.load_state_dict(model_dict_)

    #load decoder dict
    model_dict = depth_decoder.state_dict()
    decoder_dict_ = {k: v for k, v in decoder_dict.items() if k in model_dict}
    depth_decoder.load_state_dict(decoder_dict_)



    encoder.cuda()
    encoder.eval()
    depth_decoder.cuda()
    depth_decoder.eval()
    return encoder,depth_decoder




@torch.no_grad()
def evaluate(opts):
    """Evaluates a pretrained model using a specified test set
    """
    MIN_DEPTH = opts['min_depth']
    MAX_DEPTH = opts['max_depth']

    data_path = opts['dataset']['path']
    batch_size = opts['dataset']['batch_size']

    num_workers = opts['dataset']['num_workers']
    feed_height = opts['feed_height']
    feed_width = opts['feed_width']
    full_width = opts['dataset']['full_width']
    full_height = opts['dataset']['full_height']
    metric_mode = opts['metric_mode']



    #这里的度量信息是强行将gt里的值都压缩到和scanner一样的量程， 这样会让值尽量接近度量值
    #但是对于


    data_path = Path(opts['dataset']['path'])
    lines = Path(opts['dataset']['split']['path'])/opts['dataset']['split']['test_file']
    model_path = opts['model']['load_paths']
    encoder_mode = opts['model']['encoder_mode']
    frame_sides = opts['frame_sides']
    # frame_prior,frame_now,frame_next =  opts['frame_sides']
    encoder,decoder = model_init(model_path,mode=encoder_mode)
    file_names = readlines(lines)

    print('-> dataset_path:{}'.format(data_path))
    print('-> model_path')
    for k,v in opts['model']['load_paths'].items():
        print('\t'+str(v))

    print("-> metrics mode: {}".format(metric_mode))
    print("-> data split:{}".format(lines))
    print('-> total:{}'.format(len(file_names)))

    if opts['dataset']['type']=='mc':
        dataset = datasets.MCDataset(data_path=data_path,
                                       filenames=file_names,
                                       height=feed_height,
                                       width=feed_width,
                                       frame_sides=frame_sides,
                                     num_scales=1,
                                     mode="test")
    elif opts['dataset']['type']=='kitti':

        dataset = datasets.KITTIRAWDataset (  # KITTIRAWData
            data_path = data_path,
            filenames=file_names,
            height=feed_height,
            width=feed_width,
            frame_sides=frame_sides,
            num_scales=1,
            mode="test"
        )

    dataloader = DataLoader(dataset,
                            batch_size=batch_size,
                            shuffle=False,
                            num_workers=num_workers,
                            pin_memory=True,
                            drop_last=False)
    pred_depths=[]
    gt_depths = []
    disps = []
    for data in tqdm(dataloader):


        input_color = reframe(encoder_mode,data,frame_sides=frame_sides,key='color')
        input_color = input_color.cuda()


        features = encoder(input_color)
        disp = decoder(*features)



        depth_gt = data['depth_gt']

        pred_disp, pred_depth = disp_to_depth(disp,min_depth=MIN_DEPTH, max_depth=MAX_DEPTH)
        #pred_depth = disp2depth(disp)

        pred_depth = pred_depth.cpu()[:,0].numpy()
        depth_gt = depth_gt.cpu()[:,0].numpy()

        pred_depths.append(pred_depth)
        gt_depths.append(depth_gt)
    gt_depths = np.concatenate(gt_depths, axis=0)


    pred_depths = np.concatenate(pred_depths,axis=0)








    metrics = []
    ratios=[]

    for gt, pred in zip(gt_depths, pred_depths):
        gt_height, gt_width = gt.shape[:2]
        pred = cv2.resize(pred, (gt_width, gt_height))
        # crop
        # if test_dir.stem == "eigen" or test_dir.stem == 'custom':#???,可能是以前很老的
        if opts['dataset']['type'] == "kitti":  # ???,可能是以前很老的
            mask = np.logical_and(gt > MIN_DEPTH, gt < MAX_DEPTH)
            crop = np.array(
                [0.40810811 * gt_height, 0.99189189 * gt_height, 0.03594771 * gt_width, 0.96405229 * gt_width]).astype(
                np.int32)
            crop_mask = np.zeros(mask.shape)
            crop_mask[crop[0]:crop[1], crop[2]:crop[3]] = 1
            mask = np.logical_and(mask, crop_mask)
        else:
            mask = np.logical_and(gt > MIN_DEPTH, gt < MAX_DEPTH)


        pred = pred[mask]  # 并reshape成1d
        gt = gt[mask]

        ratio = np.median(gt) / np.median(pred)  # 中位数， 在eval的时候， 将pred值线性变化，尽量能使与gt接近即可
        ratios.append(ratio)
        pred *= ratio

        pred[pred < MIN_DEPTH] = MIN_DEPTH  # 所有历史数据中最小的depth, 更新,
        pred[pred > MAX_DEPTH] = MAX_DEPTH  # ...
        try:
            metric = compute_errors(gt, pred,mode=metric_mode)
            metrics.append(metric)
        except:
            print('error')

    metrics = np.array(metrics)
    mean_metrics = np.mean(metrics, axis=0)

    # print("\n  " + ("{:>8} | " * 7).format("abs_rel", "sq_rel", "rmse", "rmse_log", "a1", "a2", "a3"))
    print(("&{: 8.3f}  " * 7).format(*mean_metrics.tolist()) + "\\\\")



    ratios = np.array(ratios)
    median = np.median(ratios)
    print("\n Scaling ratios | med: {:0.3f} | std: {:0.3f}\n".format(median, np.std(ratios / median)))










if __name__ == "__main__":

    opts = YamlHandler('/home/roit/aws/aprojects/DeepSfMLearner/opts/mc_eval.yaml').read_yaml()
    # opts = YamlHandler('/home/roit/aws/aprojects/DeepSfMLearner/opts/mc_eval.yaml').read_yaml()


    evaluate(opts)
