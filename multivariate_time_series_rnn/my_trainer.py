# Copyright Niantic 2019. Patent Pending. All rights reserved.
#
# This software is licensed under the terms of the Monodepth2 licence
# which allows for non-commercial use only, the full terms of which are made
# available in the LICENSE file.

from __future__ import absolute_import, division, print_function
import time
import datetime
from path import Path
from tensorboardX import SummaryWriter
import matplotlib.pyplot as plt
import random
from utils.img_process import tensor2array

import networks
import numpy as np
import os
from utils.logger import TermLogger
from utils.assist import model_init,dataset_init
import torch


seed = 127
#random seed
np.random.seed(seed)


def set_mode(models, mode):
    """Convert all models to training mode
    """
    for m in models.values():
        if mode == 'train':
            m.train()
        else:
            m.eval()

torch.autograd.set_detect_anomaly(True)
def L1_loss(gt,pred):
    abs_diff = torch.abs(gt - pred)
    return abs_diff

def L2_loss(pred,depth_gt):
    mask = depth_gt > depth_gt.min()
    mask *= depth_gt < depth_gt.max()
    abs_sq = (depth_gt - pred)**2
    l2_loss = abs_sq[mask]  # [b,1,h,w]
    return l2_loss


class Trainer:
    def __init__(self, options,settings):
        torch.autograd.set_detect_anomaly(True)


        #self.opt = options
        self.metrics = {"abs_rel": 0.0,
                        "sq_rel": 0.0,
                        "rmse": 0.0,
                        "rmse_log": 0.0,
                        "acc": 0.0
                        }
        self.start_time = datetime.datetime.now().strftime("%m-%d-%H:%M")
        self.checkpoints_path = Path(options['log_dir'])/self.start_time




        self.tb_log_frequency = options['tb_log_frequency']
        self.weights_save_frequency = options['weights_save_frequency']
        #save model and events


        #args assert
        self.device = torch.device(options['device'])
        self.models, self.model_optimizer,self.model_lr_scheduler = model_init(options)
        self.train_loader, self.val_loader ,self.stat_dict = dataset_init(options)

        self.val_iter = iter(self.val_loader)

        self.dataset_type = options['dataset']['type']

        #
        self.logger = TermLogger(n_epochs=options['epoch'],
                                 train_size=len(self.train_loader),
                                 valid_size=len(self.val_loader))
        self.logger.reset_epoch_bar()

        self.writers = {}
        for mode in ["train", "val"]:
            self.writers[mode] = SummaryWriter(self.checkpoints_path/mode)


        print("traing files are saved to: ", options['log_dir'])
        print("Training is using: ", self.device)
        print("start time: ",self.start_time)
        os.system('cp {} {}/train_settings.yaml'.format(settings, self.checkpoints_path))

        #self.save_opts()

        #custom

        num_layers = options['model']['num_layers']
        hidden_size = options['model']['hidden_size']
        batch_size = options['batch_size']
        self.hn = torch.zeros(num_layers,batch_size,hidden_size).to(self.device)
        self.cn = torch.zeros(num_layers,batch_size,hidden_size).to(self.device)

        self.loss_func = torch.nn.SmoothL1Loss()
        self.framework = options['model']['framework']


    def compute_metrics(self, inputs, outputs):
        metrics={}
        pred = outputs['pred']
        gt = inputs['gts']#[:, -1, :]

        # gt = gt * self.stat_dict['Y_stds'] + self.stat_dict['Y_means']
        # pred = pred* self.stat_dict['Y_stds'] + self.stat_dict['Y_means']

        # gt = gt * (self.stat_dict['Ymax']-self.stat_dict['Ymin']) + self.stat_dict['Ymin']
        # pred = pred * (self.stat_dict['Ymax']-self.stat_dict['Ymin']) + self.stat_dict['Ymin']

        metrics['abs_rel'] = (torch.abs(gt - pred)/(torch.abs(gt)+1e-4)).median()#[0][2]
        # metrics['rmse_log'] = torch.log(torch.abs(gt - pred)).mean()

        return metrics

    def batch_process(self, inputs):

        for key, ipt in inputs.items():
            inputs[key] = ipt.to(self.device)
        outputs={}
        losses={}

        #depth pass
        seqs = inputs['seqs']
        gts = inputs['gts']
        # if torch.sum(seqs!=seqs)!=0 or torch.sum(gts!=gts)!=0 :
        #     print('er----------------------------------r')
        # outputs['pred'] = self.models['en-de'](seqs)

        if self.framework =='lstm':

            outputs['pred'],(self.hn,self.cn )= self.models["lstm"](seqs,self.hn,self.cn)
            self.hn = self.hn.detach()
            self.cn= self.cn.detach()

        #-----------
        elif self.framework=='rnn':
            outputs['pred'], self.hn = self.models["rnn"](seqs, self.hn)  # 喂入模型得到输出
            self.hn = self.hn.detach()


        pred = outputs['pred']
        gt = inputs['gts']#[:, -1, :]

        # losses['loss'] = torch.abs(1/gt - pred).mean()
# /
        losses['loss'] = self.loss_func(gt , pred )




        return outputs, losses





    def tb_log(self,mode, metrics, inputs=None, outputs=None, losses=None):


        log_loss = ['loss']
        log_metrics=[
            "abs_rel",
            #"sq_rel",
            # "rmse",
            "rmse_log",
            # "a1",
            # "a2",
            # "a3"
        ]




        writer = self.writers[mode]
        if losses!=None:
            for k, v in losses.items():
                if k in log_loss:
                    writer.add_scalar("{}".format(k), float(v), self.step)
        if metrics!=None and   "gts" in inputs.keys():
            for k,v in metrics.items():
                if k in log_metrics:
                    if k in ['a1','a2','a3']:
                        writer.add_scalar("acc/{}".format(k), v, self.step)
                    elif k in ['abs_rel','sq_rel','rmse','rmse_log']:
                        writer.add_scalar("err/{}".format(k), v, self.step)





    #main cycle
    def epoch_process(self):

        set_mode(self.models,'train')

        for batch_idx, inputs in enumerate(self.train_loader):

            if torch.tensor(float('nan')) in inputs['seqs']:
                print('error-->'+self.step)
                continue
            before_op_time = time.time()
            seqs = inputs['seqs']
            gts = inputs['gts']
            # if torch.sum(seqs != seqs) != 0 or torch.sum(gts != gts) :
            #     self.step-=1
            #     print('errrrr')
            #     continue

            #model forwardpass
            # try:

            outputs, losses = self.batch_process(inputs)#

            # except:
            #     print('->batch process error')
            #     continue


            self.model_optimizer.zero_grad()

            losses["loss"].backward()
            self.model_optimizer.step()

            duration = time.time() - before_op_time

            # log less frequently after the first 2000 steps to save time & disk space

            #
            self.logger.train_logger_update(batch= batch_idx,time = duration,dict=losses)

            #val, and terminal_val_log, and tb_log
            if "gts" in inputs:
                # train_set validate
                metrics = self.compute_metrics(inputs, outputs)
                self.metrics.update(metrics)

            phase0 = batch_idx % self.tb_log_frequency == 0 and self.step < 1000
            phase1 = self.step % 100 == 0 and self.step > 1000 and self.step < 10000
            phase2 = self.step % 1000 == 0 and self.step > 10000 and self.step < 100000

            if phase0 or phase1 or phase2:


                self.tb_log(mode='train',
                            metrics=self.metrics,
                            inputs=inputs,
                            outputs=outputs,
                            losses=losses
                            )
            else:
                self.tb_log(mode='train',
                            metrics=self.metrics,
                            inputs=inputs,
                            outputs=outputs,
                            losses=losses
                            )


                self.val()


            self.step += 1

        self.model_lr_scheduler.step()
        self.logger.reset_train_bar()
        self.logger.reset_valid_bar()

            #record the metric

    #only 2 methods for public call
    def __call__(self,opts):

        """Run the entire training pipeline
        """




        self.epoch = 0
        self.step = 0
        self.start_time = time.time()

        self.logger.epoch_logger_update(epoch=0,
                                        time=0,
                                        metrics_dict=self.metrics
                                        )

        for epoch in range(opts['epoch']):
            epc_st = time.time()
            self.epoch_process()
            duration = time.time() - epc_st

            try:
                self.logger.epoch_logger_update(epoch=epoch+1,
                                            time=duration,
                                            metrics_dict=self.metrics
                                            )
            except:
                print("epoch process error")
                continue
            if (epoch + 1) % opts['weights_save_frequency'] == 0 and epoch >=opts['model_first_save']:


                save_folder = self.checkpoints_path / "models" / "weights_{}".format(epoch)
                save_folder.makedirs_p()

                for model_name, model in self.models.items():
                    save_path = save_folder / "{}.pth".format(model_name)
                    to_save = model.state_dict()
                    # if model_name == 'encoder':
                    #     # save the sizes - these are needed at prediction time
                    #     to_save['height'] = input_size['height']
                    #     to_save['width'] = input_size['width']
                    torch.save(to_save, save_path)
                #optimizer
                save_path = self.checkpoints_path/'models'/ "{}.pth".format("adam")
                torch.save(self.model_optimizer.state_dict(), save_path)



    @torch.no_grad()
    def val(self):

        set_mode(self.models,'eval')

        try:
            inputs = self.val_iter.next()
        except StopIteration:
            self.val_iter = iter(self.val_loader)
            inputs = self.val_iter.next()

        val_batch_idx = self.val_iter._rcvd_idx
        time_st = time.time()
        outputs, losses = self.batch_process(inputs)



        duration =time.time() -  time_st
        self.logger.valid_logger_update(batch=val_batch_idx,
                                        time=duration,
                                        dict=losses
                                        )
        if "gts" in inputs.keys():
            self.metrics.update(self.compute_metrics(inputs, outputs))
        else:
            self.metrics=None


        phase0 = val_batch_idx % self.tb_log_frequency == 0 and self.step < 1000
        phase1 = self.step % 100 == 0 and self.step > 1000 and self.step < 10000
        phase2 = self.step % 1000 == 0 and self.step > 10000 and self.step < 100000

        if phase0 or phase1 or phase2:

            self.tb_log(mode='val',
                        metrics=self.metrics,
                        inputs=inputs,
                        outputs=outputs,
                        losses=losses
                        )
        else:
            self.tb_log(mode='val',
                        metrics=self.metrics,
                        inputs=inputs,
                        outputs=outputs,
                        losses=losses
                        )




        del inputs, outputs, losses
        set_mode(self.models,'train')


