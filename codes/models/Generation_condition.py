import logging
from collections import OrderedDict
from pickletools import uint8

import cv2
import torch
import torch.nn as nn
from torch.nn.parallel import DataParallel, DistributedDataParallel
import models.networks as networks
import models.lr_scheduler as lr_scheduler
from .base_model import BaseModel
from torchvision.utils import make_grid
import numpy as np

logger = logging.getLogger('base')

class GenerationModel(BaseModel):
    def __init__(self, opt,tb_logger):
        super(GenerationModel, self).__init__(opt)

        self.step=1
        self.tb_logger=tb_logger

        if opt['dist']:
            self.rank = torch.distributed.get_rank()
        else:
            self.rank = -1  # non dist training
        train_opt = opt['train']

        # define network and load pretrained models
        self.netG = networks.define_G(opt,tb_logger).to(self.device)

        if opt['dist']:
            self.netG = DistributedDataParallel(self.netG, device_ids=[torch.cuda.current_device()])
        else:
            self.netG = DataParallel(self.netG)
        # print network
        # self.print_network()
        self.middlemap = 0
        self.right_0degee = 0
        self.right_90degee = 0
        self.Max = 0
        self.Min = 0
        self.angle =0
        
        self.load()
        
        if self.is_train:
            self.netG.train()

            # loss
            loss_type = train_opt['pixel_criterion']
            if loss_type == 'l1':
                self.cri_pix = nn.L1Loss().to(self.device)
            elif loss_type == 'l2':
                self.cri_pix = nn.MSELoss().to(self.device)
            else:
                raise NotImplementedError('Loss type [{:s}] is not recognized.'.format(loss_type))
            self.l_pix_w = train_opt['pixel_weight']

            # optimizers
            wd_G = train_opt['weight_decay_G'] if train_opt['weight_decay_G'] else 0
            optim_params = []
            for k, v in self.netG.named_parameters():  # can optimize for a part of the model
                if v.requires_grad:
                    optim_params.append(v)
                else:
                    if self.rank <= 0:
                        logger.warning('Params [{:s}] will not optimize.'.format(k))
            self.optimizer_G = torch.optim.Adam(optim_params, lr=train_opt['lr_G'],
                                                weight_decay=wd_G,
                                                betas=(train_opt['beta1'], train_opt['beta2']))
            self.optimizers.append(self.optimizer_G)

            # schedulers
            if train_opt['lr_scheme'] == 'MultiStepLR':
                for optimizer in self.optimizers:
                    self.schedulers.append(
                        lr_scheduler.MultiStepLR_Restart(optimizer, train_opt['lr_steps'],
                                                         restarts=train_opt['restarts'],
                                                         weights=train_opt['restart_weights'],
                                                         gamma=train_opt['lr_gamma'],
                                                         clear_state=train_opt['clear_state']))
            elif train_opt['lr_scheme'] == 'CosineAnnealingLR_Restart':
                for optimizer in self.optimizers:
                    self.schedulers.append(
                        lr_scheduler.CosineAnnealingLR_Restart(
                            optimizer, train_opt['T_period'], eta_min=train_opt['eta_min'],
                            restarts=train_opt['restarts'], weights=train_opt['restart_weights']))
            else:
                raise NotImplementedError('MultiStepLR learning rate scheme is enough.')

            self.log_dict = OrderedDict()

    def feed_data(self, data, need_GT=True):
        self.var_L = data['LQ'].to(self.device)  # LQ

        if need_GT:
            self.real_H = data['GT'].to(self.device)  # GT

    def optimize_parameters(self, step):
        self.optimizer_G.zero_grad()
        self.Recon_I,self.middlemap,_,_,_,_,_ = self.netG(self.var_L)
        Loss = self.l_pix_w * self.cri_pix(self.Recon_I, self.real_H)
        Loss.backward()
        self.optimizer_G.step()
        # set log
        self.log_dict['Loss'] = Loss.item()

    def test(self):
        self.netG.eval()
        with torch.no_grad():
            self.Recon_I,self.middlemap,self.right_0degee,self.right_90degee,self.Max,self.Min, self.angle = self.netG(self.var_L)
            self.tb_logger.add_image('right_0degee',(self.right_0degee.detach().cpu().numpy()[0]*255).astype(np.uint8), self.step, dataformats="CHW")
            self.tb_logger.add_image('right_90degee',(self.right_90degee.detach().cpu().numpy()[0]*255).astype(np.uint8), self.step, dataformats="CHW")

            self.tb_logger.add_image('Recon_I',(self.Recon_I.detach().cpu().numpy()[0]*255).astype(np.uint8), self.step, dataformats="HW")
            self.tb_logger.add_image('middlemap',(self.middlemap.detach().cpu().numpy()[0]*255).astype(np.uint8), self.step, dataformats="HW")
            self.step+=1
            # for i in range(len(self.middlemap)):
            #     print(self.middlemap[i].shape)
            #     self.middlemap[i]=self.middlemap[i].cpu()
            #     self.middlemap[i]=make_grid(self.middlemap[i])   #concat the images 
            #     print(self.middlemap[i].shape)

            #     self.tb_logger.add_image('step'+str(i),self.middlemap[i])  #add image to tensorboard 

        self.netG.train()
        
    def test_eval(self):
        self.netG.eval()
        with torch.no_grad():
            self.Recon_I,self.middlemap,self.right_0degee,self.right_90degee,self.Max,self.Min, self.angle = self.netG(self.var_L)
            # print(self.middlemap.shape)
            # cv2.imwrite("middle.png", self.middlemap.detach().cpu().numpy()[0])
            # self.tb_logger.add_image('right_0degee',(self.right_0degee.detach().cpu().numpy()[0]*255).astype(np.uint8), self.step, dataformats="CHW")
            # self.tb_logger.add_image('right_90degee',(self.right_90degee.detach().cpu().numpy()[0]*255).astype(np.uint8), self.step, dataformats="CHW")
            # self.tb_logger.add_image('fake_H',(self.fake_H.detach().cpu().numpy()[0]*255).astype(np.uint8), self.step, dataformats="HW")
            # self.tb_logger.add_image('middlemap',(self.middlemap.detach().cpu().numpy()[0]*255).astype(np.uint8), self.step, dataformats="HW")
            self.step+=1

        self.netG.train()
        
    def get_current_log(self):
        return self.log_dict

    def get_current_visuals(self, need_GT=True):
        out_dict = OrderedDict()
        out_dict['LQ'] = self.var_L.detach()[0].float().cpu()
        out_dict['SR'] = self.Recon_I.detach()[0].float().cpu()
        out_dict['Middle'] = self.middlemap.detach()[0].float().cpu()
        out_dict['Angle'] = self.angle.detach()[0].float().cpu()
        out_dict['right_0degee'] = self.right_0degee.detach()[0].float().cpu()
        out_dict['right_90degee'] = self.right_90degee.detach()[0].float().cpu()
        out_dict['Max'] = self.Max.detach().float().cpu()
        out_dict['Min'] = self.Min.detach().float().cpu()

        if need_GT:
            out_dict['GT'] = self.real_H.detach()[0].float().cpu()
        return out_dict

    def print_network(self):
        s, n = self.get_network_description(self.netG)
        if isinstance(self.netG, nn.DataParallel) or isinstance(self.netG, DistributedDataParallel):
            net_struc_str = '{} - {}'.format(self.netG.__class__.__name__,
                                             self.netG.module.__class__.__name__)
        else:
            net_struc_str = '{}'.format(self.netG.__class__.__name__)
        if self.rank <= 0:
            logger.info('Network G structure: {}, with parameters: {:,d}'.format(net_struc_str, n))
            logger.info(s)

    def load(self):
        load_path_G = self.opt['path']['pretrain_model_G']
        if load_path_G is not None:
            logger.info('Loading model for G [{:s}] ...'.format(load_path_G))
            self.load_network(load_path_G, self.netG, self.opt['path']['strict_load'])

    def save(self, iter_label):
        self.save_network(self.netG, 'G', iter_label)
