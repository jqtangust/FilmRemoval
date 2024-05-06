import os
import random
import numpy as np
import cv2
import torch
import torch.utils.data as data
import data.util as util
import os.path as osp

import polanalyser as pa

K_FORD=["K1","K2","K3","K4","K5","K6","K7","K8","K9","K10"]

class LQGT_dataset(data.Dataset):

    def __init__(self, opt):
        super(LQGT_dataset, self).__init__()
        self.opt = opt
        self.data_type = self.opt['data_type']
        self.data_path = self.opt['dataroot']
        self.Test_K_ford = self.opt['Test_K_ford']
        self.paths_LQ, self.paths_GT = [], []

        for k_folder in K_FORD:
            if k_folder != self.Test_K_ford:  # 排除测试集
                self.data_GT_path = os.path.join(self.data_path,k_folder,"GT")
                self.data_LQ_path = os.path.join(self.data_path,k_folder,"input")
                current_sizes_GT,current_GT_data=util.get_image_paths(self.data_type, self.data_GT_path)
                current_sizes_LQ,current_LQ_data=util.get_image_paths(self.data_type, self.data_LQ_path)
                self.paths_GT.extend(current_GT_data)
                self.paths_LQ.extend(current_LQ_data)

        self.folder_ratio = opt['dataroot_ratio']
        
    def __getitem__(self, index):
        GT_path, LQ_path = None, None
        scale = self.opt['scale']
        GT_size = self.opt['GT_size']

        LQ_path = self.paths_LQ[index*4]
        
        I_0_LQ_path = self.paths_LQ[index*4 + 0]
        I_135_LQ_path = self.paths_LQ[index*4 + 1]
        I_45_LQ_path = self.paths_LQ[index*4 + 2]
        I_90_LQ_path = self.paths_LQ[index*4 + 3]

        I_0 = util.read_imgdata(I_0_LQ_path, ratio=255.0)
        I_135 = util.read_imgdata(I_135_LQ_path, ratio=255.0)
        I_45 = util.read_imgdata(I_45_LQ_path, ratio=255.0)
        I_90 = util.read_imgdata(I_90_LQ_path, ratio=255.0)
        
        resize_num_x=int(2448/2.5) #979
        resize_num_y=int(2048/2.5) #819
        I_0 =  cv2.resize(I_0, (resize_num_x, resize_num_y))
        I_135 =  cv2.resize(I_135, (resize_num_x, resize_num_y))
        I_45 =  cv2.resize(I_45, (resize_num_x, resize_num_y))
        I_90 =  cv2.resize(I_90, (resize_num_x, resize_num_y))
        
        I_Polar = [I_0,I_135,I_45,I_90]
        img_LQ = cv2.merge(I_Polar)
        
        GT_path = self.paths_GT[index]

        img_GT = util.read_imgdata(GT_path, ratio=255.0)
        img_GT = cv2.resize(img_GT, (resize_num_x, resize_num_y))

        if self.opt['phase'] == 'train':
            
            H, W, C = img_LQ.shape
            H_gt, W_gt = img_GT.shape
            if H != H_gt:
                print('*******wrong image*******:{}'.format(LQ_path))
            LQ_size = GT_size // scale

        # condition
        if self.opt['condition'] == 'image':
            cond = img_LQ.copy()
        elif self.opt['condition'] == 'gradient':
            cond = util.calculate_gradient(img_LQ)

        H, W, _ = img_LQ.shape
        img_GT = torch.from_numpy(np.ascontiguousarray(img_GT)).float()
        img_LQ = torch.from_numpy(np.ascontiguousarray(np.transpose(img_LQ, (2, 0, 1)))).float()
        cond = torch.from_numpy(np.ascontiguousarray(np.transpose(cond, (2, 0, 1)))).float()

        if LQ_path is None:
            LQ_path = GT_path
        return {'LQ': img_LQ, 'GT': img_GT, 'cond': cond, 'LQ_path': LQ_path, 'GT_path': GT_path}

    def __len__(self):
        return int(len(self.paths_GT))


class LQGT_dataset_Val(data.Dataset):

    def __init__(self, opt):
        super(LQGT_dataset_Val, self).__init__()
        self.opt = opt
        self.data_type = self.opt['data_type']
        self.data_path = self.opt['dataroot']
        self.Test_K_ford = self.opt['Test_K_ford']
        self.paths_LQ, self.paths_GT = [], []

        for k_folder in K_FORD:
            if k_folder == self.Test_K_ford:  # 只包含测试集
                self.data_GT_path = os.path.join(self.data_path,k_folder,"GT")
                self.data_LQ_path = os.path.join(self.data_path,k_folder,"input")
                current_sizes_GT,current_GT_data=util.get_image_paths(self.data_type, self.data_GT_path)
                current_sizes_LQ,current_LQ_data=util.get_image_paths(self.data_type, self.data_LQ_path)
                self.paths_GT.extend(current_GT_data)
                self.paths_LQ.extend(current_LQ_data)

        self.folder_ratio = opt['dataroot_ratio']
        
    def __getitem__(self, index):
        GT_path, LQ_path = None, None
        scale = self.opt['scale']
        GT_size = self.opt['GT_size']

        LQ_path = self.paths_LQ[index*4]
        
        I_0_LQ_path = self.paths_LQ[index*4 + 0]
        I_135_LQ_path = self.paths_LQ[index*4 + 1]
        I_45_LQ_path = self.paths_LQ[index*4 + 2]
        I_90_LQ_path = self.paths_LQ[index*4 + 3]

        I_0 = util.read_imgdata(I_0_LQ_path, ratio=255.0)
        I_135 = util.read_imgdata(I_135_LQ_path, ratio=255.0)
        I_45 = util.read_imgdata(I_45_LQ_path, ratio=255.0)
        I_90 = util.read_imgdata(I_90_LQ_path, ratio=255.0)
        
        resize_num_x=int(2448/2.5) #979
        resize_num_y=int(2048/2.5) #819
        I_0 =  cv2.resize(I_0, (resize_num_x, resize_num_y))
        I_135 =  cv2.resize(I_135, (resize_num_x, resize_num_y))
        I_45 =  cv2.resize(I_45, (resize_num_x, resize_num_y))
        I_90 =  cv2.resize(I_90, (resize_num_x, resize_num_y))
        
        I_Polar = [I_0,I_135,I_45,I_90]
        img_LQ = cv2.merge(I_Polar)
        
        GT_path = self.paths_GT[index]

        img_GT = util.read_imgdata(GT_path, ratio=255.0)
        img_GT = cv2.resize(img_GT, (resize_num_x, resize_num_y))
        

        if self.opt['phase'] == 'train':
            
            H, W, C = img_LQ.shape
            H_gt, W_gt = img_GT.shape
            if H != H_gt:
                print('*******wrong image*******:{}'.format(LQ_path))
            LQ_size = GT_size // scale

        # condition
        if self.opt['condition'] == 'image':
            cond = img_LQ.copy()
        elif self.opt['condition'] == 'gradient':
            cond = util.calculate_gradient(img_LQ)

        H, W, _ = img_LQ.shape
        img_GT = torch.from_numpy(np.ascontiguousarray(img_GT)).float()
        img_LQ = torch.from_numpy(np.ascontiguousarray(np.transpose(img_LQ, (2, 0, 1)))).float()
        cond = torch.from_numpy(np.ascontiguousarray(np.transpose(cond, (2, 0, 1)))).float()

        if LQ_path is None:
            LQ_path = GT_path
        return {'LQ': img_LQ, 'GT': img_GT, 'cond': cond, 'LQ_path': LQ_path, 'GT_path': GT_path}

    def __len__(self):
        return int(len(self.paths_GT))