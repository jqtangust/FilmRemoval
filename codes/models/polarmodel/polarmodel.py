from curses import A_ALTCHARSET
from pickletools import uint8
import torch
import torch.nn as nn
import torch.nn.init as init
import math
import cv2
import numpy as np

import polanalyser as pa


class polarmodel(nn.Module):
    def __init__(self, channel, growth_rate, rdb_number, upscale_factor):
        super(polarmodel, self).__init__()
        self.reconstruction_network = ReconstructionNetwork(rdb_number, upscale_factor)
        self.angle_network = AngleNetwork(rdb_number)
        self.preprocessing = Preprocessing()

    def forward(self, x):
        # Preprocess the input image
        img_DoLP, img_AoLP, Max, Min = self.preprocessing(x)

        # Concatenate the input image with DoLP and AoLP
        input_concat = torch.cat([x, img_DoLP, img_AoLP], dim=1)

        # Calculate the angle using the angle network
        angle = self.angle_network(input_concat)

        # Polarization model
        right_0degree = torch.pow(torch.sin(angle), 2)
        right_90degree = torch.pow(torch.cos(angle), 2)

        # Calculate the final image using the polarization components and intensity values
        Middle_P = Max * right_0degree + Min * right_90degree
        Middle_P = Middle_P / 255

        # Concatenate the final image with the input image
        Cat_Map = torch.cat([Middle_P, x], dim=1)

        # Reconstruct the image using the reconstruction network
        final_img = self.reconstruction_network(Cat_Map)

        # Squeeze the final image to remove the channel dimension
        final_img = final_img.squeeze(1)
        Middle_P = Middle_P.squeeze(1)

        return final_img, Middle_P, right_0degree, right_90degree, Max, Min, angle
    
    
class Preprocessing(nn.Module):
    def __init__(self):
        super(Preprocessing, self).__init__()

    def forward(self, x):
        I_0, I_135, I_45, I_90 = torch.split(x, 1, dim=1)
        img_demosaiced = torch.cat([I_0, I_45, I_90, I_135], dim=1).cpu().numpy()
        img_demosaiced = img_demosaiced.squeeze(0)

        angles = np.deg2rad([0, 45, 90, 135])
        img_demosaiced = np.transpose(img_demosaiced, (1, 2, 0))
        img_demosaiced_int = (img_demosaiced * 255).astype(np.uint8)

        img_demosaiced_int_list = [img_demosaiced_int[:, :, 0], img_demosaiced_int[:, :, 1],
                                   img_demosaiced_int[:, :, 2], img_demosaiced_int[:, :, 3]]
        img_stokes = pa.calcStokes(img_demosaiced_int_list, angles)
        img_DoLP = pa.cvtStokesToDoLP(img_stokes)
        img_AoLP = pa.cvtStokesToAoLP(img_stokes)

        img_S0, img_S1, img_S2 = cv2.split(img_stokes)
        Max = (img_S0 + np.sqrt(img_S1 ** 2 + img_S2 ** 2)) * 0.5
        Min = (img_S0 - np.sqrt(img_S1 ** 2 + img_S2 ** 2)) * 0.5

        Max = torch.from_numpy(np.ascontiguousarray(Max)).float()
        Min = torch.from_numpy(np.ascontiguousarray(Min)).float()
        Max = Max.unsqueeze(0)
        Max = Max.unsqueeze(0).cuda()

        Min = Min.unsqueeze(0)
        Min = Min.unsqueeze(0).cuda()

        img_DoLP = torch.from_numpy(np.ascontiguousarray(img_DoLP)).float()
        img_AoLP = torch.from_numpy(np.ascontiguousarray(img_AoLP)).float()

        img_DoLP = img_DoLP.unsqueeze(0)
        img_DoLP = img_DoLP.unsqueeze(0).cuda()

        img_AoLP = img_AoLP.unsqueeze(0)
        img_AoLP = img_AoLP.unsqueeze(0).cuda()

        return img_DoLP, img_AoLP, Max, Min


class ReconstructionNetwork(nn.Module):
    def __init__(self, rdb_number, upscale_factor):
        super(ReconstructionNetwork, self).__init__()
        self.SFF1 = nn.Conv2d(in_channels=5, out_channels=64, kernel_size=3, padding=1, stride=1)
        self.SFF2 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding=1, stride=1)
        self.RDB1 = RDB(nb_layers=rdb_number, input_dim=64, growth_rate=64)
        self.RDB2 = RDB(nb_layers=rdb_number, input_dim=64, growth_rate=64)
        self.RDB3 = RDB(nb_layers=rdb_number, input_dim=64, growth_rate=64)
        self.GFF1 = nn.Conv2d(in_channels=64 * 3, out_channels=64, kernel_size=1, padding=0)
        self.GFF2 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding=1)
        self.upconv = nn.Conv2d(in_channels=64, out_channels=(64 * upscale_factor * upscale_factor),
                                kernel_size=3, padding=1)
        self.pixelshuffle = nn.PixelShuffle(upscale_factor)
        self.conv2 = nn.Conv2d(in_channels=64, out_channels=1, kernel_size=3, padding=1)

    def forward(self, x):
        f_ = self.SFF1(x)
        f_0 = self.SFF2(f_)
        f_1 = self.RDB1(f_0)
        f_2 = self.RDB2(f_1)
        f_3 = self.RDB3(f_2)
        f_D = torch.cat((f_1, f_2, f_3), 1)
        f_1x1 = self.GFF1(f_D)
        f_GF = self.GFF2(f_1x1)
        f_DF = f_GF + f_
        f_upconv = self.upconv(f_DF)
        f_upscale = self.pixelshuffle(f_upconv)
        f_conv2 = self.conv2(f_upscale)
        f_conv2 = f_conv2.squeeze(1)

        return f_conv2


class AngleNetwork(nn.Module):
    def __init__(self, rdb_number):
        super(AngleNetwork, self).__init__()
        self.SFF3 = nn.Conv2d(in_channels=6, out_channels=64, kernel_size=3, padding=1, stride=1)
        self.SFF4 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding=1, stride=1)
        self.RDB_Angle_1 = RDB_Lightweight(nb_layers=rdb_number, input_dim=64, growth_rate=64)
        self.RDB_Angle_2 = RDB_Lightweight(nb_layers=rdb_number, input_dim=64, growth_rate=64)
        self.RDB_Angle_3 = RDB_Lightweight(nb_layers=rdb_number, input_dim=64, growth_rate=64)
        self.GFF3 = nn.Conv2d(in_channels=64 * 3, out_channels=64, kernel_size=1, padding=0)
        self.GFF4 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding=1)
        self.tail = nn.Conv2d(in_channels=64, out_channels=1, kernel_size=3, padding=1)

    def forward(self, x):
        a = self.SFF3(x)
        a = self.SFF4(a)
        a_1 = self.RDB_Angle_1(a)
        a_2 = self.RDB_Angle_2(a_1)
        a_3 = self.RDB_Angle_3(a_2)
        a_D = torch.cat((a_1, a_2, a_3), 1)
        a_1x1 = self.GFF3(a_D)
        a_last = self.GFF4(a_1x1)
        angle = self.tail(a_last)

        return angle


class BasicBlock(nn.Module):
    def __init__(self,input_dim,output_dim):
        super(BasicBlock,self).__init__()
        self.ID = input_dim
        self.conv = nn.Conv2d(in_channels = input_dim,out_channels = output_dim,kernel_size=3,padding=1,stride=1)
        self.relu = nn.ReLU()
    
    def forward(self,x):
        out = self.conv(x)
        out = self.relu(out)
        return torch.cat((x,out),1)


class RDB(nn.Module):
    def __init__(self,nb_layers,input_dim,growth_rate):
        super(RDB,self).__init__()
        self.ID = input_dim
        self.GR = growth_rate
        self.layer = self._make_layer(nb_layers,input_dim,growth_rate)
        self.conv1x1 = nn.Conv2d(in_channels = input_dim+nb_layers*growth_rate,\
                                    out_channels =growth_rate,\
                                    kernel_size = 1,\
                                    stride=1,\
                                    padding=0  )
    def _make_layer(self,nb_layers,input_dim,growth_rate):
        layers = []
        for i in range(nb_layers):
            layers.append(BasicBlock(input_dim+i*growth_rate,growth_rate))
        return nn.Sequential(*layers)

    def forward(self,x):
        out = self.layer(x)
        out = self.conv1x1(out) 
        return out+x

class BasicBlock_Lightweight(nn.Module):
    def __init__(self,input_dim,output_dim):
        super(BasicBlock_Lightweight,self).__init__()
        self.ID = input_dim
        self.conv = nn.Conv2d(in_channels = input_dim, out_channels = output_dim,kernel_size=3,padding=2,stride=1,dilation=2, groups=64)
        self.relu = nn.ReLU()
    
    def forward(self,x):
        out = self.conv(x)
        out = self.relu(out)
        return torch.cat((x,out),1)


class RDB_Lightweight(nn.Module):
    def __init__(self,nb_layers,input_dim,growth_rate):
        super(RDB_Lightweight,self).__init__()
        self.ID = input_dim
        self.GR = growth_rate
        self.layer = self._make_layer(nb_layers,input_dim,growth_rate)
        self.conv1x1 = nn.Conv2d(in_channels = input_dim+nb_layers*growth_rate,\
                                    out_channels =growth_rate,\
                                    kernel_size = 1,\
                                    stride=1,\
                                    padding=0  )
    def _make_layer(self,nb_layers,input_dim,growth_rate):
        layers = []
        for i in range(nb_layers):
            layers.append(BasicBlock_Lightweight(input_dim+i*growth_rate,growth_rate))
        return nn.Sequential(*layers)

    def forward(self,x):
        out = self.layer(x)
        out = self.conv1x1(out) 
        return out+x
