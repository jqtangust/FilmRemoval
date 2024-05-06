import torch
import logging
import models.polarmodel.polarmodel as polarmodel

logger = logging.getLogger('base')

####################
# define network
####################

def define_G(opt,tb_logger):
    netG = polarmodel.polarmodel(channel = 4,growth_rate = 64, rdb_number = 3, upscale_factor=1)
    return netG
