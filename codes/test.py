import os
import os.path as osp
import logging
import time
import argparse
from collections import OrderedDict

import options.options as option
import utils.util as util
from data import create_dataset, create_dataloader
from models import create_model

import cv2
import numpy as np
import matplotlib.pyplot as plt

#### options
parser = argparse.ArgumentParser()
parser.add_argument('-opt', type=str, required=False, help='Path to options YMAL file.',
                    default="/remote-home/share/jiaqi2/FilmRemoval/codes/options/test/test.yml")
opt = option.parse(parser.parse_args().opt, is_train=False)
opt = option.dict_to_nonedict(opt)

util.mkdirs(
    (path for key, path in opt['path'].items()
     if not key == 'experiments_root' and 'pretrain_model' not in key and 'resume' not in key))
util.setup_logger('base', opt['path']['log'], 'test_' + opt['name'], level=logging.INFO,
                  screen=True, tofile=True)
logger = logging.getLogger('base')
logger.info(option.dict2str(opt))

flag_num = 1

#### Create test dataset and dataloader
test_loaders = []
for phase, dataset_opt in sorted(opt['datasets'].items()):
    test_set = create_dataset(dataset_opt)
    test_loader = create_dataloader(test_set, dataset_opt)
    logger.info('Number of test images in [{:s}]: {:d}'.format(
        dataset_opt['name'], len(test_set)))
    test_loaders.append(test_loader)

model = create_model(opt, 0)
for test_loader in test_loaders:
    #test_set_name = test_loader.dataset.opt['name']
    #logger.info('\nTesting [{:s}]...'.format(test_set_name))
    test_start_time = time.time()
    dataset_dir = osp.join(opt['path']['results_root'])
    print(dataset_dir)
    util.mkdir(dataset_dir)
    restore_path = osp.join(dataset_dir, "restore")
    util.mkdir(restore_path)
    util.mkdir(osp.join(dataset_dir, "prior"))

    test_results = OrderedDict()
    test_results['psnr'] = []

    for data in test_loader:
        need_GT = False if test_loader.dataset.opt['dataroot_GT'] is None else True
        model.feed_data(data, need_GT=need_GT)
        img_path = data['GT_path'][0] if need_GT else data['LQ_path'][0]
        img_name = osp.splitext(osp.basename(img_path))[0]

        model.test_eval()
        visuals = model.get_current_visuals(need_GT=need_GT)

        sr_img = util.tensor2numpy(visuals['SR'])
        prior = util.tensor2numpy(visuals['Middle'])

        image_path, prior_path = util.generate_paths(dataset_dir, img_name, flag_num)

        util.save_img_with_ratio(image_path, sr_img, prior_path, prior)

        logger.info('{:20s}'.format(img_name))
        flag_num += 1
