# ------------------------------------------------------------------------------
# Copyright (c) Microsoft
# Licensed under the MIT License.
# Created by Tianheng Cheng(tianhengcheng@gmail.com)
# ------------------------------------------------------------------------------

import os
import pprint
import argparse

import pandas as pd
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
import lib.models as models
from lib.config import config, update_config
from lib.utils import utils
from lib.datasets import get_dataset
from lib.core import function


def parse_args(cfg=None,model_file=None):

    parser = argparse.ArgumentParser(description='Train Face Alignment')
    if cfg == None:
        parser.add_argument('--cfg', help='experiment configuration filename',
                            required=True, type=str)
    else:
        parser.add_argument('--cfg',default=cfg,type=str)
    if model_file == None:
        parser.add_argument('--model-file', help='model parameters', required=True, type=str)
    else:
        parser.add_argument('--model-file',default=model_file,type=str)

    args = parser.parse_args()
    update_config(config, args)
    return args


def alignment(cfg=None,model_file=None,data_path=None):

    args = parse_args(cfg,model_file)

    cudnn.benchmark = config.CUDNN.BENCHMARK
    cudnn.determinstic = config.CUDNN.DETERMINISTIC
    cudnn.enabled = config.CUDNN.ENABLED

    config.defrost()
    config.MODEL.INIT_WEIGHTS = False
    if data_path != None:

        config.DATASET.ROOT = os.path.join(config.DATASET.ROOT,data_path)
    config.freeze()
    model = models.get_face_alignment_net(config)

    gpus = list(config.GPUS)
    model = torch.nn.DataParallel(model, device_ids=gpus).cuda()

    # load model
    state_dict = torch.load(args.model_file)
    try:
        if 'state_dict' in state_dict.keys():
            model = state_dict['state_dict']
            model.load_state_dict(state_dict)
        else:
            model.module.load_state_dict(state_dict)
    except:
        model = state_dict
        model = torch.nn.DataParallel(model, device_ids=gpus).cuda()


    dataset_type = get_dataset(config)

    test_loader = DataLoader(
        dataset=dataset_type(config,
                             is_train=False,is_test=True),
        batch_size=config.TEST.BATCH_SIZE_PER_GPU*len(gpus),
        shuffle=False,
        num_workers=config.WORKERS,
        pin_memory=config.PIN_MEMORY
    )

    nme, predictions = function.inference(config, test_loader, model)
    #torch.save(predictions, os.path.join(final_output_dir, 'predictions.pth')
    predictions = predictions.numpy()
    predictions = predictions.reshape(predictions.shape[0], (predictions.shape[1] * predictions.shape[2]))

    # df = pd.DataFrame(predictions)
    # df.to_csv('./save.csv')
    return predictions

if __name__ == '__main__':
    alignment()

