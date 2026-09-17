# ------------------------------------------------------------------------------
# Copyright (c) Microsoft
# Licensed under the MIT License.
# Written by Tianheng Cheng(tianhengcheng@gmail.com)
# ------------------------------------------------------------------------------

from .animal import Animal

__all__ = ['get_dataset', 'Animal']


def get_dataset(config):

    if config.DATASET.DATASET == 'ANIMAL':
        return Animal
    else:
        raise NotImplemented()
