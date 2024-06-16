import os
import sys
import warnings
from dataclasses import dataclass
from pathlib import Path

#########################################################
import argbind
import torch
"""
audiotools全部是与数据处理相关的库
https://github.com/descriptinc/audiotools/blob/master/audiotools/data/datasets.py
"""
from audiotools import AudioSignal
from audiotools import ml
from audiotools.core import util
from audiotools.data import transforms
from audiotools.data.datasets import AudioDataset
from audiotools.data.datasets import AudioLoader
from audiotools.data.datasets import ConcatDataset
from audiotools.ml.decorators import timer
from audiotools.ml.decorators import Tracker
from audiotools.ml.decorators import when
from torch.utils.tensorboard import SummaryWriter
###################################################################
import dac  ##当前的dac文件夹

warnings.filterwarnings("ignore", category=UserWarning)

# Enable cudnn autotuner to speed up training
# (can be altered by the funcs.seed function)
torch.backends.cudnn.benchmark = bool(int(os.getenv("CUDNN_BENCHMARK", 1)))
# Uncomment to trade memory for speed.




# Models
#################################################
DAC = argbind.bind(dac.model.DAC)
Discriminator = argbind.bind(dac.model.Discriminator)
###################################################

########################################
# Optimizers
AdamW = argbind.bind(torch.optim.AdamW, "generator", "discriminator")
Accelerator = argbind.bind(ml.Accelerator, without_prefix=True)
##LR_schedual
@argbind.bind("generator", "discriminator")
def ExponentialLR(optimizer, gamma: float = 1.0):
    return torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma)
############################################################


# Data
############################################
AudioDataset = argbind.bind(AudioDataset, "train", "val")
AudioLoader = argbind.bind(AudioLoader, "train", "val")
#############################################################
# Transforms
filter_fn = lambda fn: hasattr(fn, "transform") and fn.__qualname__ not in [
    "BaseTransform",
    "Compose",
    "Choose",
]
tfm = argbind.bind_module(transforms, "train", "val", filter_fn=filter_fn)
#####################################################################


###########################################################
# Loss
filter_fn = lambda fn: hasattr(fn, "forward") and "Loss" in fn.__name__
losses = argbind.bind_module(dac.nn.loss, filter_fn=filter_fn)
#######################################################


def get_infinite_loader(dataloader):
    while True:
        for batch in dataloader:
            yield batch


@argbind.bind("train", "val")
def build_transform(
    augment_prob: float = 1.0,
    preprocess: list = ["Identity"],
    augment: list = ["Identity"],
    postprocess: list = ["Identity"],
):
    to_tfm = lambda l: [getattr(tfm, x)() for x in l]
    preprocess = transforms.Compose(*to_tfm(preprocess), name="preprocess")
    augment = transforms.Compose(*to_tfm(augment), name="augment", prob=augment_prob)
    postprocess = transforms.Compose(*to_tfm(postprocess), name="postprocess")
    transform = transforms.Compose(preprocess, augment, postprocess)
    return transform



@argbind.bind("train", "val", "test")
def build_dataset(
    sample_rate: int,
    folders: dict = None,
):
    ###################################
    ##这里相当于是把folders中列出的所有数据都包含到一起了
    datasets = []
    for _, v in folders.items():
        # import pdb;pdb.set_trace()
        loader = AudioLoader(sources=v)
        transform = build_transform()
        dataset = AudioDataset(loader, sample_rate, transform=transform)
        datasets.append(dataset)
    ######################################
    dataset = ConcatDataset(datasets) 
    dataset.transform = transform
    return dataset  #<audiotools.data.datasets.ConcatDataset object at 0x7f3a180da7c0>





########################
        
if __name__ == "__main__":
    args = argbind.parse_args()
    args["args.debug"] = int(os.getenv("LOCAL_RANK", 0)) == 0  ###这是给args新增参数
    with argbind.scope(args):
        with Accelerator() as accel:
            if accel.local_rank != 0:
                sys.tracebacklimit = 0
            # train(args, accel)
            #########debug###########################
            ###加载数据集
            sample_rate = 16000
            with argbind.scope(args, "train"):
                train_data = build_dataset(sample_rate)
                
            ########################################