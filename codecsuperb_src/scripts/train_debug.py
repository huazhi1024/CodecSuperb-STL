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
#######################################
from audiotools.data.datasets import AudioDataset
from audiotools.data.datasets import AudioLoader
from audiotools.data.datasets import ConcatDataset
##############################################
from audiotools.ml.decorators import timer
from audiotools.ml.decorators import Tracker
from audiotools.ml.decorators import when
###############################################
from torch.utils.tensorboard import SummaryWriter
###################################################################
import dac  ##当前的dac文件夹

warnings.filterwarnings("ignore", category=UserWarning)
torch.backends.cudnn.benchmark = bool(int(os.getenv("CUDNN_BENCHMARK", 1)))

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
#hasattr(fn, "transform")：检查 fn 对象是否有一个叫做 transform 的属性。
filter_fn = lambda fn: hasattr(fn, "transform") and fn.__qualname__ not in ["BaseTransform","Compose","Choose",]#返回一个bool值
tfm = argbind.bind_module(transforms, "train", "val", filter_fn=filter_fn)#给transforms模块绑定train和val的args参数，并过滤一些函数;
#tfm:<argbind.argbind.bind_module object at 0x7f7070c7d550>
#这个fn到底是谁啊？这两行都相当于函数定义而已啦，就是绑定参数！
#####################################################################


###########################################################
# Loss
filter_fn = lambda fn: hasattr(fn, "forward") and "Loss" in fn.__name__
losses = argbind.bind_module(dac.nn.loss, filter_fn=filter_fn)
#######################################################

def get_infinite_loader(dataloader):
    while True:
        for batch in dataloader:
       
            # print(len(batch['path']))#72,刚好等于databatch的数量
            # print(batch["path"])
            # exit(-1)
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

"""
对应base.yml中的xxx.folders
train/build_dataset.folders:
val/build_dataset.folders:
test/build_dataset.folders:
"""
@argbind.bind("train", "val", "test")
def build_dataset(
    sample_rate: int,
    folders: dict = None,
):
    # Give one loader per key/value of dictionary, where
    # value is a list of folders. Create a dataset for each one.
    # Concatenate the datasets with ConcatDataset, which
    # cycles through them.
    ###################################
    ##这里相当于是把folders中列出的所有数据都包含到一起了
    
    datasets = []
    for _, v in folders.items():
        loader = AudioLoader(sources=v) 
        transform = build_transform()
        dataset = AudioDataset(loader, sample_rate, transform=transform)
        datasets.append(dataset)
    ######################################
    # import pdb;pdb.set_trace()
    dataset = ConcatDataset(datasets) 
    dataset.transform = transform  ##给dataset的transform属性赋值为transform方法
    return dataset  #<audiotools.data.datasets.ConcatDataset object at 0x7f3a180da7c0>


@dataclass
class State:
    generator: DAC
    optimizer_g: AdamW
    scheduler_g: ExponentialLR

    discriminator: Discriminator
    optimizer_d: AdamW
    scheduler_d: ExponentialLR

    stft_loss: losses.MultiScaleSTFTLoss
    mel_loss: losses.MelSpectrogramLoss
    gan_loss: losses.GANLoss
    waveform_loss: losses.L1Loss

    train_data: AudioDataset
    val_data: AudioDataset

    tracker: Tracker  

"""
类似trainer,pipline
"""
@argbind.bind(without_prefix=True)
def load(
    args,
    accel: ml.Accelerator,
    tracker: Tracker,
    save_path: str,
    resume: bool = False,
    tag: str = "latest",
    load_weights: bool = False,
    pretrained_path: str = None,#zhyadd
):
    generator, g_extra = None, {}
    discriminator, d_extra = None, {}
    ##############################################
    ##################################################
    # 如果resume为True的话，这里package=!F=T，仍是True
    if resume:##断点续训,默认情况是F,即不会进行自动resume，除非你提供resume参数
        kwargs = {
            "folder": f"{save_path}/{tag}",
            "map_location": "cpu",
            "package": not load_weights,#默认情况是!F=T，那就是即导入模型结构又导入模型参数
        }
        #使用 tracker.print 打印当前绝对路径和即将加载的模型文件夹路径。
        tracker.print(f"Resuming from {str(Path('.').absolute())}/{kwargs['folder']}")
        ####################################
        if (Path(kwargs["folder"]) / "dac").exists():###生成器导入
            generator, g_extra = DAC.load_from_folder(**kwargs)#若package为True，返回的就是model
        if (Path(kwargs["folder"]) / "discriminator").exists():####判别器导入
            discriminator, d_extra = Discriminator.load_from_folder(**kwargs)
    #####################################################
    ####这里是正式实例化生成器和判别器,并打印模型结构########
    generator = DAC() if generator is None else generator
    discriminator = Discriminator() if discriminator is None else discriminator
    #################
    if pretrained_path and not resume:###只有在提供预训练模型且不属于resume的时候的时候才会走这里
        tracker.print(f"Loading pretrained models from {pretrained_path}")#只有weight.pth
        pretrain_dac_state_dict = torch.load(pretrained_path, "cpu")["state_dict"]
        # new_generator = DAC()
        generator_state_dict=generator.state_dict()
        for name,param in pretrain_dac_state_dict.items():
            # print('load param')
            # exit(-1)
            ##########################################
            if name in generator_state_dict:
                try:
                    generator_state_dict[name].copy_(param)
                    print(f"Loaded {name} from pretrained model")
                except Exception as e:
                    print(f"Cound not load {name} due to:{e}")
            #########################################
        print("Model paramters loaded successfully.")
    #################
    tracker.print(generator)
    tracker.print(discriminator)

    ###类似DDP加速！
    generator = accel.prepare_model(generator)
    discriminator = accel.prepare_model(discriminator)

    ####设置优化器和学习率调度器#######
    with argbind.scope(args, "generator"):
        optimizer_g = AdamW(generator.parameters(), use_zero=accel.use_ddp)
        scheduler_g = ExponentialLR(optimizer_g)
    with argbind.scope(args, "discriminator"):
        optimizer_d = AdamW(discriminator.parameters(), use_zero=accel.use_ddp)
        scheduler_d = ExponentialLR(optimizer_d)
    ##########################################################

    ######如何有初始化模型的话，优化器和学习率调度器则从有初始化的状态开始跑#########
    if "optimizer.pth" in g_extra:
        optimizer_g.load_state_dict(g_extra["optimizer.pth"])
    if "scheduler.pth" in g_extra:
        scheduler_g.load_state_dict(g_extra["scheduler.pth"])
    if "tracker.pth" in g_extra:
        tracker.load_state_dict(g_extra["tracker.pth"])
    ##################################################
    if "optimizer.pth" in d_extra:
        optimizer_d.load_state_dict(d_extra["optimizer.pth"])
    if "scheduler.pth" in d_extra:
        scheduler_d.load_state_dict(d_extra["scheduler.pth"])

    #################################################
    ###加载数据集
    sample_rate = accel.unwrap(generator).sample_rate
    with argbind.scope(args, "train"):
        train_data = build_dataset(sample_rate)
    with argbind.scope(args, "val"):
        val_data = build_dataset(sample_rate)
    ########################

    waveform_loss = losses.L1Loss()
    stft_loss = losses.MultiScaleSTFTLoss()
    mel_loss = losses.MelSpectrogramLoss()
    gan_loss = losses.GANLoss(discriminator)##为何没有量化损失？

    return State(
        generator=generator,
        optimizer_g=optimizer_g,
        scheduler_g=scheduler_g,
        discriminator=discriminator,
        optimizer_d=optimizer_d,
        scheduler_d=scheduler_d,
        waveform_loss=waveform_loss,
        stft_loss=stft_loss,
        mel_loss=mel_loss,
        gan_loss=gan_loss,
        tracker=tracker,
        train_data=train_data,
        val_data=val_data,
    )


@timer()
@torch.no_grad()
def val_loop(batch, state, accel):
    ##验证阶段只有生成器工作-调成eval模式
    state.generator.eval()
    ##准备准备验证数据
    batch = util.prepare_batch(batch, accel.device)
    signal = state.val_data.transform(
        batch["signal"].clone(), **batch["transform_args"]
    )
    ##生成器-生成信号
    out = state.generator(signal.audio_data, signal.sample_rate)
    recons = AudioSignal(out["audio"], signal.sample_rate)
    ####################################################
    return {
        "loss": state.mel_loss(recons, signal),#这两项重了？
        "mel/loss": state.mel_loss(recons, signal),
        "stft/loss": state.stft_loss(recons, signal),
        "waveform/loss": state.waveform_loss(recons, signal),
    }


@timer()
def train_loop(state, batch, accel, lambdas):
    ##训练阶段生成器和判别器都进行训练，调成train()模式
    
    state.generator.train()
    state.discriminator.train()
    output = {}
    ###准备训练数据
    
    batch = util.prepare_batch(batch, accel.device)

    with torch.no_grad():
        signal = state.train_data.transform(
            batch["signal"].clone(), **batch["transform_args"]
        )
    # print('signal',signal.shape)#BCT
    ###########################################
    ###加速训练---判别器
    with accel.autocast():
        out = state.generator(signal.audio_data, signal.sample_rate)
        recons = AudioSignal(out["audio"], signal.sample_rate)
        #zhyadd，因为在state.stft_loss(recons, signal)出现shape不匹配
        # print(signal.shape,recons.shape) #torch.Size([12, 1, 16000]) torch.Size([12, 1, 15992])
        # min_len=min(signal.shape[-1],recons.shape[-1])
        # print(min_len)
        # exit(-1)
        #####################################
        commitment_loss = out["vq/commitment_loss"]
        codebook_loss = out["vq/codebook_loss"]
    #####################################################
    with accel.autocast():
        output["adv/disc_loss"] = state.gan_loss.discriminator_loss(recons, signal)
    ##########################################
    state.optimizer_d.zero_grad()
    accel.backward(output["adv/disc_loss"])
    accel.scaler.unscale_(state.optimizer_d)
    output["other/grad_norm_d"] = torch.nn.utils.clip_grad_norm_(
        state.discriminator.parameters(), 10.0
    )
    accel.step(state.optimizer_d)
    state.scheduler_d.step()

    ######################################################################
    ###加速训练---生成器
    with accel.autocast():
        #print()#recons:torch.Size([12, 1, 6072]);signal:torch.Size([12, 1, 6080])
        output["stft/loss"] = state.stft_loss(recons, signal)###raise-error
        output["mel/loss"] = state.mel_loss(recons, signal)
        output["waveform/loss"] = state.waveform_loss(recons, signal)
        (
            output["adv/gen_loss"],
            output["adv/feat_loss"],
        ) = state.gan_loss.generator_loss(recons, signal)
        output["vq/commitment_loss"] = commitment_loss
        output["vq/codebook_loss"] = codebook_loss
        output["loss"] = sum([v * output[k] for k, v in lambdas.items() if k in output])
    #########################################
    state.optimizer_g.zero_grad()
    accel.backward(output["loss"])
    accel.scaler.unscale_(state.optimizer_g)
    output["other/grad_norm"] = torch.nn.utils.clip_grad_norm_(
        state.generator.parameters(), 1e3
    )
    accel.step(state.optimizer_g)
    state.scheduler_g.step()
    accel.update()
    ##########################################
    output["other/learning_rate"] = state.optimizer_g.param_groups[0]["lr"]
    output["other/batch_size"] = signal.batch_size * accel.world_size
    #################################################
    return {k: v for k, v in sorted(output.items())}


####保存中间模型####
def checkpoint(state, save_iters, save_path):
    metadata = {"logs": state.tracker.history} ##保存的可视化的东西tf.events
    ######################################
    tags = ["latest"]
    state.tracker.print(f"Saving to {str(Path('.').absolute())}")
    ##############################################################
    if state.tracker.is_best("val", "mel/loss"):
        state.tracker.print(f"Best generator so far")
        tags.append("best")
    ##############################################################
    if state.tracker.step in save_iters:
        tags.append(f"{state.tracker.step // 1000}k")
    #################################################################
    for tag in tags:
        #########生成器一定要保存优化器和学习率调度器的参数###############
        ######除此之外，就是tracker和metadata了###########
        generator_extra = {
            "optimizer.pth": state.optimizer_g.state_dict(),
            "scheduler.pth": state.scheduler_g.state_dict(),
            "tracker.pth": state.tracker.state_dict(),
            "metadata.pth": metadata,
        }
        #############################################
        accel.unwrap(state.generator).metadata = metadata
        accel.unwrap(state.generator).save_to_folder(f"{save_path}/{tag}", generator_extra)
        ###############判别器一定要保存优化器和学习率调度器的参数#############
        discriminator_extra = {
            "optimizer.pth": state.optimizer_d.state_dict(),
            "scheduler.pth": state.scheduler_d.state_dict(),
        }
        accel.unwrap(state.discriminator).save_to_folder(f"{save_path}/{tag}", discriminator_extra)


@torch.no_grad()
def save_samples(state, val_idx, writer):
    state.tracker.print("Saving audio samples to TensorBoard")
    state.generator.eval()
    #####取验证数据并预处理#########
    samples = [state.val_data[idx] for idx in val_idx]
    batch = state.val_data.collate(samples)
    batch = util.prepare_batch(batch, accel.device)
    signal = state.train_data.transform(
        batch["signal"].clone(), **batch["transform_args"]
    )
    #############################
    ##重建###
    out = state.generator(signal.audio_data, signal.sample_rate)
    recons = AudioSignal(out["audio"], signal.sample_rate)

    audio_dict = {"recons": recons}
    if state.tracker.step == 0:
        audio_dict["signal"] = signal
    ###写入tensorboard#######
    for k, v in audio_dict.items():
        for nb in range(v.batch_size):#100
            v[nb].cpu().write_audio_to_tb(
                f"{k}/sample_{nb}.wav", writer, state.tracker.step
            )
    #################################################

def validate(state, val_dataloader, accel):
    for batch in val_dataloader:
        output = val_loop(batch, state, accel)
    # Consolidate state dicts if using ZeroRedundancyOptimizer
    if hasattr(state.optimizer_g, "consolidate_state_dict"):
        state.optimizer_g.consolidate_state_dict()
        state.optimizer_d.consolidate_state_dict()
    return output


"""这里才是真正的训练过程"""
@argbind.bind(without_prefix=True)
def train(
    args,
    accel: ml.Accelerator,
    seed: int = 0,
    save_path: str = "ckpt",
    num_iters: int = 250000,
    save_iters: list = [10000, 50000, 100000, 200000],
    sample_freq: int = 10000,
    valid_freq: int = 1000,
    batch_size: int = 12,
    val_batch_size: int = 10,
    num_workers: int = 8,
    val_idx: list = [0, 1, 2, 3, 4, 5, 6, 7],
    lambdas: dict = {
        "mel/loss": 100.0,
        "adv/feat_loss": 2.0,
        "adv/gen_loss": 1.0,
        "vq/commitment_loss": 0.25,
        "vq/codebook_loss": 1.0,
    },
):
    #############################################
    ########这里是if __name__=='__main__"的函数入口##############
    util.seed(seed)
    Path(save_path).mkdir(exist_ok=True, parents=True)##这里是创建模型的保存地方的，也就是--save_path后传的路径
    writer = (SummaryWriter(log_dir=f"{save_path}/logs") if accel.local_rank == 0 else None)#多卡时在主进程写入
    tracker = Tracker(writer=writer, log_file=f"{save_path}/log.txt", rank=accel.local_rank)
    ###########################################################################
    #####实际上是实例化训练器#####
    state = load(args, accel, tracker, save_path)##在实例化训练器的时候就已经执行了很多功能了
    ##加载数据
    
    ###这里是创建数据集和加载对应数据集的dataloader，真的DDP和非DDP进行了两类分支处理##########
    ##这里的dataset是concat之后的数据集，其中dataset在这个训练器中就创建好了
    train_dataloader = accel.prepare_dataloader(
        state.train_data,
        start_idx=state.tracker.step * batch_size,
        num_workers=num_workers,
        batch_size=batch_size,
        collate_fn=state.train_data.collate,
    )
    ###################################################################################

    ##############################################################
    # print(len(train_dataloader))#6250000；如果是vctk/train的话。请问这个值是怎么得到额？
    # print('train_dataloader',train_dataloader)
    train_dataloader = get_infinite_loader(train_dataloader)#似乎只是把它变成可迭代的
    
    # print('train_dataloader2',train_dataloader)
    # train_dataloader <torch.utils.data.dataloader.DataLoader object at 0x7f21600dfa00>
    # train_dataloader2 <generator object get_infinite_loader at 0x7f218adeb430>

    ########################################################
    val_dataloader = accel.prepare_dataloader(
        state.val_data,
        start_idx=0,
        num_workers=num_workers,
        batch_size=val_batch_size,
        collate_fn=state.val_data.collate,
        persistent_workers=True if num_workers > 0 else False,
    )
    ##########################################################

    # Wrap the functions so that they neatly track in TensorBoard + progress bars
    # and only run when specific conditions are met.
    global train_loop, val_loop, validate, save_samples, checkpoint
    train_loop = tracker.log("train", "value", history=False)(
        tracker.track("train", num_iters, completed=state.tracker.step)(train_loop)
    )
    val_loop = tracker.track("val", len(val_dataloader))(val_loop)
    validate = tracker.log("val", "mean")(validate)

    # These functions run only on the 0-rank process
    save_samples = when(lambda: accel.local_rank == 0)(save_samples)
    checkpoint = when(lambda: accel.local_rank == 0)(checkpoint)

    with tracker.live:
        for tracker.step, batch in enumerate(train_dataloader, start=tracker.step):##这里是正式迭代dataloader,且batch是一个字典！里面path是data_batchsize个音频的绝对路径
            train_loop(state, batch, accel, lambdas)
            ##############################################################################
            last_iter = (tracker.step == num_iters - 1 if num_iters is not None else False)
            if tracker.step % sample_freq == 0 or last_iter:
                save_samples(state, val_idx, writer)
            ##################################################################################
            if tracker.step % valid_freq == 0 or last_iter:
                validate(state, val_dataloader, accel)
                checkpoint(state, save_iters, save_path)
                # Reset validation progress bar, print summary since last validation.
                tracker.done("val", f"Iteration {tracker.step}")
            ##############################################################
            if last_iter:
                break



if __name__ == "__main__":
    args = argbind.parse_args()
    args["args.debug"] = int(os.getenv("LOCAL_RANK", 0)) == 0  ###这是给args新增参数
    with argbind.scope(args):
        with Accelerator() as accel:
            if accel.local_rank != 0:
                sys.tracebacklimit = 0
            train(args, accel)
