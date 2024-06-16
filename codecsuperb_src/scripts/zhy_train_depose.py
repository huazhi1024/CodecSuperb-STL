"""
假设 bind_module 的主要功能是：

1-绑定某些参数到 transforms 模块的训练和验证过程中。
2-使用 filter_fn 来过滤符合条件的函数或方法。
"""
import argbind
from audiotools.data import transforms
# 定义过滤函数
def filter_fn(fn):
    ##返回的是bool值
    return hasattr(fn, "transform") and fn.__qualname__ not in ["BaseTransform", "Compose", "Choose"]


"""
我们可以定义一个叫 bind_module 的普通函数来模拟 argbind.bind_module 的行为。
这个函数将接受一个模块（例如 transforms）、训练参数、验证参数和过滤函数，
然后执行一些操作（例如参数绑定和过滤）。
"""
# 定义参数绑定函数
def bind_params(method, train_param, val_param):
    # 示例函数，用于绑定参数
    method.train_param = train_param
    method.val_param = val_param

# 定义模块绑定函数
def bind_module(module, train_param, val_param, filter_fn):
    # 过滤模块中的方法
    filtered_methods = {name: method for name, method in vars(module).items() if filter_fn(method)}
    
    # 绑定参数
    for name, method in filtered_methods.items():
        bind_params(method, train_param, val_param)
    
    return filtered_methods
#######################################################################


#########################(2)###############################
# 使用 argbind 进行装饰
@argbind.bind("train", "val")
def build_transform(
    augment_prob: float = 1.0,
    preprocess: list = ["Identity"],
    augment: list = ["Identity"],
    postprocess: list = ["Identity"],
):
    # 定义一个 lambda 函数，将列表中的字符串转换为变换对象
    to_tfm = lambda l: [getattr(tfm, x)() for x in l]
    ##to_tfm到底有哪些？
    # 创建 preprocess 变换管道
    preprocess = transforms.Compose(*to_tfm(preprocess), name="preprocess")
    
    # 创建 augment 变换管道，并设置应用概率
    augment = transforms.Compose(*to_tfm(augment), name="augment", prob=augment_prob)
    
    # 创建 postprocess 变换管道
    postprocess = transforms.Compose(*to_tfm(postprocess), name="postprocess")
    
    # 将 preprocess、augment 和 postprocess 变换管道组合成一个最终的变换管道
    transform = transforms.Compose(preprocess, augment, postprocess)
    
    # 返回最终的变换管道
    return transform

#############################################################
"""
类似：
preprocess = transforms.Compose(
    tfm.GlobalVolumeNorm(),
    tfm.CrossTalk(),
    name="preprocess",
)
augment = transforms.Compose(
    tfm.RoomImpulseResponse(),
    tfm.BackgroundNoise(),
    name="augment",
)
postprocess = transforms.Compose(
    tfm.VolumeChange(),
    tfm.RescaleAudio(),
    tfm.ShiftPhase(),
    name="postprocess",
)
transform = transforms.Compose(preprocess, augment, postprocess)

"""





if __name__=="__main__":
    # 调用绑定函数
    tfm = bind_module(transforms, "train", "val", filter_fn=filter_fn)
