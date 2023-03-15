import os
import random
from datetime import timedelta

import numpy as np
import torch
import torchvision

import wandb
from hydra.utils import get_original_cwd
from omegaconf import OmegaConf
from src import ff_mnist, ff_model
#from src import ff_mnist, ff_model_conv


def parse_args(opt):
    '''为 np/torch/random 都设置同一种子, 并打印本次实验的 config 信息, 最后原样返回 opt.'''
    np.random.seed(opt.seed)
    torch.manual_seed(opt.seed)
    random.seed(opt.seed)

    # print(OmegaConf.to_yaml(opt))   # 输出配置文件的所有参数信息.
    return opt


def show_model_parameters(opt):
    '''(自定义)打印出 model.classification_loss.parameters(), 查看其具体有哪些参数.'''
    model = ff_model.FF_model(opt)
    #model = ff_model_conv.FF_model_conv(opt)
    if "cuda" in opt.device:
        model = model.cuda() 
    for x in model.classification_loss.parameters():
        print(x)


def get_model_and_optimizer(opt):
    model = ff_model.FF_model(opt)
    #model = ff_model_conv.FF_model_conv(opt)
    if "cuda" in opt.device:
        model = model.cuda()
    # print(model, "\n")  # 输出 FF_model 的组件信息.

    # Create optimizer with different hyper-parameters for the main model
    # and the downstream classification model.
    main_model_params = [
        p
        for p in model.parameters()
        if all(p is not x for x in model.classification_loss.parameters())
        # 疑问: 经试验, model.classification_loss.parameters() 是一个空的 generator?
    ]
    # Torch.optim 的 "per-parameter options" 初始化方法: 用多个字典定义多个独立的 parameter group.
    # 纠错: main model 中的参数是否应该是 model.model.parameters(),
    # 而 downstream classification model 中的参数是否应该是 model.linear_classifier.parameters()?
    optimizer = torch.optim.SGD(
        [
            {
                # main model 中需要更新的参数.
                "params": main_model_params,
                "lr": opt.training.learning_rate,
                "weight_decay": opt.training.weight_decay,
                "momentum": opt.training.momentum,
            },
            {
                # downstream classification model 中需要更新的参数.
                "params": model.classification_loss.parameters(),
                "lr": opt.training.downstream_learning_rate,
                "weight_decay": opt.training.downstream_weight_decay,
                "momentum": opt.training.momentum,
            },
        ]
    )
    return model, optimizer


def get_data(opt, partition):
    '''由 FF_MNIST dataset 封装返回一个 dataloader'''
    dataset = ff_mnist.FF_MNIST(opt, partition)

    # Improve reproducibility in dataloader.
    g = torch.Generator()
    g.manual_seed(opt.seed)

    return torch.utils.data.DataLoader(
        dataset,
        batch_size=opt.input.batch_size,
        drop_last=True,
        shuffle=True,
        worker_init_fn=seed_worker,
        generator=g,
        num_workers=4,
        persistent_workers=True,
    )


def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2 ** 32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def get_MNIST_partition(opt, partition):
    '''获取 dataset 的主要成分(用于 dataset 的构造)'''
    # 这里的 dataset 只被 transform 成为 Tensor, 没有经过 normalization 或 flatten. 
    if partition in ["train", "val", "train_val"]:
        mnist = torchvision.datasets.MNIST(
            os.path.join(get_original_cwd(), opt.input.path),
            train=True,
            download=True,
            transform=torchvision.transforms.ToTensor(),
        )
    elif partition in ["test"]:
        mnist = torchvision.datasets.MNIST(
            os.path.join(get_original_cwd(), opt.input.path),
            train=False,
            download=True,
            transform=torchvision.transforms.ToTensor(),
        )
    else:
        raise NotImplementedError

    # 分 train 与 val 两种情况, 进一步分割 training set.
    if partition == "train":
        mnist = torch.utils.data.Subset(mnist, range(50000))
    elif partition == "val":
        mnist = torchvision.datasets.MNIST(
            os.path.join(get_original_cwd(), opt.input.path),
            train=True,
            download=True,
            transform=torchvision.transforms.ToTensor(),
        )   # 这里为什么要再写一遍完全相同的数据集? 这样会打乱得到的 mnist 中图像的顺序吗?
        mnist = torch.utils.data.Subset(mnist, range(50000, 60000))

    return mnist


def dict_to_cuda(dict):
    '''把 dict 中存储的 value 放到 cuda 上'''
    for key, value in dict.items():
        dict[key] = value.cuda(non_blocking=True)
    return dict


def preprocess_inputs(opt, inputs, labels):
    '''把 inputs 和 labels 两个 dict 都放到 cuda 上'''
    if "cuda" in opt.device:
        inputs = dict_to_cuda(inputs)
        labels = dict_to_cuda(labels)
    return inputs, labels


def get_linear_cooldown_lr(opt, epoch, lr):
    '''当 epoch 过半之后, 对于每个新的 epoch, lr 都会以线性的速率减小.'''
    if epoch > (opt.training.epochs // 2):
        return lr * 2 * (1 + opt.training.epochs - epoch) / opt.training.epochs
    else:
        return lr


def update_learning_rate(optimizer, opt, epoch):
    '''在每个新的 epoch 都要 cooldown 当前 optimizer 的 lr.'''
    optimizer.param_groups[0]["lr"] = get_linear_cooldown_lr(
        opt, epoch, opt.training.learning_rate
    )
    optimizer.param_groups[1]["lr"] = get_linear_cooldown_lr(
        opt, epoch, opt.training.downstream_learning_rate
    )
    return optimizer


def get_accuracy(opt, output, target):
    """Computes the accuracy."""
    with torch.no_grad():
        prediction = torch.argmax(output, dim=1)
        return (prediction == target).sum() / opt.input.batch_size


def print_results(partition, iteration_time, scalar_outputs, epoch=None):
    if epoch is not None:
        print(f"Epoch {epoch} \t", end="")

    print(
        f"{partition} \t \t"
        f"Time: {timedelta(seconds=iteration_time)} \t",
        end="",
    )
    if scalar_outputs is not None:
        for key, value in scalar_outputs.items():
            print(f"{key}: {value:.4f} \t", end="") # 先输出当前 epoch 的训练/测试结果.
        
        # 再把当前 epoch 的训练/测试结果上传到 wandb.
        if partition == "train":
            wandb.log({ "Loss": scalar_outputs["Loss"],
                        "Peer Normalization": scalar_outputs["Peer Normalization"],
                        "loss_layer_0": scalar_outputs["loss_layer_0"],
                        "loss_layer_1": scalar_outputs["loss_layer_1"],
                        "loss_layer_2": scalar_outputs["loss_layer_2"],
                        "ff_acc_layer_0": scalar_outputs["ff_accuracy_layer_0"],
                        "ff_acc_layer_1": scalar_outputs["ff_accuracy_layer_1"],
                        "ff_acc_layer_2": scalar_outputs["ff_accuracy_layer_2"],
                        "cls_loss": scalar_outputs["classification_loss"],
                        "cls_acc": scalar_outputs["classification_accuracy"] })
        elif partition == "val":
            wandb.log({ "Val Loss": scalar_outputs["Loss"],
                        "Val cls_loss": scalar_outputs["classification_loss"],
                        "Val cls_acc": scalar_outputs["classification_accuracy"] })
        elif partition == "test":
            wandb.log({ "Test Loss": scalar_outputs["Loss"],
                        "Test cls_loss": scalar_outputs["classification_loss"],
                        "Test cls_acc": scalar_outputs["classification_accuracy"] })
    print()


def log_results(result_dict, scalar_outputs, num_steps):
    for key, value in scalar_outputs.items():
        if isinstance(value, float):
            result_dict[key] += value / num_steps
        else:
            result_dict[key] += value.item() / num_steps
    return result_dict
