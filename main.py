import time
from collections import defaultdict

import hydra
import wandb
import torch
import omegaconf
from omegaconf import DictConfig
import os
from src import utils


def train(opt, model, optimizer):
    '''epoch -> batch -> layer, 完成所有训练, 返回参数训练完毕的 model.'''
    start_time = time.time()
    train_loader = utils.get_data(opt, "train")
    num_steps_per_epoch = len(train_loader) # 即 num_batches, 总的批次数目.

    for epoch in range(opt.training.epochs):
        train_results = defaultdict(float)
        optimizer = utils.update_learning_rate(opt, optimizer, epoch)

        for inputs, labels in train_loader:
            inputs, labels = utils.preprocess_inputs(opt, inputs, labels)

            optimizer.zero_grad()

            scalar_outputs = model(inputs, labels)
            scalar_outputs["Loss"].backward()   # 3层 ff_loss 与 线性分类器的 classification_loss 加在一起,
            # .backward() 时相当于对每层的 loss 分别求导, 不需要多层的反向传播.

            optimizer.step()

            # 把当前这个 batch 的训练结果更新到总的 train_results 中, num_batches 决定了这次结果占总结果的权重.
            train_results = utils.log_results(
                train_results, scalar_outputs, num_steps_per_epoch
            )

        utils.print_results(opt, "train", time.time() - start_time, train_results, epoch)
        start_time = time.time()

        # Validate.
        if epoch % opt.training.val_idx == 0 and opt.training.val_idx != -1:
            validate_or_test(opt, model, "val", epoch=epoch)

    return model


def validate_or_test(opt, model, partition, epoch=None):
    test_time = time.time()
    test_results = defaultdict(float)

    data_loader = utils.get_data(opt, partition)
    num_steps_per_epoch = len(data_loader)

    model.eval()    # model 默认是在 train() 状态, 因此上面的 train 函数中不需要 model.train().
    print(partition)
    with torch.no_grad():
        for inputs, labels in data_loader:
            inputs, labels = utils.preprocess_inputs(opt, inputs, labels)

            if opt.training.test_mode == "one_pass_softmax":
                scalar_outputs = model.forward_downstream_classification_model(
                    inputs, labels
                )
            elif opt.training.test_mode == "compute_each_label":
                scalar_outputs = model.forward_accumulate_label_goodness(
                    inputs, labels
                )

            test_results = utils.log_results(
                test_results, scalar_outputs, num_steps_per_epoch
            )

    utils.print_results(opt, partition, time.time() - test_time, test_results, epoch=epoch)
    model.train()   # test 结束后把 model 状态恢复到默认的 train().


# 每次跑实验前, 记得更改 config_name="对应数据集".
@hydra.main(version_base=None, config_path="configs/", config_name="cifar10")
def my_main(opt: DictConfig) -> None:
    opt = utils.parse_args(opt)

    wandb_cfg = omegaconf.OmegaConf.to_container(
        opt, resolve=True, throw_on_missing=True
    )
    wandb.init(entity=opt.wandb.setup.entity, project=opt.wandb.setup.project, config=wandb_cfg)

    model, optimizer = utils.get_model_and_optimizer(opt)
    model = train(opt, model, optimizer)
    validate_or_test(opt, model, "val")

    if opt.training.final_test:
        validate_or_test(opt, model, "test")

    wandb.finish()


# 每次跑实验前, 记得更改 config_name="对应数据集".
@hydra.main(version_base=None, config_path="configs/", config_name="cifar10")
def show_parameters(opt: DictConfig) -> None:
    '''(自定义)查看 model.parameters() 属性.'''
    opt = utils.parse_args(opt)
    utils.show_model_parameters(opt)


if __name__ == "__main__":
    os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
    my_main()

