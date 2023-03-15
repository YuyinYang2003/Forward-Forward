import math

import torch
import torch.nn as nn

from src import utils


class FF_model_conv(torch.nn.Module):
    """The model trained with Forward-Forward (FF)."""
    """这里实现的是将原代码中三层全连接层换成三个共享权重的卷积神经网络,有pooling layer,三层生成的特征图拉长成的向量都成为最后线性分类器的输入"""
    def __init__(self, opt):
        super(FF_model_conv, self).__init__()

        self.opt = opt
        self.num_channels = [128,128,256,256,512,512,512,512]
        self.num_strides=[1,1,1,1,1,1,2,1]
        self.receptions=[3,3,3,3,3,3,3,2]
        self.padding=[1,1,1,1,1,1,0,0]
        self.act_fn = ReLU_full_grad()

        # Initialize the model.
        self.model = nn.ModuleList([nn.Conv2d(in_channels=1, out_channels=self.num_channels[0], kernel_size=self.receptions[0],stride=self.num_strides[0],padding=self.padding[0])])
        for i in range(1, len(self.num_channels)):
            self.model.append(nn.Conv2d(self.num_channels[i-1], out_channels=self.num_channels[i], kernel_size=self.receptions[i],stride=self.num_strides[i],padding=self.padding[i]))
            if i==1 or i==3:
                self.model.append(nn.MaxPool2d(kernel_size=2))

        self.dim=[28,28,14,14,7,7,3,2]
        # Initialize forward-forward loss.
        self.ff_loss = nn.BCEWithLogitsLoss()

        # Initialize peer normalization loss.
        self.running_means = [
            torch.zeros(self.num_channels[i]*(self.dim[i]**2), device=self.opt.device) + 0.5
            for i in range(self.opt.model.num_layers-2)
        ]

        # Initialize downstream classification loss.
        # 这里应该是实现原论文中 "one-pass" softmax 的 test 方法.
        channels_for_classification_loss = sum(
            self.num_channels[-i]*(self.dim[-i]**2) for i in range(self.opt.model.num_layers-2)
        )
        # 下游的线性分类器 并不被包括在 self.model 中, 而是单独列为 self.linear_classifier.
        self.linear_classifier = nn.Sequential(
            nn.Linear(channels_for_classification_loss, 10, bias=False)
        )
        self.classification_loss = nn.CrossEntropyLoss()

        # Initialize weights.
        self._init_weights()

    def _init_weights(self):
        # 纠错: 是否该改为 model.children() ?
        for m in self.model.modules():
            if isinstance(m, nn.Conv2d):
                # 用了正态分布来初始化 weight_matrix, 而没有用 nn.Linear 默认的均匀分布.
                torch.nn.init.normal_(
                    m.weight, mean=0, std=1 / math.sqrt(m.weight.shape[0]*m.weight.shape[1]**2)
                )
                torch.nn.init.zeros_(m.bias)

        for m in self.linear_classifier.modules():
            if isinstance(m, nn.Linear):
                nn.init.zeros_(m.weight)

    def _layer_norm(self, z, eps=1e-8):
        # dim=-1 指最后一个 dim, 应该总归指一个 sample 中的多个分量所在的维度.
        t=torch.reshape(z,(z.shape[0],-1))
        s=z.shape
        t=t/(torch.sqrt(torch.mean(t ** 2, dim=-1, keepdim=True)) + eps)
        return torch.reshape(t,(s))

    def _calc_peer_normalization_loss(self, idx, x):
        # Only calculate mean activity over positive samples.
        z=torch.reshape(x,(x.shape[0],-1))
        mean_activity = torch.mean(z[:self.opt.input.batch_size], dim=0)

        self.running_means[idx] = self.running_means[
            idx
        ].detach() * self.opt.model.momentum + mean_activity * (
            1 - self.opt.model.momentum
        )

        peer_loss = (torch.mean(self.running_means[idx]) - self.running_means[idx]) ** 2
        return torch.mean(peer_loss)

    def _calc_ff_loss(self, x, labels):
        z=torch.reshape(x,(x.shape[0],-1))
        sum_of_squares = torch.sum(z ** 2, dim=-1)

        # 这里的 z.shape[1] 即为原论文 2.0 节所写公式中的 theta.
        logits = sum_of_squares - z.shape[1]
        ff_loss = self.ff_loss(logits, labels.float())

        with torch.no_grad():
            # 计算每层 ff_layer 的 accuracy 的方式:
            # 对于 pos_sample, 若 logits > 0, 则代表分类正确; 对于 neg_sample, 若 logits < 0, 则代表分类正确.
            # 最后计算 "分类正确" 的 sample 数目占整个 batch 的 sample 总数的比例, 即为 ff_accuracy.
            ff_accuracy = (
                torch.sum((torch.sigmoid(logits) > 0.5) == labels)  # equivalent to `logits > 0`
                / z.shape[0]
            ).item()
        return ff_loss, ff_accuracy

    def forward(self, inputs, labels):
        '''输入一个 batch 的 samples, 经过3层线性层后, 再经过线性分类器, 最后返回输出 dict.'''
        scalar_outputs = {
            "Loss": torch.zeros(1, device=self.opt.device),
            "Peer Normalization": torch.zeros(1, device=self.opt.device),
        }

        # Concatenate positive and negative samples and create corresponding labels.
        # 沿着 batch 维度把 pos_tensors 与 neg_tensors 拼在一起构成 z, 并创建对应的 labels.
        # 其中 pos data 的 label 值为 1, neg data 的 label 值为 0.
        z = torch.cat([inputs["pos_images"], inputs["neg_images"]], dim=0) 
        posneg_labels = torch.zeros(z.shape[0], device=self.opt.device)
        posneg_labels[: self.opt.input.batch_size] = 1



        z = self._layer_norm(z)

        for idx, layer in enumerate(self.model):
            z = layer(z)
            z = self.act_fn.apply(z)

            # 可选项: 是否把 peer loss 这个正则因子加入最终的 Loss 中.
            if isinstance(layer, nn.Conv2d):
                if self.opt.model.peer_normalization > 0:
                    if idx==3 or idx==4:
                        idx=idx-1
                    elif idx>=6 and idx<=9:
                        idx=idx-2
                    peer_loss = self._calc_peer_normalization_loss(idx, z)
                    scalar_outputs["Peer Normalization"] += peer_loss
                    scalar_outputs["Loss"] += self.opt.model.peer_normalization * peer_loss
                    
                ff_loss, ff_accuracy = self._calc_ff_loss(z, posneg_labels)
                scalar_outputs[f"loss_layer_{idx}"] = ff_loss
                scalar_outputs[f"ff_accuracy_layer_{idx}"] = ff_accuracy
                scalar_outputs["Loss"] += ff_loss
            z = z.detach()  # 特别注意: 把 z 送入下一层 layer 的 forward 之前, 必须 detach 掉 z 在上一层 layer 中的计算图.

            z = self._layer_norm(z)

        scalar_outputs = self.forward_downstream_classification_model(
            inputs, labels, scalar_outputs=scalar_outputs
        )

        return scalar_outputs

    def forward_downstream_classification_model(
        self, inputs, labels, scalar_outputs=None,
    ):
        '''输入一个 batch 的 samples, 通过 ff_layers 后把 activity vector 送入线性分类器,
        计算 分类误差 和 分类精度, 最终返回更新后的 scalar_outputs.'''
        if scalar_outputs is None:
            scalar_outputs = {
                "Loss": torch.zeros(1, device=self.opt.device),
            }

        z = inputs["neutral_sample"]
        z = self._layer_norm(z)

        input_classification_model = []

        with torch.no_grad():
            for idx, layer in enumerate(self.model):
                z = layer(z)
                z = self.act_fn.apply(z)
                z = self._layer_norm(z)
                # 收集从第2层 hidden layer 开始的 activity vector.
                # 注意: 选取的都是归一化后的 activity vector.
                #if idx >= 1:
                if isinstance(layer, nn.Conv2d):
                    input_classification_model.append(torch.reshape(z,(z.shape[0],-1)))

        input_classification_model = torch.concat(input_classification_model, dim=-1)

        # 合并后的 activity vector 在输入线性分类器之前, 先要与它在 ff_layers 中的计算图解绑.
        # 这可能是为了, 最终对 classification_loss 求导时, 求到 input_classification_model 就停止,
        # 以免继续往前回溯计算图.
        output = self.linear_classifier(input_classification_model.detach())
        output = output - torch.max(output, dim=-1, keepdim=True)[0] # torch.max 的返回值是一个 
        # namedtuple (values, indices), 因此需要再取一次 [0]; 这就相当于 output 每一行减去该行最大值.
        classification_loss = self.classification_loss(output, labels["class_labels"])
        classification_accuracy = utils.get_accuracy(
            self.opt, output.data, labels["class_labels"]
        )

        scalar_outputs["Loss"] += classification_loss
        scalar_outputs["classification_loss"] = classification_loss
        scalar_outputs["classification_accuracy"] = classification_accuracy
        return scalar_outputs


class ReLU_full_grad(torch.autograd.Function):
    """ ReLU activation function that passes through the gradient irrespective of its input value. """
    # 是否意味着 x < 0 时同样有 ReLU(x) = x, 而不会有 "死亡 ReLU" 产生?

    @staticmethod
    def forward(ctx, input):
        return input.clamp(min=0)

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output.clone()
