import math

import torch
import torch.nn as nn

from src import utils


class FF_model(torch.nn.Module):
    """The model trained with Forward-Forward (FF)."""

    def __init__(self, opt):
        super(FF_model, self).__init__()

        self.opt = opt
        self.num_channels = [self.opt.model.hidden_dim] * self.opt.model.num_layers #每一层维数一定相同吗
        self.act_fn = ReLU_full_grad()

        # Initialize the model.
        self.model = nn.ModuleList([nn.Linear(784, self.num_channels[0])])
        for i in range(1, len(self.num_channels)):
            self.model.append(nn.Linear(self.num_channels[i - 1], self.num_channels[i]))

        # Initialize forward-forward loss.
        self.ff_loss = nn.BCEWithLogitsLoss()

        # Initialize peer normalization loss.
        self.running_means = [
            torch.zeros(self.num_channels[i], device=self.opt.device) + 0.5 #初始设置为0.5，因为activity经过sigmoid会在0-1中
            for i in range(self.opt.model.num_layers)
        ]

        # Initialize downstream classification loss.
        channels_for_classification_loss = sum(
            self.num_channels[-i] for i in range(self.opt.model.num_layers - 1)
        )   #为什么要用-i？-i意味着第二层hidden layer没加？                                                                                                                                                                                                                                                                                                                                                                                                                                                                       
        self.linear_classifier = nn.Sequential(
            nn.Linear(channels_for_classification_loss, 10, bias=False)
        )   #linear classifier将（除了第一层？）所有hidden layers 的 hidden activities用作输入，输出10维
        self.classification_loss = nn.CrossEntropyLoss()    #计算最后linear classifier的loss

        # Initialize weights.
        self._init_weights()

    def _init_weights(self):
        for m in self.model.modules():
            if isinstance(m, nn.Linear):    #为什么需要判断？
                torch.nn.init.normal_(
                    m.weight, mean=0, std=1 / math.sqrt(m.weight.shape[0])  #初始化weight为均值为0，方差差为1/行维数（输出维数）的正态分布
                )
                torch.nn.init.zeros_(m.bias)

        for m in self.linear_classifier.modules():
            if isinstance(m, nn.Linear):
                nn.init.zeros_(m.weight)    #初始化为0

    def _layer_norm(self, z, eps=1e-8):
        return z / (torch.sqrt(torch.mean(z ** 2, dim=-1, keepdim=True)) + eps) #经过操作后z每个sample平方和为z.shape[-1]，也就是theta

    def _calc_peer_normalization_loss(self, idx, z):
        # Only calculate mean activity over positive samples.(为什么？)
        mean_activity = torch.mean(z[:self.opt.input.batch_size], dim=0)    #sample间求平均

        self.running_means[idx] = self.running_means[
            idx
        ].detach() * self.opt.model.momentum + mean_activity * (
            1 - self.opt.model.momentum
        )   #running means初始都为0.5，进行更新

        peer_loss = (torch.mean(self.running_means[idx]) - self.running_means[idx]) ** 2
        return torch.mean(peer_loss)    #相当于返回了更新后running means方差，如果这个方差很大，说明可能有些hidden units太活跃或者太没影响，计入peer_normalization_loss

    def _calc_ff_loss(self, z, labels):
        sum_of_squares = torch.sum(z ** 2, dim=-1)

        logits = sum_of_squares - z.shape[1]    #theta=z.shape[1]
        ff_loss = self.ff_loss(logits, labels.float())  #希望pos sample的logit为正数尽可能大，对应label为1，neg sample的logit为负数，对应label为0

        with torch.no_grad():
            ff_accuracy = (
                torch.sum((torch.sigmoid(logits) > 0.5) == labels)  #例如pos sample的logits>0时，torch.sigmoid(logits)>0.5，与label为1对应时，计入accuracy
                / z.shape[0]
            ).item()
        return ff_loss, ff_accuracy

    def forward(self, inputs, labels):
        #让sample经过所有线性层，叠加每个线性层的ff_loss和peer_normalization_loss
        scalar_outputs = {
            "Loss": torch.zeros(1, device=self.opt.device),
            "Peer Normalization": torch.zeros(1, device=self.opt.device),
        }

        # Concatenate positive and negative samples and create corresponding labels.
        z = torch.cat([inputs["pos_images"], inputs["neg_images"]], dim=0)
        posneg_labels = torch.zeros(z.shape[0], device=self.opt.device)
        posneg_labels[: self.opt.input.batch_size] = 1  #posneg_labels在pos_images对应位置为1，neg为0

        z = z.reshape(z.shape[0], -1)   #每个sample变成一维
        z = self._layer_norm(z)     #每个sample向量元素平方和为z.shape[-1]，也就是theta

        for idx, layer in enumerate(self.model):
            #经过所有线性层，叠加每个线性层的ff_loss和peer_normalization_loss
            z = layer(z)
            z = self.act_fn.apply(z)

            if self.opt.model.peer_normalization > 0:
                peer_loss = self._calc_peer_normalization_loss(idx, z)
                scalar_outputs["Peer Normalization"] += peer_loss
                scalar_outputs["Loss"] += self.opt.model.peer_normalization * peer_loss

            ff_loss, ff_accuracy = self._calc_ff_loss(z, posneg_labels)
            scalar_outputs[f"loss_layer_{idx}"] = ff_loss
            scalar_outputs[f"ff_accuracy_layer_{idx}"] = ff_accuracy
            scalar_outputs["Loss"] += ff_loss
            z = z.detach()  #把 z 送入下一层 layer 的 forward 之前, 必须 detach 掉 z 在上一层 layer 中的计算图

            z = self._layer_norm(z)     #每个layer结束‘normalize the length’，强迫下一个layer用hidden vector的相对信息

        scalar_outputs = self.forward_downstream_classification_model(
            inputs, labels, scalar_outputs=scalar_outputs
        )

        return scalar_outputs

    def forward_downstream_classification_model(
        self, inputs, labels, scalar_outputs=None,
    ):  #forward是pos和neg的sample经过hidden layer过程中产生的loss，这里是neutral的sample经过layer后经过线性分类器
        #输出dict更新了Loss，加上了classification_loss，新增了classification_loss和accuracy
        if scalar_outputs is None:
            scalar_outputs = {
                "Loss": torch.zeros(1, device=self.opt.device),
            }

        z = inputs["neutral_sample"]
        z = z.reshape(z.shape[0], -1)
        z = self._layer_norm(z)

        input_classification_model = []

        with torch.no_grad():
            for idx, layer in enumerate(self.model):
                z = layer(z)
                z = self.act_fn.apply(z)
                z = self._layer_norm(z)

                if idx >= 1:
                    input_classification_model.append(z)    #第二层开始

        input_classification_model = torch.concat(input_classification_model, dim=-1)

        output = self.linear_classifier(input_classification_model.detach())    #输出10维
        output = output - torch.max(output, dim=-1, keepdim=True)[0]    #每个元素减去最大值
        classification_loss = self.classification_loss(output, labels["class_labels"])  #和label比较，计算线性分类器的loss
        classification_accuracy = utils.get_accuracy(
            self.opt, output.data, labels["class_labels"]
        )   #最大元素和label位置相同即accurate

        scalar_outputs["Loss"] += classification_loss
        scalar_outputs["classification_loss"] = classification_loss
        scalar_outputs["classification_accuracy"] = classification_accuracy
        return scalar_outputs


class ReLU_full_grad(torch.autograd.Function):
    """ ReLU activation function that passes through the gradient irrespective of its input value. """

    @staticmethod
    def forward(ctx, input):
        return input.clamp(min=0)

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output.clone()
