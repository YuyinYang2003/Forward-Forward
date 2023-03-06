import numpy as np
import torch

from src import utils


class FF_MNIST(torch.utils.data.Dataset):
    def __init__(self, opt, partition, num_classes=10):
        self.opt = opt
        self.mnist = utils.get_MNIST_partition(opt, partition)
        self.num_classes = num_classes
        self.uniform_label = torch.ones(self.num_classes) / self.num_classes    #对所有label权重相同的向量

    def __getitem__(self, index):
        #输入index,输出数据集相应index的sample变化出pos/neg/neutral sample和对应的正确label
        pos_sample, neg_sample, neutral_sample, class_label = self._generate_sample(
            index
        )

        inputs = {
            "pos_images": pos_sample,
            "neg_images": neg_sample,
            "neutral_sample": neutral_sample,
        }
        labels = {"class_labels": class_label}
        return inputs, labels

    def __len__(self):
        return len(self.mnist)

    def _get_pos_sample(self, sample, class_label):
        #输入sample和对应的label,输出前十个元素对应了正确label的pos sample
        one_hot_label = torch.nn.functional.one_hot(
            torch.tensor(class_label), num_classes=self.num_classes
        )   #class_label位置为1,其余为0的长度为num_classes的向量
        pos_sample = sample.clone()
        pos_sample[:, 0, : self.num_classes] = one_hot_label    #pos_sample形状是CxHxW,将第一行前num_classes个元素替换为对应label的one-hot vector
        return pos_sample

    def _get_neg_sample(self, sample, class_label):
        #输入sample和对应的label,输出前十个元素对应了错误label的pos sample
        # Create randomly sampled one-hot label.
        classes = list(range(self.num_classes))
        classes.remove(class_label)  # Remove true label from possible choices.
        wrong_class_label = np.random.choice(classes)
        one_hot_label = torch.nn.functional.one_hot(
            torch.tensor(wrong_class_label), num_classes=self.num_classes
        )
        neg_sample = sample.clone()
        neg_sample[:, 0, : self.num_classes] = one_hot_label
        return neg_sample

    def _get_neutral_sample(self, z):
        #输入sample,输出前十个元素相同,且和为1,对所有label倾向相同的sample
        z[:, 0, : self.num_classes] = self.uniform_label
        return z

    def _generate_sample(self, index):
        # Get MNIST sample.
        sample, class_label = self.mnist[index]
        pos_sample = self._get_pos_sample(sample, class_label)
        neg_sample = self._get_neg_sample(sample, class_label)
        neutral_sample = self._get_neutral_sample(sample)
        return pos_sample, neg_sample, neutral_sample, class_label
