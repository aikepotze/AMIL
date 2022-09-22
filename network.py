"""
Code provided with the paper:
"Attribute Prediction as Multiple Instance Learning"
Network and losses
22-09-2022
"""


import numpy as np
import torch

class Net(torch.nn.Module):
    def __init__(self, init_model, n_attr):
        super().__init__()
        self.conv1 = init_model.conv1
        self.bn1 = init_model.bn1
        self.relu = init_model.relu
        self.maxpool = init_model.maxpool
        self.layer1 = init_model.layer1
        self.layer2 = init_model.layer2
        self.layer3 = init_model.layer3
        self.layer4 = init_model.layer4
        self.layer4[2].relu = torch.nn.Identity()
        self.dropout = torch.nn.Dropout2d(0.5)
        self.finalpool = torch.nn.AdaptiveMaxPool2d(output_size=(1, 1))
        self.bn = torch.nn.BatchNorm2d(2048, affine=True)
        self.fc = torch.nn.Conv2d(2048, n_attr, 1, bias=False)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        n = torch.norm(x, dim=1, keepdim=True)  # .detach()
        nw = torch.norm(self.fc.weight, dim=1, keepdim=False).unsqueeze(0) #[A] changed model to self

        maps = self.fc(x)  # .relu().add(1e-32).log()
        dist = 1 - torch.nn.functional.conv2d(x.detach(), self.fc.weight) / (n.detach() * nw)
        x = maps.reshape([maps.shape[0], maps.shape[1], maps.shape[2] * maps.shape[3]])

        attn = x.softmax(2)
        x = (x * attn).sum(2)

        return x, attn.reshape(maps.shape), dist  # *maps


class TruncatedLoss(torch.nn.Module):
    def __init__(self, device, n_attr, q=0.7, k=0.5, trainset_size=50000):
        super(TruncatedLoss, self).__init__()
        self.q = q
        self.k = k
        self.device = device
        self.weight = torch.nn.Parameter(data=torch.ones(trainset_size, n_attr), requires_grad=False)

    def forward(self, logits, targets, indexes):
        Yg = logits  # torch.gather(p, 1, torch.unsqueeze(targets, 1))
        loss_pos = ((1 - torch.pow(Yg.relu() + 1e-5,
                                   self.q)) / self.q)  # *self.weight[indexes].to(self.device)  - ((1-(self.k**self.q))/self.q)*self.weight[indexes].to(self.device)
        loss_neg = ((1 - torch.pow((1 - Yg).relu() + 1e-5,
                                   self.q)) / self.q)  # *self.weight[indexes].to(self.device) - ((1-(self.k**self.q))/self.q)*self.weight[indexes].to(self.device)
        loss_pos = torch.masked_select(loss_pos, targets == 1)
        loss_neg = torch.masked_select(loss_neg, targets == 0)
        loss = loss_pos.mean() + loss_neg.mean()
        # pdb.set_trace()
        return loss

    def update_weight(self, logits, targets, indexes):
        p = logits  # F.softmax(logits, dim=1)
        Yg = logits  # torch.gather(p, 1, torch.unsqueeze(targets, 1))
        Lq = ((1 - (Yg ** self.q)) / self.q) * targets + ((1 - ((1 - Yg) ** self.q)) / self.q) * (1 - targets)
        Lqk = np.repeat(((1 - (self.k ** self.q)) / self.q), targets.size(0))
        Lqk = torch.from_numpy(Lqk).type(torch.FloatTensor)
        Lqk = torch.unsqueeze(Lqk, 1)
        condition = torch.gt(Lqk, Lq)
        self.weight[indexes] = condition.type(torch.FloatTensor)
