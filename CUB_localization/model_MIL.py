import torch


class Net(torch.nn.Module):
  def __init__(self,init_model):
    super(Net,self).__init__()
    self.conv1 = init_model.conv1
    self.bn1 = init_model.bn1
    self.relu = init_model.relu
    self.maxpool = init_model.maxpool
    self.layer1 = init_model.layer1
    self.layer2 = init_model.layer2
    self.layer3 = init_model.layer3
    self.layer4 = init_model.layer4
    self.dropout = torch.nn.Dropout2d(0.9)
    self.finalpool = torch.nn.AdaptiveMaxPool2d(output_size=(1, 1))
    self.fc = torch.nn.Conv2d(2048,312,1,bias=False)
    
  def forward(self, x,attribute):
    x = self.conv1(x)
    x = self.bn1(x)
    x = self.relu(x)
    x = self.maxpool(x)
    x = self.layer1(x)
    x = self.layer2(x)
    x = self.layer3(x)
    x = self.layer4(x)
    
    attention = dict()
    pre_attri = dict()
    pre_class = dict()

    maps = self.fc(x)
    x = maps.reshape([maps.shape[0],maps.shape[1],maps.shape[2]*maps.shape[3]])
    attn = x.softmax(2)
    x = (x*attn).sum(2)
    output_final = None
    pre_attri['layer4'] = x
    attention['layer4'] = attn.reshape(maps.shape)
    pre_class['layer4'] = None
    return output_final, pre_attri, attention, pre_class
