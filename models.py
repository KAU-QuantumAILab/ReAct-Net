import torch
import lightning.pytorch as pl
from torch.autograd import Variable
from torch import nn
import torch.nn.functional as F

class SamplingLayer(pl.LightningModule):
    def __init__(self, hidden_feature=None):
        super(SamplingLayer, self).__init__()
        # if(hidden_feature):
        #     self.std = Variable(torch.randn(1,hidden_feature).type(torch.FloatTensor), requires_grad=True).to(self.device)

    def forward(self, x):
        epsilon = None
        if(len(x.shape) == 4):
        
            output_std = x.view(x.shape[0], x.shape[1], -1).std(axis=-1, keepdim=True, unbiased = False)
            output_std = output_std.unsqueeze(-1)
            output_std = output_std/torch.sqrt(torch.tensor(x.shape[-2] * x.shape[-1]).detach())

            epsilon = torch.randn(x.shape[0],x.shape[1],1,1).expand_as(x)
            return x + output_std.expand_as(x) * epsilon.to(self.device)
        else:
            # output_std = self.std / torch.sqrt(torch.tensor(x.shape[-1])).detach()
            output_std = x.std(axis=1, keepdim = True)
            output_std = output_std / torch.sqrt(torch.tensor(x.shape[-1]).detach())
            # print("std: ",output_std.shape)
            epsilon = torch.randn_like(x)

            return x + output_std.expand_as(x) * epsilon.to(self.device)


class BeforeRaCUN(pl.LightningModule):
    def __init__(self, hidden_feature):
        super(BeforeRaCUN, self).__init__()
        self.std = Variable(torch.randn(1,hidden_feature).type(torch.FloatTensor), requires_grad=True).to(self.device)

    def forward(self, x):
        output_std = self.std / torch.sqrt(torch.tensor(x.shape[-1])).detach()
        epsilon = torch.randn_like(x)
        return x + output_std.to(self.device) * epsilon.to(self.device)


class BernoulliSampling(pl.LightningModule):
    def __init__(self, p = 0.5):
        super(BernoulliSampling, self).__init__()
        self.p = torch.tensor(p)
        
    def forward(self, x):
        if(len(x.shape) == 4):
        
            output_std = x.view(x.shape[0], x.shape[1], -1).std(axis=-1, keepdim=True, unbiased = False)
            output_std = output_std.unsqueeze(-1)
            output_std = output_std/torch.sqrt(torch.tensor(x.shape[-2] * x.shape[-1]).detach())

            epsilon = torch.bernoulli(self.p.expand_as(x))
            return x + output_std.expand_as(x) * epsilon.to(self.device)
        
        else:
            output_std = x.std(axis=1, keepdim = True)
            output_std = output_std / torch.sqrt(torch.tensor(x.shape[-1]).detach())
            
            epsilon = torch.bernoulli(self.p.expand_as(x))
            return x + output_std.expand_as(x) * epsilon.to(self.device)


class Leaky_BReLU(pl.LightningModule):
    def __init__(self):
        super(Leaky_BReLU, self).__init__()
        
    def forward(self, x):
        epsilon = torch.distributions.bernoulli.Bernoulli(logits=x).sample()
        epsilon = torch.where(epsilon==0, 0.1, 1.0)
        
        return x * epsilon.to(self.device)



class BReLU(pl.LightningModule):
    def __init__(self):
        super(BReLU, self).__init__()

    def minmax(self, x):
        x_min, _ = torch.min(x, axis=-1, keepdim=True)
        x_max, _ = torch.max(x, axis=-1, keepdim=True)
        return (x - x_min) / (x_max - x_min)
        
        
    def forward(self, x):
        if(len(x.shape) == 4):
        
            # output_mean = self.minmax(x.view(x.shape[0], x.shape[1], -1)).mean(axis=-1, keepdim=True)

            # epsilon = torch.bernoulli(output_mean.expand_as(x))

            epsilon = torch.distributions.bernoulli.Bernoulli(logits=x).sample()

            return x * epsilon.to(self.device)
        
        else:
            # 범위 조절 후 평균을 확률로 사용
            # pm = self.minmax(x).mean(axis=1, keepdim = True)
            # epsilon = torch.bernoulli(pm.expand_as(x))
            
            # 입력 자체를 logits을 통해 확률로 사용
            epsilon = torch.distributions.bernoulli.Bernoulli(logits=x).sample()

            # 절대값만 씌우고 확률로 사용
            # epsilon = torch.distributions.bernoulli.Bernoulli(logits=torch.abs(x)).sample()
            
            # 범위 조절 없이 평균 구해서 logits으로 사용
            # pm = x.mean(axis=1, keepdim = True)
            # epsilon = torch.distributions.bernoulli.Bernoulli(logits=pm.expand_as(x)).sample()
            
            return x * epsilon.to(self.device)



class MLPMnist(pl.LightningModule):
    def __init__(self, config):
        super(MLPMnist, self).__init__()
        self.config = config
        self.config['num_classes'] = 10
        if self.config["activation"] == 'relu':
            self.activation = nn.ReLU(inplace=True)
        elif self.config["activation"] == 'before':
            self.activation = BeforeRaCUN(100)
        elif self.config["activation"] == 'after':
            self.activation = SamplingLayer()
        elif self.config["activation"] == 'brelu':
            self.activation = BReLU()
        self.input_layer = nn.Linear(784, 100)
        self.output_layer = nn.Linear(100, 10)

    def forward(self, x):
        x = torch.flatten(x, 1)
        x = self.input_layer(x)
        x = self.activation(x)
        x = self.output_layer(x)
        return x
    
    
class Layer4Conv(pl.LightningModule):
    def __init__(self, config):
        super(Layer4Conv, self).__init__()
        self.config = config
        self.config['num_classes'] = 10
        if self.config["activation"] == 'rrr':
            self.module = nn.Sequential(
                nn.Conv2d(1, 3, 4, stride=2), #28 -> 13
                nn.ReLU(),
                nn.Conv2d(3, 3, 7), #13 -> 7
                nn.ReLU(),
                nn.Conv2d(3, 3, 5), #7 -> 3
                nn.ReLU(),
                nn.Conv2d(3, 10, 3), #3 -> 1
                nn.Flatten()
            )
        elif self.config["activation"] == 'sss':
            self.module = nn.Sequential(
                nn.Conv2d(1, 3, 4, stride=2), #28 -> 13
                SamplingLayer(),
                nn.Conv2d(3, 3, 7), #13 -> 7
                SamplingLayer(),
                nn.Conv2d(3, 3, 5), #7 -> 3
                SamplingLayer(),
                nn.Conv2d(3, 10, 3), #3 -> 1
                nn.Flatten()
            )
        elif self.config["activation"] == 'rss':
            self.module = nn.Sequential(
                nn.Conv2d(1, 3, 4, stride=2), #28 -> 13
                nn.ReLU(),
                nn.Conv2d(3, 3, 7), #13 -> 7
                SamplingLayer(),
                nn.Conv2d(3, 3, 5), #7 -> 3
                SamplingLayer(),
                nn.Conv2d(3, 10, 3), #3 -> 1
                nn.Flatten()
            )
        elif self.config["activation"] == 'srs':
            self.module = nn.Sequential(
                nn.Conv2d(1, 3, 4, stride=2), #28 -> 13
                SamplingLayer(),
                nn.Conv2d(3, 3, 7), #13 -> 7
                nn.ReLU(),
                nn.Conv2d(3, 3, 5), #7 -> 3
                SamplingLayer(),
                nn.Conv2d(3, 10, 3), #3 -> 1
                nn.Flatten()
            )
        elif self.config["activation"] == 'ssr':
            self.module = nn.Sequential(
                nn.Conv2d(1, 3, 4, stride=2), #28 -> 13
                SamplingLayer(),
                nn.Conv2d(3, 3, 7), #13 -> 7
                SamplingLayer(),
                nn.Conv2d(3, 3, 5), #7 -> 3
                nn.ReLU(),
                nn.Conv2d(3, 10, 3), #3 -> 1
                nn.Flatten()
            )
        elif self.config["activation"] == 'srr':
            self.module = nn.Sequential(
                nn.Conv2d(1, 3, 4, stride=2), #28 -> 13
                SamplingLayer(),
                nn.Conv2d(3, 3, 7), #13 -> 7
                nn.ReLU(),
                nn.Conv2d(3, 3, 5), #7 -> 3
                nn.ReLU(),
                nn.Conv2d(3, 10, 3), #3 -> 1
                nn.Flatten()
            )
        elif self.config["activation"] == 'rsr':
            self.module = nn.Sequential(
                nn.Conv2d(1, 3, 4, stride=2), #28 -> 13
                nn.ReLU(),
                nn.Conv2d(3, 3, 7), #13 -> 7
                SamplingLayer(),
                nn.Conv2d(3, 3, 5), #7 -> 3
                nn.ReLU(),
                nn.Conv2d(3, 10, 3), #3 -> 1
                nn.Flatten()
            )
        elif self.config["activation"] == 'rrs':
            self.module = nn.Sequential(
                nn.Conv2d(1, 3, 4, stride=2), #28 -> 13
                nn.ReLU(),
                nn.Conv2d(3, 3, 7), #13 -> 7
                nn.ReLU(),
                nn.Conv2d(3, 3, 5), #7 -> 3
                SamplingLayer(),
                nn.Conv2d(3, 10, 3), #3 -> 1
                nn.Flatten()
            )


    def forward(self, x):
        x = self.module(x)
        return x


class Net(pl.LightningModule):
    def __init__(self, config):
        super(Net, self).__init__()
        self.config = config
        self.config['num_classes'] = 10
        self.conv1 = nn.Conv2d(1, 32, 3, 1)
        self.conv2 = nn.Conv2d(32, 64, 3, 1)
        self.dropout1 = nn.Dropout(0.25)
        self.dropout2 = nn.Dropout(0.5)
        self.fc1 = nn.Linear(9216, 128)
        self.fc2 = nn.Linear(128, 10)
        self.sample = SamplingLayer()
        self.fcSample = SamplingLayer(128)
    
    def forward(self, x):
        x = self.conv1(x)
        # x = F.relu(x)
        x = self.sample(x)
        x = self.conv2(x)
        # x = F.relu(x)
        x = self.sample(x)
        x = F.max_pool2d(x, 2)
        x = self.dropout1(x)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        # x = F.relu(x)
        x = self.fcSample(x)
        x = self.dropout2(x)
        x = self.fc2(x)
        output = F.log_softmax(x, dim=1)
        return output


    
    
class Layer3Conv(pl.LightningModule):
    def __init__(self, config):
        super(Layer3Conv, self).__init__()
        self.config = config
        self.config['num_classes'] = 10
        if self.config["activation"] == 'relu':
            self.module = nn.Sequential(
                nn.Conv2d(1, 3, 4, stride=2), #28 -> 13
                nn.ReLU(),
                nn.Conv2d(3, 3, 7), #13 -> 7
                nn.ReLU(),
                nn.Conv2d(3, 10, 7), #7 -> 1
                nn.Flatten()
            )
        else:
            self.module = nn.Sequential(
                nn.Conv2d(1, 3, 4, stride=2), #28 -> 13
                SamplingLayer(),
                nn.Conv2d(3, 3, 7), #13 -> 7
                SamplingLayer(),
                nn.Conv2d(3, 10, 7), #7 -> 1
                nn.Flatten()
            )

    def forward(self, x):
        x = self.module(x)
        return x
