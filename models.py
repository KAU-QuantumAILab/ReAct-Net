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
<<<<<<< HEAD
            return x + output_std.expand_as(x) * epsilon.to(self.device)


class BeforeRaCUN(pl.LightningModule):
    def __init__(self, hidden_feature):
        super(BeforeRaCUN, self).__init__()
        self.std = Variable(torch.randn(1,hidden_feature).type(torch.FloatTensor), requires_grad=True).to(self.device)

    def forward(self, x):
        output_std = self.std / torch.sqrt(torch.tensor(x.shape[-1])).detach()
        epsilon = torch.randn_like(x)
        return x + output_std.to(self.device) * epsilon.to(self.device)
        

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
=======
            return x + output_std.to(self.device) * epsilon.to(self.device)
<<<<<<< HEAD

        

class MLPMnist(pl.LightningModule):
    def __init__(self, config):
        super(MLPMnist, self).__init__()
        self.config = config
        self.config['num_classes'] = 10
        if self.config["activation"] == 'relu':
            self.activation = nn.ReLU(inplace=True)
        else:
            self.activation = SamplingLayer(100)
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
        if self.config["activation"] == 'relu':
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
        else:
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

    def forward(self, x):
        x = self.module(x)
        return x
    

class Net(nn.Module):
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
        self.fcsample = SamplingLayer(128)
        
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
        x = self.fcsample(x)
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
=======

        
        
>>>>>>> 71936f686f19c4941c35d2d67021d9dc3ace522a
>>>>>>> be889062aea6abda1e862b394fbb7cc77981c2b4
