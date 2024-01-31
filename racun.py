
import torch
import lightning as L
import lightning.pytorch as pl
from torch.autograd import Variable
import torch.nn.functional as F
from torch import optim, nn, utils, Tensor

     
class SamplingLayer(pl.LightningModule):
    def __init__(self, hidden_feature=None):
        super(SamplingLayer, self).__init__()
        if(hidden_feature):
            self.std = Variable(torch.randn(1,hidden_feature).type(torch.FloatTensor), requires_grad=True).to(self.device)

    def forward(self, x, identity):
        
        middleIdx = x.shape[1]//2
        mu = x
        std = identity
        # mu = x[:,::2]
        # std = x[:,1::2]
        epsilon = torch.randn_like(std)

        return mu + std*epsilon