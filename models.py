
import torch
import lightning.pytorch as pl
from torch.autograd import Variable

class SamplingLayer(pl.LightningModule):
    def __init__(self):
        super(SamplingLayer, self).__init__()
    
    def forward(self, x):
        epsilon = None
        if(len(x.shape) == 4):
        
            output_std = x.view(x.shape[0], x.shape[1], -1).std(axis=-1, keepdim=True, unbiased = False)
            output_std = output_std.unsqueeze(-1)


            epsilon = torch.rand(x.shape[0],x.shape[1],1,1).expand_as(x)
        else:
            print('None Epsilon')
            return
            epsilon = torch.rand_like(x)
        
        return x + output_std.expand_as(x) * epsilon.to(self.device)


class LearnableSamplingLayer(pl.LightningModule):
    def __init__(self, width):
        super(LearnableSamplingLayer, self).__init__()
        self.std = None
        print(self.device)
        self.std = Variable(torch.randn(1,width,1,1).type(torch.FloatTensor), requires_grad=True).to(self.device)
    
    def forward(self, x):
        epsilon = None
        if(len(x.shape) == 4):
            batch, ch, h, w = x.shape
        
            # output_std = x.view(x.shape[0], x.shape[1], -1).std(axis=-1, keepdim=True, unbiased = False)
            # output_std = output_std.unsqueeze(-1)
            # batch, ch, h, w = output_std.shape
            
            output_std = self.std
            epsilon = torch.rand_like(self.std)
        else:
            print('None Epsilon')
            return
            epsilon = torch.rand_like(x)
        
        return x + output_std.to(self.device) * epsilon.to(self.device)

