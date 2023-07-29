
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
            output_std = output_std/torch.sqrt(torch.tensor(x.shape[-2] * x.shape[-1]).detach())

            epsilon = torch.randn(x.shape[0],x.shape[1],1,1).expand_as(x)
        else:
            print('None Epsilon')
            return
            epsilon = torch.randn_like(x)
        
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
            epsilon = torch.randn_like(self.std)
        else:
            print('None Epsilon')
            return
            epsilon = torch.randn_like(x)
        
        return x + output_std.to(self.device) * epsilon.to(self.device)

