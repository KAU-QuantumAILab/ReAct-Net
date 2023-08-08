
import torch
import lightning.pytorch as pl
from torch.autograd import Variable

class SamplingLayer(pl.LightningModule):
    def __init__(self, hidden_feature=None):
        super(SamplingLayer, self).__init__()
        if(hidden_feature):
            self.std = Variable(torch.randn(1,hidden_feature).type(torch.FloatTensor), requires_grad=True).to(self.device)

    def forward(self, x):
        epsilon = None
        if(len(x.shape) == 4):
        
            output_std = x.view(x.shape[0], x.shape[1], -1).std(axis=-1, keepdim=True, unbiased = False)
            output_std = output_std.unsqueeze(-1)
            output_std = output_std/torch.sqrt(torch.tensor(x.shape[-2] * x.shape[-1]).detach())

            epsilon = torch.randn(x.shape[0],x.shape[1],1,1).expand_as(x)
            return x + output_std.expand_as(x) * epsilon.to(self.device)
        else:
            output_std = self.std / torch.sqrt(torch.tensor(x.shape[-1])).detach()
            epsilon = torch.randn_like(x)
            return x + output_std.to(self.device) * epsilon.to(self.device)

        
        