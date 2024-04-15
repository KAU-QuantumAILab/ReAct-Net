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

class VariableBReLU(pl.LightningModule):
    def __init__(self, alpha=1):
        super(VariableBReLU, self).__init__()
        self.alpha = alpha
        
    def forward(self, x):
        epsilon = torch.distributions.bernoulli.Bernoulli(logits=self.alpha * x).sample()
        
        return x * epsilon.to(self.device)
    
    def extra_repr(self) -> str:
        return 'alpha={}'.format(self.alpha)
    
    
class LeakyVariableBReLU(pl.LightningModule):
    def __init__(self, alpha=1):
        super(LeakyVariableBReLU, self).__init__()
        self.alpha = alpha
        
    def forward(self, x):
        epsilon = torch.distributions.bernoulli.Bernoulli(logits=self.alpha * x).sample()
        epsilon = torch.where(epsilon==0, 0.1, 1.0)
        return x * epsilon.to(self.device)
    
    def extra_repr(self) -> str:
        return 'alpha={}'.format(self.alpha)
    

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


class MeasureAct(pl.LightningModule):                   # 확률적으로 모든 값을 0 또는 1로 변환
    def __init__(self):
        super(MeasureAct, self).__init__()
        
    def forward(self, x):
        epsilon = torch.distributions.bernoulli.Bernoulli(logits=x).sample()
        epsilon = epsilon * (1/(x + 1e-16))
        
        return x * epsilon.to(self.device)


class SomeMeasureAct(pl.LightningModule):               # 확률적으로 일부 값들을 0 또는 1로 변환
    def __init__(self):
        super(SomeMeasureAct, self).__init__()
        
    def forward(self, x):
        b = torch.distributions.bernoulli.Bernoulli(logits=x)
        b1 = b.sample()
        b2 = b.sample()

        epsilon = b1*b2*(1/(x+1e-16)) + torch.logical_xor(b1, b2)
        
        return x * epsilon


class SARB(pl.LightningModule):                         #Stochastic Activation Relu or Brelu(확률적으로 Relu나 BRelu를 적용)
    def __init__(self, zero=BReLU(), one=torch.nn.ReLU(inplace=True)):
        super(SARB, self).__init__() 
        self.zero=zero
        self.one=one
        
    def forward(self, x):
        mu = x.to(torch.float32).mean()
        b = torch.distributions.bernoulli.Bernoulli(logits=mu).sample()
        # print('mu=%d'%mu)
        # print('b=%d'%b)
        
        if b == 0:
            return self.zero(x)
        
        else:
            return self.one(x)
        
    
    
class SSCA(pl.LightningModule):          # Stochastic Sin Cos Activation (확률 함수를 Sin, Cos을 활용)
    def __init__(self):
        super(SSCA, self).__init__()
        
    def forward(self, x):
        p = -torch.sin(x) * torch.cos(x + (torch.pi/2))
        p = torch.where(p < 0, 0, p)
        p = torch.where(p > 1, 1, p)
        # print(p)
        # print('out of 1')
        # print((p > 1).nonzero(as_tuple=True))
        # print(p[(p > 1).nonzero(as_tuple=True)])
        # print('out of 0')
        # print((p < 0 ).nonzero(as_tuple=True))
        # print(p[(p < 0 ).nonzero(as_tuple=True)])
        # print()
        
        epsilon = torch.distributions.bernoulli.Bernoulli(probs=p).sample()
        return x * epsilon.to(self.device)


class MeasureActSC(pl.LightningModule):                   # 확률적으로 모든 값을 0 또는 1로 변환
    def __init__(self):
        super(MeasureActSC, self).__init__()
        
    def forward(self, x):
        p = -torch.sin(x) * torch.cos(x + (torch.pi/2))
        p = torch.where(p < 0, 0, p)
        p = torch.where(p > 1, 1, p)
        # epsilon = torch.distributions.bernoulli.Bernoulli(logits=x).sample()
        epsilon = torch.distributions.bernoulli.Bernoulli(probs=p).sample()
        epsilon = epsilon * (1/(x + 1e-16))
        
        return x * epsilon.to(self.device)


class SomeMeasureActSC(pl.LightningModule):               # 확률적으로 일부 값들을 0 또는 1로 변환
    def __init__(self):
        super(SomeMeasureActSC, self).__init__()
        
    def forward(self, x):
        p = -torch.sin(x) * torch.cos(x + (torch.pi/2))
        p = torch.where(p < 0, 0, p)
        p = torch.where(p > 1, 1, p)
        # b = torch.distributions.bernoulli.Bernoulli(logits=x)
        b = torch.distributions.bernoulli.Bernoulli(probs=p)
        b1 = b.sample()
        b2 = b.sample()

        epsilon = b1*b2*(1/(x+1e-16)) + torch.logical_xor(b1, b2)
        
        return x * epsilon


class SARBSC(pl.LightningModule):                         #Stochastic Activation Relu or Brelu(확률적으로 Relu나 BRelu를 적용)
    def __init__(self, zero=BReLU(), one=torch.nn.ReLU(inplace=True)):
        super(SARBSC, self).__init__() 
        self.zero=zero
        self.one=one
        
    def forward(self, x):
        mu = x.to(torch.float32).mean()
        p = -torch.sin(mu) * torch.cos(mu + (torch.pi/2))
        p = torch.where(p < 0, 0, p)
        p = torch.where(p > 1, 1, p)
        # b = torch.distributions.bernoulli.Bernoulli(logits=mu).sample()
        b = torch.distributions.bernoulli.Bernoulli(probs=p).sample()
        # print('mu=%d'%mu)
        # print('b=%d'%b)
        
        if b == 0:
            return self.zero(x)
        
        else:
            return self.one(x)
        
        
class BatchWiseSARB(pl.LightningModule):
    def __init__(self, zero=BReLU(), one=torch.nn.ReLU(inplace=False)):
        super(BatchWiseSARB, self).__init__() #Stochastic Activation Relu or Brelu
        self.zero=zero
        self.one=one
        
    def forward(self, x):
        tmp = []
        mu = x.to(torch.float32).mean(dim=(1,2,3), keepdim=False)
        switch = torch.distributions.bernoulli.Bernoulli(logits=mu).sample()
        # print('mu=%d'%mu)
        # print('b=%d'%b)
        
        for b, batch in zip(switch, [x[j] for j in range(x.size(0))]):
            if b == 0:
                # tmp.append(self.zero(batch.clone()))
                tmp.append(self.zero(batch))
            
            else:
                # tmp.append(self.one(batch.clone()))
                tmp.append(self.one(batch))
        return torch.stack(tmp)
        
        
class BatchWiseSARBSC(pl.LightningModule):
    def __init__(self, zero=BReLU(), one=torch.nn.ReLU(inplace=False)):
        super(BatchWiseSARBSC, self).__init__() #Stochastic Activation Relu or Brelu
        self.zero=zero
        self.one=one
        
    def forward(self, x):
        tmp = []
        mu = x.to(torch.float32).mean(dim=(1,2,3), keepdim=False)
        p = -torch.sin(mu) * torch.cos(mu + (torch.pi/2))
        p = torch.where(p < 0, 0, p)
        p = torch.where(p > 1, 1, p)
        switch = torch.distributions.bernoulli.Bernoulli(probs=p).sample()
        # print('mu=%d'%mu)
        # print('b=%d'%b)
        
        for b, batch in zip(switch, [x[j] for j in range(x.size(0))]):
            if b == 0:
                # tmp.append(self.zero(batch.clone()))
                tmp.append(self.zero(batch))
            
            else:
                # tmp.append(self.one(batch.clone()))
                tmp.append(self.one(batch))
        return torch.stack(tmp)


class ImproveBatchWiseSARB(pl.LightningModule):
    def __init__(self, zero=BReLU(), one=torch.nn.ReLU(inplace=True)):
        super(ImproveBatchWiseSARB, self).__init__() #Stochastic Activation Relu or Brelu
        self.zero=zero
        self.one=one
        
    def forward(self, x):
        mu = x.to(torch.float32).mean(dim=(1,2,3), keepdim=True)

        b = torch.distributions.bernoulli.Bernoulli(logits=mu).sample()
        # print('mu=%d'%mu)
        # print('b=%d'%b)
        
        return self.zero((1-b) * x) + self.one(b * x)

    
class ImproveBatchWiseSARBSC(pl.LightningModule):
    def __init__(self, zero=BReLU(), one=torch.nn.ReLU(inplace=True)):
        super(ImproveBatchWiseSARBSC, self).__init__() #Stochastic Activation Relu or Brelu
        self.zero=zero
        self.one=one
        
    def forward(self, x):
        mu = x.to(torch.float32).mean(dim=(1,2,3), keepdim=True)
        p = -torch.sin(mu) * torch.cos(mu + (torch.pi/2))
        p = torch.where(p < 0, 0, p)
        p = torch.where(p > 1, 1, p)
        b = torch.distributions.bernoulli.Bernoulli(probs=p).sample()
        # print('mu=%d'%mu)
        # print('b=%d'%b)
        
        return self.zero((1-b) * x) + self.one(b * x)
    
    
class ChannelWiseSARB(pl.LightningModule):
    def __init__(self, zero=BReLU(), one=torch.nn.ReLU(inplace=True)):
        super(ChannelWiseSARB, self).__init__() #Stochastic Activation Relu or Brelu
        self.zero=zero
        self.one=one
        
    def forward(self, x):
        mu = x.to(torch.float32).mean(dim=(2,3), keepdim=True)

        b = torch.distributions.bernoulli.Bernoulli(logits=mu).sample()
        # print('mu=%d'%mu)
        # print('b=%d'%b)
        
        return self.zero((1-b) * x) + self.one(b * x)

    
class ChannelWiseSARBSC(pl.LightningModule):
    def __init__(self, zero=BReLU(), one=torch.nn.ReLU(inplace=True)):
        super(ChannelWiseSARBSC, self).__init__() #Stochastic Activation Relu or Brelu
        self.zero=zero
        self.one=one
        
    def forward(self, x):
        mu = x.to(torch.float32).mean(dim=(2,3), keepdim=True)
        p = -torch.sin(mu) * torch.cos(mu + (torch.pi/2))
        p = torch.where(p < 0, 0, p)
        p = torch.where(p > 1, 1, p)
        b = torch.distributions.bernoulli.Bernoulli(probs=p).sample()
        # print('mu=%d'%mu)
        # print('b=%d'%b)
        
        return self.zero((1-b) * x) + self.one(b * x)
    
class ReLUPlusBRelu(pl.LightningModule):
    def __init__(self):
        super(ReLUPlusBRelu, self).__init__() #Relu + brelu
        self.relu = nn.ReLU(inplace=True)
        self.brelu = BReLU()
        
    def forward(self, x):
        return self.relu(x) + self.brelu(x)
        


class TernaryOut(pl.LightningModule):
    '''
    2번 베르누이 샘플링을 진행한다. 
    샘플링 결과에 따라서 다음과 같이 출력한다.
    00 : 0
    01 : -1
    10 : -1
    11 : 1
    '''
    def __init__(self):
        super(TernaryOut, self).__init__()
        
    def forward(self, x):
        b = torch.distributions.bernoulli.Bernoulli(logits=x)
        b1 = b.sample()
        b2 = b.sample()

        epsilon = (-torch.logical_xor(b1, b2, out=torch.empty(b1.shape, dtype=x.dtype, device=x.device)) + torch.logical_and(b1, b2)) * (1/(x+1e-16))
        
        return x * epsilon


class TernaryOutSC(pl.LightningModule):
    '''
    2번 베르누이 샘플링을 진행한다. 
    샘플링 결과에 따라서 다음과 같이 출력한다.
    확률식으로 sin, cos 사용
    00 : 0
    01 : -1
    10 : -1
    11 : 1
    '''
    def __init__(self):
        super(TernaryOutSC, self).__init__()
        
    def forward(self, x):
        p = -torch.sin(x) * torch.cos(x + (torch.pi/2))
        p = torch.where(p < 0, 0, p)
        p = torch.where(p > 1, 1, p)
        
        b = torch.distributions.bernoulli.Bernoulli(probs=p)
        b1 = b.sample()
        b2 = b.sample()

        epsilon = (-torch.logical_xor(b1, b2, out=torch.empty(b1.shape, dtype=x.dtype, device=x.device)) + torch.logical_and(b1, b2)) * (1/(x+1e-16))
        
        return x * epsilon


class SomeTernaryOut(pl.LightningModule):
    '''
    2번 베르누이 샘플링을 진행한다. 
    샘플링 결과에 따라서 다음과 같이 출력한다.
    00 : 0
    01 : 1
    10 : -1
    11 : x
    '''
    def __init__(self):
        super(SomeTernaryOut, self).__init__()
        
    def forward(self, x):
        b = torch.distributions.bernoulli.Bernoulli(logits=x)
        b1 = b.sample()
        b2 = b.sample()
        
        epsilon = (torch.logical_xor(b1, b2, out=torch.empty(b1.shape, dtype=x.dtype, device=x.device)) * (1/(x+1e-8)) * (torch.ones_like(b1) - (2*b1))) + torch.logical_and(b1, b2)
        
        return x * epsilon


class SomeTernaryOutSC(pl.LightningModule):
    '''
    2번 베르누이 샘플링을 진행한다. 
    샘플링 결과에 따라서 다음과 같이 출력한다.
    00 : 0
    01 : 1
    10 : -1
    11 : x
    '''
    def __init__(self):
        super(SomeTernaryOutSC, self).__init__()
        
    def forward(self, x):
        p = -torch.sin(x) * torch.cos(x + (torch.pi/2))
        p = torch.where(p < 0, 0, p)
        p = torch.where(p > 1, 1, p)
        
        b = torch.distributions.bernoulli.Bernoulli(probs=p)
        b1 = b.sample()
        b2 = b.sample()
        
        epsilon = (torch.logical_xor(b1, b2, out=torch.empty(b1.shape, dtype=x.dtype, device=x.device)) * (1/(x+1e-8)) * (torch.ones_like(b1) - (2*b1))) + torch.logical_and(b1, b2)
        
        return x * epsilon


class TernaryMul(pl.LightningModule):
    '''
    베르누이 샘플링 결과에 따라 다음 값을 곱해서 출력한다.
    00 : 0 -> 0
    01 : -1 -> -x
    10 : -1 -> -x
    11 : 1 -> x
    '''
    def __init__(self):
        super(TernaryMul, self).__init__()
        
    def forward(self, x):
        b = torch.distributions.bernoulli.Bernoulli(logits=x)
        b1 = b.sample()
        b2 = b.sample()

        epsilon = -torch.logical_xor(b1, b2, out=torch.empty(b1.shape, dtype=x.dtype, device=x.device)) + torch.logical_and(b1, b2)

        return x * epsilon


class TernaryMulSC(pl.LightningModule):
    '''
    베르누이 샘플링 결과에 따라 다음 값을 곱해서 출력한다.
    00 : 0 -> 0
    01 : -1 -> -x
    10 : -1 -> -x
    11 : 1 -> x
    '''
    def __init__(self):
        super(TernaryMulSC, self).__init__()
        
    def forward(self, x):
        p = -torch.sin(x) * torch.cos(x + (torch.pi/2))
        p = torch.where(p < 0, 0, p)
        p = torch.where(p > 1, 1, p)
        
        b = torch.distributions.bernoulli.Bernoulli(probs=p)
        b1 = b.sample()
        b2 = b.sample()

        epsilon = -torch.logical_xor(b1, b2, out=torch.empty(b1.shape, dtype=x.dtype, device=x.device)) + torch.logical_and(b1, b2)
        
        return x * epsilon


class TernaryMulSym(pl.LightningModule):
    '''
    베르누이 샘플링 결과에 따라 다음 값을 곱해서 출력한다.
    00 : -1 -> -x
    01 : 0 -> 0
    10 : 0 -> 0
    11 : 1 -> x
    '''
    def __init__(self):
        super(TernaryMulSym, self).__init__()
        
    def forward(self, x):
        b = torch.distributions.bernoulli.Bernoulli(logits=x)
        b1 = b.sample()
        b2 = b.sample()

        epsilon = torch.logical_xor(b1, b2, out=torch.empty(b1.shape, dtype=x.dtype, device=x.device)) -torch.ones_like(b1) + 2 * torch.logical_and(b1, b2)
        
        return x * epsilon


class TernaryMulSymSC(pl.LightningModule):
    '''
    베르누이 샘플링 결과에 따라 다음 값을 곱해서 출력한다.
    00 : -1 -> -x
    01 : 0 -> 0
    10 : 0 -> 0
    11 : 1 -> x
    '''
    def __init__(self):
        super(TernaryMulSymSC, self).__init__()
        
    def forward(self, x):
        p = -torch.sin(x) * torch.cos(x + (torch.pi/2))
        p = torch.where(p < 0, 0, p)
        p = torch.where(p > 1, 1, p)
        
        b = torch.distributions.bernoulli.Bernoulli(probs=p)
        b1 = b.sample()
        b2 = b.sample()

        epsilon = torch.logical_xor(b1, b2, out=torch.empty(b1.shape, dtype=x.dtype, device=x.device)) -torch.ones_like(b1) + 2 * torch.logical_and(b1, b2)
        
        return x * epsilon

        
        
        
        
        
        

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
