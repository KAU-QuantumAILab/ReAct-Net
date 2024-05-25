import os
from torch import optim, nn, utils, Tensor
import torch
from torchvision.datasets import MNIST, CIFAR10, ImageFolder
from torchvision.transforms import ToTensor
import torchvision.transforms as transforms
# from torchvision import models
from custom_resnet import resnet
import lightning.pytorch as pl
from torchmetrics.functional import accuracy
from lightning.pytorch.loggers import WandbLogger, TensorBoardLogger 
from lightning.pytorch.callbacks import ModelCheckpoint, LearningRateMonitor, EarlyStopping
import torch.optim.lr_scheduler as lr_scheduler
from torch.optim.lr_scheduler import OneCycleLR
from models import *
import wandb
import os
import argparse
import random
# from lightning.pytorch.accelerators import find_usable_cuda_devices
from torchattacks import FGSM, PGD
import yaml
import numpy as np
from torch.nn import GELU, SiLU, ELU, LeakyReLU, PReLU
from torch.utils.data import Dataset


def seed_everything(seed:int = 1004):
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)  # current gpu seed
    torch.cuda.manual_seed_all(seed) # All gpu seed
    torch.backends.cudnn.deterministic = True  # type: ignore
    torch.backends.cudnn.benchmark = False  # True로 하면 gpu에 적합한 알고리즘을 선택함.


resnet_models = {
    'resnet18' : resnet.resnet18,
    'resnet50' : resnet.resnet50,
    'resnet101' : resnet.resnet101
}

def getDataNormalization(dataset):
    if(dataset == 'CIFAR10'):
        return (0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)
    elif(dataset == 'ImageNet'):
        return (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)
    elif(dataset == 'MNIST'):
        return (0.1307, ), (0.3081, )
    elif(dataset == 'TinyImagenet'):
        return (0.4802, 0.4481, 0.3975), (0.2302, 0.2265, 0.2262)
    elif(dataset == 'ImageNet100'):
        return (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)


class ModelWrapper(pl.LightningModule):
    def __init__(self, config):
        super(ModelWrapper, self).__init__()
        self.config = config
        self.cnn = self.create_model(self.config['activation'])
        mean, std = getDataNormalization(self.config["dataset"])
        self.normalization = transforms.Normalize(mean, std)

    def create_model(self, activation):
        if(self.config["dataset"] == 'CIFAR10' or self.config["dataset"] == 'MNIST'):
            self.config['num_classes'] = 10
        elif(self.config["dataset"] == 'ImageNet'):
            self.config['num_classes'] = 1000
        elif(self.config["dataset"] == 'TinyImagenet'):
            self.config['num_classes'] = 200
        elif(self.config["dataset"] == 'ImageNet100'):
            self.config['num_classes'] = 100

        model = resnet_models[self.config['architecture']](weights=False, num_classes=self.config['num_classes'])
        if(self.config['dataset'] == 'CIFAR10' or self.config['dataset'] == 'TinyImagenet'):
            model.conv1 = nn.Conv2d(3, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
            model.maxpool = nn.Identity()
        elif(self.config['dataset'] == 'MNIST'):
            model.conv1 = nn.Conv2d(1, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
            model.maxpool = nn.Identity()


        activation_functions = {
            'gelu' : GELU,
            'silu' : SiLU,
            'elu' : ELU,
            'LReLU' : LeakyReLU,
            'PReLU' : PReLU,
            'sampling' : SamplingLayer,
            'brelu' : BReLU,
            'Vbrelu' : VariableBReLU,
            'leakyVbrelu' : LeakyVariableBReLU,
            'PVbrelu' : PVariableBReLU,
            'leaky_brelu' : Leaky_BReLU,
            'pbrelu' : PBReLU,
            'belu' : BELU,
            'vbelu' : VariableBELU,
            'SSCA' : SSCA,
            'SARB' : SARB,
            'SARBSC' : SARBSC,
            'BatchWiseSARB' : BatchWiseSARB,
            'BatchWiseSARBSC' : BatchWiseSARBSC,
            'ChannelWiseSARB' : ChannelWiseSARB,
            'ChannelWiseSARBSC' : ChannelWiseSARBSC,
            'ReLUPlusBRelu' : ReLUPlusBRelu,
            'MAct' : MeasureAct,
            'MActSC' : MeasureActSC,
            'SMAct' : SomeMeasureAct,
            'SMActSC' : SomeMeasureActSC,
            'TernaryOut' : TernaryOut,
            'TernaryOutSC' : TernaryOutSC,
            'SomeTernaryOut' : SomeTernaryOut,
            'SomeTernaryOutSC' : SomeTernaryOutSC,
            'TernaryMul' : TernaryMul,
            'TernaryMulSC' : TernaryMulSC,
            'TernaryMulSym' : TernaryMulSym,
            'TernaryMulSymSC' : TernaryMulSymSC,
        }
        
        if activation == 'relu':
            pass
        
        elif activation == 'Vbrelu' or activation == 'leakyVbrelu' or activation == 'PVbrelu' or activation == 'vbelu':
            for name,child in model.named_children():
                if(isinstance(child, nn.Sequential)):
                    for sub_name, sub_child in child.named_children():
                        sub_child.configure_react(activation_functions[self.config['activation']], replaceAll=self.config.get('replaceAll'), alpha=self.config.get('alpha'))
        
        else:
            for name,child in model.named_children():
                if(isinstance(child, nn.Sequential)):
                    for sub_name, sub_child in child.named_children():
                        sub_child.configure_react(activation_functions[self.config['activation']], replaceAll=self.config.get('replaceAll'))
        
        
        return model

    def forward(self, x):
        x = self.normalization(x)
        return self.cnn(x)



# define the LightningModule
class LitAutoEncoder(pl.LightningModule):
    def __init__(self, config):
        super().__init__()
        self.config = config
        # self.save_hyperparameters() # sweep 오류시 제거
        if config["architecture"] == "MLP":
            self.encoder = MLPMnist(config)
        elif config["architecture"] == "conv":
            self.encoder = Layer4Conv(config)
        elif config["architecture"] == "Net":
            self.encoder = Net(config)
        else:
            self.encoder = ModelWrapper(config)
        print(self.encoder)

    def training_step(self, batch, batch_idx):
        # training_step defines the train loop.
        # it is independent of forward
        x, y = batch
        z = self.encoder(x)
        loss = nn.functional.cross_entropy(z, y)
        # Logging to TensorBoard (if installed) by default
        self.log("train_loss", loss)
        # print(loss)

        if(self.config['adv']):
            advIdx = torch.randint(x.shape[0], (int(x.shape[0] * 0.2),))
            advExample = self.generateAdv(x[advIdx], y[advIdx])
            advZ = self.encoder(advExample)
            advLoss = nn.functional.cross_entropy(advZ, y[advIdx])
            return loss + advLoss
        else:
            return loss

    def configure_optimizers(self):
        if(self.config['optimizer'] == "Adam"):
            optimizer = torch.optim.Adam(
                self.parameters(),
                lr=self.config["lr"],
                betas=(self.config.beta1, self.config.beta2)
            )
        elif(self.config['optimizer'] == "SGD"):
            optimizer = torch.optim.SGD(
                self.parameters(),
                lr=self.config["lr"],
                momentum=self.config["momentum"],
                weight_decay=self.config['wd'],
            )
        elif(self.config['optimizer'] == "AdamW"):
            optimizer = torch.optim.AdamW(
                self.parameters(),
                lr=self.config["lr"],
                betas=(self.config.beta1, self.config.beta2)
            )
        
        if(self.config["lr_scheduler"]):
            # momentum = self.config.get('momentum') if self.config['optimizer'] == "SGD" else self.config.get('beta1')
            momentum = self.config.get('momentum') if self.config['optimizer'] == "SGD" else 0.85
            steps_per_epoch = self.config["dataloader_len"]
            scheduler_dict = {
                "scheduler": OneCycleLR(
                    optimizer,
                    max_lr=self.config['lr'],
                    epochs=self.config["epochs"],
                    steps_per_epoch=steps_per_epoch,
                    base_momentum=momentum,
                ),
                "interval": "step",
            }
            # scheduler_dict = {
            #     "scheduler": CosineAnnealingLR(
            #         optimizer,
            #         100,
            #     ),
            #     "interval": "epoch",
            # }
            return {"optimizer": optimizer, "lr_scheduler" : scheduler_dict, "monitor": "val_acc"}
        else:
            return {"optimizer": optimizer, "monitor": "val_acc"}
    

    def generateAdv(self, x, y, atkType = 'PGD', eps = 0.0314, alpha=0.00784, steps=3):
        with torch.enable_grad():
            if atkType == 'PGD':
                atk = PGD(self.encoder, eps=eps, alpha=alpha, steps=steps)
            elif atkType == 'FGSM':
                atk = FGSM(self.encoder, eps=eps)
            adv_images = atk(x, y)
        return adv_images


    def evaluateRobust(self, x, y):
        atkType = self.config.get('atk') if self.config.get('atk') is not None else 'PGD'
        atk_step = self.config.get('steps') if self.config.get('steps') is not None else 3
        adv_images = self.generateAdv(x=x, y=y, atkType=atkType, eps = self.config['eps'], steps=atk_step)
        logits = self.encoder(adv_images)
        preds = torch.argmax(logits, dim=1)
        acc = accuracy(preds, y, num_classes=self.config["num_classes"], task="multiclass")
        self.log("Robust_acc", acc, prog_bar=True, sync_dist=True)


    def evaluate(self, batch, stage=None):
        x, y = batch
        logits = self.encoder(x)
        loss = nn.functional.cross_entropy(logits, y)
        preds = torch.argmax(logits, dim=1)
        acc = accuracy(preds, y, num_classes=self.config["num_classes"], task="multiclass")

        if stage:
            self.log(f"{stage}_loss", loss, prog_bar=True, sync_dist=True)
            self.log(f"{stage}_acc", acc, prog_bar=True, sync_dist=True)
            if self.config["adv"]:
                self.evaluateRobust(x, y)

        return loss

    def validation_step(self, batch, batch_idx):
        return self.evaluate(batch, "val")

    def test_step(self, batch, batch_idx):
        return self.evaluate(batch, "test")


def choose_dataset(config):
    if config["dataset"] == "CIFAR10":
        train_transform = transforms.Compose(
            [
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
        ])

        test_transform = transforms.Compose(
            [
            transforms.ToTensor(),
        ])
            
        trainset = CIFAR10(root='~/data', train=True,
                                                download=True, transform=train_transform)
        trainloader = torch.utils.data.DataLoader(trainset, batch_size=config['batch_size'],
                                                shuffle=True, num_workers = config['num_workers'])

        testset = CIFAR10(root='~/data', train=False,
                                            download=True, transform=test_transform)
        testloader = torch.utils.data.DataLoader(testset, batch_size=config['batch_size'],
                                                shuffle=False, num_workers = config['num_workers'])
        
        data = (trainloader, testloader)
        
    
    elif config["dataset"] == "ImageNet100":
        transform = transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.CenterCrop((224,224)),
            transforms.ToTensor(),
        ])
        
        trainset = ImageFolder('~/data/ImageNet100/train', transform=transform)
        testset = ImageFolder('~/data/ImageNet100/val', transform=transform)
        
        trainloader = torch.utils.data.DataLoader(trainset, batch_size=config['batch_size'], shuffle=True, num_workers =config['num_workers'])
        testloader = torch.utils.data.DataLoader(testset, batch_size=config['batch_size'], shuffle=False, num_workers =config['num_workers'])
        
        data = (trainloader, testloader)


    elif config["dataset"] == "ImageNet":
        transform = transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.CenterCrop((224,224)),
            transforms.ToTensor(),
        ])

        trainset = ImageFolder('~/data/ImageNet/2012/ILSVRC2012_img_train', transform=transform)
        testset = ImageFolder('~/data/ImageNet/2012/ILSVRC2012_img_val', transform=transform)
        
        trainloader = torch.utils.data.DataLoader(trainset, batch_size=config['batch_size'], shuffle=True, num_workers =config['num_workers'])

        # testset = ImageFolder('dataset/ImageNet/2012/ILSVRC2012_img_val', transform=transform)
        testloader = torch.utils.data.DataLoader(testset, batch_size=config['batch_size'], shuffle=False, num_workers =config['num_workers'])
        
        data = (trainloader, testloader)

    elif config["dataset"] == "TinyImagenet":
        transform = transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.RandomResizedCrop(64),
            transforms.ToTensor(),

        ])
        data_raw = ImageFolder('~/data/tiny-imagenet-200/train', transform=transform)
        trainset, testset = torch.utils.data.random_split(data_raw, [0.9, 0.1])
        trainloader = torch.utils.data.DataLoader(trainset, batch_size=config['batch_size'], shuffle=True, num_workers =config['num_workers'])
        testloader = torch.utils.data.DataLoader(testset, batch_size=config['batch_size'], shuffle=False, num_workers =config['num_workers'])
        data = (trainloader, testloader)
        
    elif(config["dataset"]=="MNIST"):
        transform = transforms.Compose(
            [transforms.ToTensor(),
        ])

        trainset = MNIST(root='~/data', train=True,
                                                download=True, transform=transform)
        trainloader = torch.utils.data.DataLoader(trainset, batch_size=config['batch_size'],
                                                shuffle=True, num_workers = config['num_workers'])

        testset = MNIST(root='~/data', train=False,
                                            download=True, transform=transform)
        testloader = torch.utils.data.DataLoader(testset, batch_size=config['batch_size'],
                                                shuffle=False, num_workers = config['num_workers'])
        data = (trainloader, testloader)
    
    return data



def load_model(ckpt, config):
    model = LitAutoEncoder.load_from_checkpoint(ckpt, config=config)
    return model


from PIL.Image import Image

class TransformSubset(Dataset):
    def __init__(self, subset):
        self.subset = subset
        self.transform = randintchange()
        
        img = subset[0][0]
        if not isinstance(img, Image):
            size = img.shape[-1]
        
        else:
            size = img.size[-1]
        
        self.record = {}
        for i in range(len(self.subset)):
            self.record[i] = dict(
                cor = tuple(torch.randint(0, size, (2, ))),
                rgb = tuple(torch.randint(0, 256, (3, )))
                # rgb = (255, 255, 255)
            )
        
    def __getitem__(self, index):
        x, y = self.subset[index]
        if self.transform:
            x = self.transform(x, self.record[index])
        return x, y
        
    def __len__(self):
        return len(self.subset)
    

class randintchange(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.topil = transforms.ToPILImage()
        self.totensor = transforms.ToTensor()
        
    def forward(self, img, task):
        if not isinstance(img, Image):
            img = self.topil(img)
        

        cor = task['cor']
        rgb = task['rgb']
        
        img.putpixel(cor, rgb)
        
        return self.totensor(img)
    

from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import ToPILImage
import matplotlib.pyplot as plt

def ltol(loader):
    dataset = loader.dataset
    transformed_dataset = TransformSubset(dataset)
    return DataLoader(transformed_dataset, batch_size=1, shuffle=False, num_workers=1)


def test():
    data_config = {
        "dataset" : "CIFAR10",
        "batch_size" : 1,
        "num_workers" : 1
    }

    _, valloader = choose_dataset(data_config)
    
    
    im = ToPILImage()

    imgs = []

    for i in range(0, 5):
        sample = next(iter(valloader))[0][0]
        imgs.append(im(sample))
        print(f"{i} attack image append")
        valloader = ltol(valloader)

        print(f"{i}th loop end\n")
        
        
    for img in imgs:
        plt.imshow(img)
        plt.axis('off')  # 축 제거 (선택 사항)
        plt.show()