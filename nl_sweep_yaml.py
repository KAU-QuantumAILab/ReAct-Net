import os
from torch import optim, nn, utils, Tensor
import torch
from torchvision.datasets import MNIST, CIFAR10, ImageNet, ImageFolder
from torchvision.transforms import ToTensor
import torchvision.transforms as transforms
# from torchvision import models
from custom_resnet import resnet
import lightning.pytorch as pl
from torchmetrics.functional import accuracy
from lightning.pytorch.loggers import WandbLogger, TensorBoardLogger 
from lightning.pytorch.callbacks import ModelCheckpoint, LearningRateMonitor
import torch.optim.lr_scheduler as lr_scheduler
<<<<<<< HEAD
from torch.optim.lr_scheduler import OneCycleLR, CosineAnnealingLR
from models import SamplingLayer, MLPMnist, Layer4Conv, Net
=======
from torch.optim.lr_scheduler import OneCycleLR
<<<<<<< HEAD
from models import SamplingLayer, MLPMnist, Layer4Conv, Net, Layer3Conv
=======
from models import SamplingLayer, LearnableSamplingLayer
>>>>>>> 71936f686f19c4941c35d2d67021d9dc3ace522a
>>>>>>> be889062aea6abda1e862b394fbb7cc77981c2b4
import wandb
import os
import argparse
import random
# from lightning.pytorch.accelerators import find_usable_cuda_devices
import torchattacks
import yaml


##################################################################################


parser = argparse.ArgumentParser(
    description='reAct sweep with yaml \n usage: nl_sweep_yaml.py --yaml [yaml_path] --devices 0 --project_name [pname] -- entity [ename]\n')

parser.add_argument('--yaml', required=True, help='yaml 파일 경로 입력')
<<<<<<< HEAD
parser.add_argument('--project_name', default="MLP_MNIST", help='wandb project name')
=======
<<<<<<< HEAD
parser.add_argument('--project_name', default="reAct_sweep_Tiny50", help='wandb project name')
=======
parser.add_argument('--project_name', default="reAct_sweep_MNIST_noFC", help='wandb project name')
>>>>>>> 71936f686f19c4941c35d2d67021d9dc3ace522a
>>>>>>> be889062aea6abda1e862b394fbb7cc77981c2b4
parser.add_argument('--entity', default='kau-quantum', help='wandb entity name')
parser.add_argument('--devices', default=0, type=int, help='choose the CUDA(ex: 0, 1, 2, -1)')

args = parser.parse_args()

torch.set_float32_matmul_precision('high')

global project_name, sweep_config, device_num

project_name = args.project_name        # wandb project name
entity = args.entity
device_num = [args.devices]

ypath = args.yaml

with open(ypath) as file:
    sweep_config = yaml.load(file, Loader=yaml.FullLoader)


# dataset = "CIFAR10"                     # CIFAR10 / ImageNet
# project_name = "reAct_sweep_ImageNet"            # wandb project name
# dataset = "ImageNet"

###################################################################################

resnet_models = {
    'resnet18' : resnet.resnet18,
    'resnet50' : resnet.resnet50,
    'resnet101' : resnet.resnet101
}

def getDataNormalization(dataset):
    if(dataset == 'CIFAR10'):
        return (0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)
    elif(dataset == 'ImageNet' or dataset == 'TinyImagenet'):
        return (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)
    elif(dataset == 'MNIST'):
        return (0.1307, ), (0.3081, )
<<<<<<< HEAD
    elif(dataset == 'TinyImagenet'):
        return (0.4802, 0.4481, 0.3975), (0.2302, 0.2265, 0.2262)
=======
>>>>>>> be889062aea6abda1e862b394fbb7cc77981c2b4

# class NoFCModel(pl.LightningModule):
#     def __init__(self, config):
#         super(NoFCModel, self).__init__()
#         self.config = config
#         self.config['num_classes'] = 10
#         if self.config["activation"] == 'relu':
#             self.module = nn.Sequential(
#                 nn.Conv2d(1, 3, 4, stride=2), #28 -> 13
#                 nn.ReLU(),
#                 nn.Conv2d(3, 3, 3), #13 -> 11
#                 nn.ReLU(),
#                 nn.Conv2d(3, 3, 3), #11 -> 9
#                 nn.ReLU(),
#                 nn.Conv2d(3, 3, 3), #9 -> 7
#                 nn.ReLU(),
#                 nn.Conv2d(3, 3, 3), #7 -> 5
#                 nn.ReLU(),
#                 nn.Conv2d(3, 3, 3), #5 -> 3
#                 nn.ReLU(),
#                 nn.Conv2d(3, 10, 3), #3 -> 1
#                 nn.Flatten()
#             )
#         else:
#             self.module = nn.Sequential(
#                 nn.Conv2d(1, 3, 4, stride=2), #28 -> 13
#                 nn.ReLU(),
#                 nn.Conv2d(3, 3, 3), #13 -> 11
#                 SamplingLayer(),
#                 nn.Conv2d(3, 3, 3), #11 -> 9
#                 nn.ReLU(),
#                 nn.Conv2d(3, 3, 3), #9 -> 7
#                 SamplingLayer(),
#                 nn.Conv2d(3, 3, 3), #7 -> 5
#                 nn.ReLU(),
#                 nn.Conv2d(3, 3, 3), #5 -> 3
#                 SamplingLayer(),
#                 nn.Conv2d(3, 10, 3), #3 -> 1
#                 nn.Flatten()
#             )
    
#     def forward(self, x):
#         return self.module(x)

    
    
# class MNISTModel(pl.LightningModule):
#     def __init__(self, config):
#         super(MNISTModel, self).__init__()
#         self.config = config
#         self.layer1 = nn.Sequential(
#             nn.Conv2d(1, 16, 5, 1, 2),
#             nn.BatchNorm2d(16),
#             nn.ReLU(inplace=True),
#             nn.Conv2d(16, 64, 3, 1, 1),
#             nn.BatchNorm2d(64),
#             nn.ReLU(inplace=True)
#         )
        
#         self.layer2 = nn.Sequential(
#             nn.Conv2d(64, 64, 3, 1, 1),
#             nn.BatchNorm2d(64)
#         )
        
#         self.layer3 = nn.Sequential(
#             nn.Conv2d(64, 128, 3, 1, 1),
#             nn.BatchNorm2d(128),
#             nn.ReLU(inplace=True)
#         )
        
#         self.layer4 = nn.Sequential(
#             nn.Conv2d(128, 128, 3, 1, 1),
#             nn.BatchNorm2d(128)
#         )
#         self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
#         self.fc = nn.Linear(128, 10)
#         self.config['num_classes'] = 10
#         if self.config['activation'] == 'sampling':
#             self.sampling = SamplingLayer()
#         else:
#             self.sampling = nn.ReLU(inplace=True)
        
#     def forward(self, x):
#         out1 = self.layer1(x)
        
#         out2 = self.layer2(out1)
#         out2 += out1
#         out2 = self.sampling(out2)
        
#         out3 = self.layer3(out2)
#         out4 = self.layer4(out3)
#         out4 += out3
#         out4 = self.sampling(out4)
        
#         y = self.avgpool(out4)
#         y = torch.flatten(y, 1)
#         y = self.fc(y)
        
#         return y
        
        
        
        
        

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
<<<<<<< HEAD
        elif(self.config["dataset"] == 'TinyImagenet'):
            self.config['num_classes'] = 200
=======
<<<<<<< HEAD
        elif(self.config["dataset"] == 'TinyImagenet'):
            self.config['num_classes'] = 200
=======
>>>>>>> 71936f686f19c4941c35d2d67021d9dc3ace522a
>>>>>>> be889062aea6abda1e862b394fbb7cc77981c2b4

        model = resnet_models[self.config['architecture']](weights=False, num_classes=self.config['num_classes'])
        if(self.config['dataset'] == 'CIFAR10' or self.config['dataset'] == 'TinyImagenet'):
            model.conv1 = nn.Conv2d(3, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
            model.maxpool = nn.Identity()
        elif(self.config['dataset'] == 'MNIST'):
            model.conv1 = nn.Conv2d(1, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
            model.maxpool = nn.Identity()

        if(activation == 'sampling'):
            for name,child in model.named_children():
                if(isinstance(child, nn.Sequential)):
                    for sub_name, sub_child in child.named_children():
                        sub_child.configure_react(SamplingLayer)

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
<<<<<<< HEAD
=======
<<<<<<< HEAD
>>>>>>> be889062aea6abda1e862b394fbb7cc77981c2b4
        if config["architecture"] == "MLP":
            self.encoder = MLPMnist(config)
        elif config["architecture"] == "conv":
            self.encoder = Layer4Conv(config)
        elif config["architecture"] == "Net":
            self.encoder = Net(config)
<<<<<<< HEAD
=======
        elif config["architecture"] == "3conv":
            self.encoder = Layer3Conv(config)
=======
        if config["architecture"] == "custom":
            self.encoder = NoFCModel(config)
>>>>>>> 71936f686f19c4941c35d2d67021d9dc3ace522a
>>>>>>> be889062aea6abda1e862b394fbb7cc77981c2b4
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
                weight_decay=5e-4,
            )
        elif(self.config['optimizer'] == "AdamW"):
            optimizer = torch.optim.AdamW(
                self.parameters(),
                lr=self.config["lr"],
                betas=(self.config.beta1, self.config.beta2)
            )
        
        if(self.config["lr_scheduler"]):
            steps_per_epoch = self.config["dataloader_len"]
            scheduler_dict = {
                "scheduler": OneCycleLR(
                    optimizer,
                    0.1,
                    epochs=self.config["epochs"],
                    steps_per_epoch=steps_per_epoch,
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
    

    def generateAdv(self, x, y, eps = 0.0314, alpha=0.00784, steps=3):
        with torch.enable_grad():
            atk = torchattacks.PGD(self.encoder, eps=eps, alpha=alpha, steps=steps)
            adv_images = atk(x, y)
        return adv_images


    def evaluateRobust(self, x, y):
<<<<<<< HEAD
        adv_images = self.generateAdv(x, y, self.config['eps'])
=======
<<<<<<< HEAD
        adv_images = self.generateAdv(x, y, self.config['eps'])
=======
        adv_images = self.generateAdv(x, y, eps=0.05)
>>>>>>> 71936f686f19c4941c35d2d67021d9dc3ace522a
>>>>>>> be889062aea6abda1e862b394fbb7cc77981c2b4
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


    elif config["dataset"] == "ImageNet":
        transform = transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.CenterCrop((224,224)),
            transforms.ToTensor(),
        ])

        data_raw = ImageFolder('~/dataset/ImageNet/2012/ILSVRC2012_img_train', transform=transform)
        trainset, testset = torch.utils.data.random_split(data_raw, [0.9, 0.1])
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
        
<<<<<<< HEAD
    elif config["dataset"] == "TinyImagenet":
        transform = transforms.Compose([
            transforms.ToTensor(),
        ])
        data_raw = ImageFolder('./data/tiny-imagenet-200/train', transform=transform)
        trainset, testset = torch.utils.data.random_split(data_raw, [0.9, 0.1])
        trainloader = torch.utils.data.DataLoader(trainset, batch_size=config['batch_size'], shuffle=True, num_workers =config['num_workers'])
        testloader = torch.utils.data.DataLoader(testset, batch_size=config['batch_size'], shuffle=False, num_workers =config['num_workers'])
        data = (trainloader, testloader)
        
=======
>>>>>>> 71936f686f19c4941c35d2d67021d9dc3ace522a
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



def train_model():
    run = wandb.init(project=project_name, entity=entity)
    config = wandb.config
<<<<<<< HEAD
    wandb.define_metric("val_acc", summary="max")
    wandb.define_metric("Robust_acc", summary="max")
    # name_postfix = "reference" if config['activation'] == 'relu' else "ReAct"
    name_postfix = config['activation']
    adver = "-adv" + 'eps:' + str(config['eps']) if config['adv'] else ''
=======
    name_postfix = "reference" if config['activation'] == 'relu' else "ReAct"
    adver = "-adv" if config['adv'] else ''
<<<<<<< HEAD
    name = config["dataset"] + "-" + config["architecture"] + "-" + name_postfix + "-" + config.optimizer + " lr:" + str(round(config.lr, 4)) + adver  + 'eps:' + str(config['eps'])
=======
>>>>>>> be889062aea6abda1e862b394fbb7cc77981c2b4
    name = config["dataset"] + "-" + config["architecture"] + "-" + name_postfix + "-" + config.optimizer + " lr:" + str(round(config.lr, 4)) + adver
>>>>>>> 71936f686f19c4941c35d2d67021d9dc3ace522a
    run.name = name
    wandb_logger = WandbLogger(config=config, save_code=True, log_model="all")

    data = choose_dataset(config=config)


    config.dataloader_len = len(data[0])

    modified_resnet_encoder = LitAutoEncoder(config)

    wandb_logger.watch(modified_resnet_encoder, log="all")

    lr_monitor = LearningRateMonitor(logging_interval='step')
    checkpoint_callback = ModelCheckpoint(monitor="val_acc", mode="max")
    # trainer = pl.Trainer(max_epochs = config["epochs"],logger= wandb_logger, callbacks=[checkpoint_callback,lr_monitor], devices = find_usable_cuda_devices(1))
    trainer = pl.Trainer(max_epochs = config["epochs"],logger= wandb_logger, callbacks=[checkpoint_callback,lr_monitor], devices = device_num)
    trainer.fit(model=modified_resnet_encoder, train_dataloaders=data[0], val_dataloaders=data[1])

    wandb.finish()



def main():
    sweep_id = wandb.sweep(sweep_config, project=project_name)
<<<<<<< HEAD
    # sweep_id = "gt3qp3cj"
=======
<<<<<<< HEAD
    # sweep_id = "wrpr3elw"
=======
>>>>>>> 71936f686f19c4941c35d2d67021d9dc3ace522a
>>>>>>> be889062aea6abda1e862b394fbb7cc77981c2b4
    wandb.agent(sweep_id=sweep_id, function=train_model, project=project_name, entity=entity)


main()