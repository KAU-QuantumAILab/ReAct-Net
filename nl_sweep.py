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
from torch.optim.lr_scheduler import OneCycleLR
from models import SamplingLayer, LearnableSamplingLayer
import wandb
import os
import argparse
import random
# from pl_bolts.datamodules import CIFAR10DataModule, MNISTDataModule, ImagenetDataModule
from lightning.pytorch.accelerators import find_usable_cuda_devices

##################################################################################
################################# config #########################################
##################################################################################

# sweep_config = {
#     'method': 'bayes',
#     'name': 'reaAct_sweep_test',
#     'metric':{
#         'goal': 'minimize',
#         'name': 'val_loss'
#     },
#     'parameters': {
#         'lr': {'min': 0.0001, 'max': 0.1},
#         'activation': {'values': ['sampling', 'relu']},
#         'optimizer': {'values': ['SGD', 'Adam', 'AdamW']},
#         'architecture': {'values': ['resnet18', 'resnet50', 'resnet101']},
#         'batch_size': {'min': 32, 'max':512},
#         'num_workers': {'min': 10, 'max': 40},
#         'lr_scheduler': {'values': [True, False]},
#         'epochs': {'value': 200}
#     }
# }

# sweep_config = {
#     'method': 'grid',
#     'name': 'reaAct_sweep_test',
#     'metric':{
#         'goal': 'minimize',
#         'name': 'val_loss'
#     },
#     'parameters': {
#         'lr': {'values':[0.1, 0.01, 0.001, 0.0001, 0.05, 0.005]},
#         'activation': {'values': ['sampling']},
#         'optimizer': {'values': ['SGD', 'Adam', 'AdamW']},
#         'beta1' : {'values':[0.9, 0.8, 0.7]},
#         'beta2' : {'values':[0.999, 0.9, 0.8, 0.7]},
#         'architecture': {'values': ['resnet18']},
#         'batch_size': {'value': 512},
#         'num_workers': {'value': 24},
#         'lr_scheduler': {'values': [True, False]},
#         'epochs': {'value': 200}
#     }
# }

sweep_config = {
    'method': 'grid',
    'name': 'reaAct_sweep_CIFAR10',
    'metric':{
        'goal': 'minimize',
        'name': 'val_loss'
    },
    'parameters': {
        'lr': {'values':[0.1, 0.01, 0.001, 0.0001, 0.05, 0.005]},
        'activation': {'values': ['relu', 'sampling']},
        'optimizer': {'values': ['SGD']},
        'momentum': {'values':[0, 0.9, 0.8, 0.5, 0.1]},
        'architecture': {'values': ['resnet18']},
        'batch_size': {'value': 256},
        'num_workers': {'value': 24},
        'lr_scheduler': {'values': [False, True]},
        'epochs': {'value': 200}
    }
}
project_name = "reAct_sweep_CIFAR10"            # wandb project name
dataset = "CIFAR10"                     # CIFAR10 / ImageNet
# project_name = "reAct_sweep_ImageNet"            # wandb project name
# dataset = "ImageNet"

###################################################################################

resnet_models = {
    'resnet18' : resnet.resnet18,
    'resnet50' : resnet.resnet50,
    'resnet101' : resnet.resnet101
}


class CustomModel(pl.LightningModule):
    def __init__(self):
        super(CustomModel, self).__init__()
        self.sequenceModule = nn.Sequential(
            nn.Conv2d(1, 3, 4, stride=2), #28 -> 13
            nn.ReLU(),
            nn.Conv2d(3, 3, 3), #13 -> 11
            nn.ReLU(),
            nn.Conv2d(3, 3, 3), #11 -> 9
            nn.ReLU(),
            nn.Conv2d(3, 3, 3), #9 -> 7
            nn.ReLU(),
            nn.Conv2d(3, 3, 3), #7 -> 5
            nn.ReLU(),
            nn.Conv2d(3, 3, 3), #5 -> 3
            nn.ReLU(),
            nn.Conv2d(3, 10, 3), #3 -> 1
            nn.Flatten()
        )
    def forward(self, x):
        return self.sequenceModule(x)

def create_model(config):
    if(dataset == 'CIFAR10' or dataset == 'MNIST'):
        config.num_classes = 10
    elif(dataset == 'ImageNet'):
        config.num_classes = 1000
    model = resnet_models[config['architecture']](weights=False, num_classes=config.num_classes)
    if(dataset != 'ImageNet'):
        model.conv1 = nn.Conv2d(3, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
        model.maxpool = nn.Identity()

    if(config.activation == 'sampling'):
        for name,child in model.named_children():
            if(isinstance(child, nn.Sequential)):
                for sub_name, sub_child in child.named_children():
                    sub_child.configure_react(SamplingLayer)
    
    return model

# define the LightningModule
class LitAutoEncoder(pl.LightningModule):
    def __init__(self, config):
        super().__init__()
        self.config = config
        # self.save_hyperparameters() # sweep 오류시 제거

        if (self.config['architecture'] == "custom"):
            self.encoder = CustomModel()
        else: 
            self.encoder = create_model(self.config)
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
            steps_per_epoch = 45000 // self.config["batch_size"]
            scheduler_dict = {
                "scheduler": OneCycleLR(
                    optimizer,
                    0.1,
                    epochs=self.config["epochs"],
                    steps_per_epoch=steps_per_epoch,
                ),
                "interval": "step",
            }
            return {"optimizer": optimizer, "lr_scheduler" : scheduler_dict, "monitor": "val_acc"}
        else:
            return {"optimizer": optimizer, "monitor": "val_acc"}
    
    def evaluate(self, batch, stage=None):
        x, y = batch
        logits = self.encoder(x)
        loss = nn.functional.cross_entropy(logits, y)
        preds = torch.argmax(logits, dim=1)
        acc = accuracy(preds, y, num_classes=self.config.num_classes, task="multiclass")

        if stage:
            self.log(f"{stage}_loss", loss, prog_bar=True, sync_dist=True)
            self.log(f"{stage}_acc", acc, prog_bar=True, sync_dist=True)
        return loss

    def validation_step(self, batch, batch_idx):
        return self.evaluate(batch, "val")

    def test_step(self, batch, batch_idx):
        return self.evaluate(batch, "test")


def choose_dataset(config, dataset_name = "CIFAR10"):
    if dataset_name == "CIFAR10":
        train_transform = transforms.Compose(
            [
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        ])

        test_transform = transforms.Compose(
            [
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
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


    elif dataset_name == "ImageNet":
        transform = transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.CenterCrop((224,224)),
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        ])

        data_raw = ImageFolder('~/dataset/ImageNet/2012/ILSVRC2012_img_train', transform=transform)
        trainset, testset = torch.utils.data.random_split(data_raw, [0.9, 0.1])
        trainloader = torch.utils.data.DataLoader(trainset, batch_size=config['batch_size'], shuffle=True, num_workers =config['num_workers'])

        # testset = ImageFolder('dataset/ImageNet/2012/ILSVRC2012_img_val', transform=transform)
        testloader = torch.utils.data.DataLoader(testset, batch_size=config['batch_size'], shuffle=False, num_workers =config['num_workers'])
        data = (trainloader, testloader)
    
    return data

def train_model():
    run = wandb.init(project=project_name, entity='kau-quantum')
    config = wandb.config
    name_postfix = "reference" if config['activation'] == 'relu' else "ReAct"
    name = dataset + "-" + config["architecture"] + "-" + name_postfix + "-" + config.optimizer + " lr:" + str(round(config.lr, 4))
    run.name = name
    wandb_logger = WandbLogger(project=project_name, entity='kau-quantum',
        config=config, save_code=True, log_model="all", name=name)

    data = choose_dataset(config=config, dataset_name=dataset)

    modified_resnet_encoder = LitAutoEncoder(config)

    wandb_logger.watch(modified_resnet_encoder, log="all")

    lr_monitor = LearningRateMonitor(logging_interval='step')
    checkpoint_callback = ModelCheckpoint(monitor="val_acc", mode="max")
    trainer = pl.Trainer(max_epochs = config["epochs"],logger= wandb_logger, callbacks=[checkpoint_callback,lr_monitor])
    trainer.fit(model=modified_resnet_encoder, train_dataloaders=data[0], val_dataloaders=data[1])



if __name__ == '__main__':
    sweep_id = wandb.sweep(sweep_config, project=project_name)
    wandb.agent(sweep_id=sweep_id, function=train_model)