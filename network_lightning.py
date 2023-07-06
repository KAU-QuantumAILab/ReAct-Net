# %%
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
torch.set_float32_matmul_precision('high')

TESTMODE = False

config = {
    "optimizer" : "Adam",
    "lr": 0.1,
    "architecture": "resnet50",
    "dataset": "CIFAR10",
    "epochs": 200,
    "batch_size" : 128,
    'activation' : 'relu',
    "num_workers" : int(os.cpu_count() / 2)
    }

resnet_models = {
    'resnet18' : resnet.resnet18,
    'resnet50' : resnet.resnet50,
    'resnet101' : resnet.resnet101
}

name_postfix = "reference" if config['activation'] == 'relu' else "ReAct"

if(not TESTMODE):
    wandb_logger = WandbLogger(project='Sampling_Network',
        config=config, save_code=True, log_model="all", name=config["dataset"] + "-" + config["architecture"] + "-" + name_postfix)

if(config["dataset"] == 'CIFAR10' or config["dataset"] == 'MNIST'):
    num_classes = 10
elif(config["dataset"] == 'ImageNet'):
    num_classes = 1000

# %%


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

def create_model(activation):
    model = resnet_models[config['architecture']](weights=False, num_classes=10)
    model.conv1 = nn.Conv2d(3, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
    model.maxpool = nn.Identity()

    if(activation == 'sampling'):
        for name,child in model.named_children():
            if(isinstance(child, nn.Sequential)):
                for sub_name, sub_child in child.named_children():
                    sub_child.bottleneckLayer = SamplingLayer()
            # if isinstance(child,nn.ReLU):
            #     model._modules['bottleneckLayer'] = SamplingLayer()
    return model

# define the LightningModule
class LitAutoEncoder(pl.LightningModule):
    def __init__(self, activation):
        super().__init__()
        self.save_hyperparameters()

        if (config['architecture'] == "custom"):
            self.encoder = CustomModel()
        else: 
            self.encoder = create_model(activation)
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
        if(config['optimizer'] == "Adam"):
            optimizer = torch.optim.Adam(
                self.parameters(),
                lr=config["lr"],
            )
        elif(config['optimizer'] == "SGD"):
            optimizer = torch.optim.SGD(
                self.parameters(),
                lr=config["lr"],
                momentum=0.9,
                weight_decay=5e-4,
            )
        elif(config['optimizer'] == "AdamW"):
            optimizer = torch.optim.AdamW(
                self.parameters(),
                lr=config["lr"]
            )
        steps_per_epoch = 45000 // config["batch_size"]
        scheduler_dict = {
            "scheduler": OneCycleLR(
                optimizer,
                0.1,
                epochs=config["epochs"],
                steps_per_epoch=steps_per_epoch,
            ),
            "interval": "step",
        }
        return {"optimizer": optimizer, "lr_scheduler" : scheduler_dict, "monitor": "val_acc"}
    
    def evaluate(self, batch, stage=None):
        x, y = batch
        logits = self.encoder(x)
        loss = nn.functional.cross_entropy(logits, y)
        preds = torch.argmax(logits, dim=1)
        acc = accuracy(preds, y, num_classes=num_classes, task="multiclass")

        if stage:
            self.log(f"{stage}_loss", loss, prog_bar=True, sync_dist=True)
            self.log(f"{stage}_acc", acc, prog_bar=True, sync_dist=True)

    def validation_step(self, batch, batch_idx):
        self.evaluate(batch, "val")

    def test_step(self, batch, batch_idx):
        self.evaluate(batch, "test")

# init the autoencoder
modified_resnet_encoder = LitAutoEncoder(config['activation'])

if(not TESTMODE):
    wandb_logger.watch(modified_resnet_encoder, log="all")
else:
    tb_logger = TensorBoardLogger(save_dir="logs/")

# %%

if(config["dataset"]=="CIFAR10"):
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
                                            
elif(config["dataset"]=="ImageNet"):
    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.CenterCrop((224,224)),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    ])

    data_raw = ImageFolder('~/dataset/ImageNet/2012/ILSVRC2012_img_train', transform=transform)
    trainset, testset = torch.utils.data.random_split(data_raw, [0.9, 0.1])
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=config['batch_size'], shuffle=True, num_workers = 15)

    # testset = ImageFolder('dataset/ImageNet/2012/ILSVRC2012_img_val', transform=transform)
    testloader = torch.utils.data.DataLoader(testset, batch_size=config['batch_size'], shuffle=False, num_workers = 15)

# %%
# train the model (hint: here are some helpful Trainer arguments for rapid idea iteration)
# torch.set_float32_matmul_precision('medium')

lr_monitor = LearningRateMonitor(logging_interval='step')
checkpoint_callback = ModelCheckpoint(monitor="val_acc", mode="max")

trainer = pl.Trainer(max_epochs = config["epochs"],logger= tb_logger if TESTMODE else wandb_logger, callbacks=[checkpoint_callback,lr_monitor])
trainer.fit(model=modified_resnet_encoder, train_dataloaders=trainloader, val_dataloaders=testloader)
# trainer.test(model=modified_resnet_encoder,dataloaders=testloader)