
import torch
import lightning as L
import lightning.pytorch as pl
from torch.autograd import Variable
from custom_resnet import resnet
from torch import optim, nn, Tensor
from torch.utils.data import DataLoader, random_split
import torch.optim.lr_scheduler as lr_scheduler
from torch.optim.lr_scheduler import OneCycleLR
import torchvision.transforms as transforms
from torchvision.datasets import MNIST, CIFAR10, ImageNet, ImageFolder
from torchmetrics.functional import accuracy
from racun import SamplingLayer
import torchattacks
import utils

class CustomDataModule(L.LightningDataModule):
    def __init__(self, dataset, classes=1000, batchsize=128, num_workers=20):
        super().__init__()
        self.dataset = dataset
        self.batchsize = batchsize
        self.num_workers = num_workers
        self.num_classes = classes if dataset == 'ImageNet' else 10
        self.input_ch = 1 if dataset=='MNIST' else 3

    def prepare_data(self):
        if(self.dataset=="CIFAR10"):
            CIFAR10(root='~/data', train=True,download=True)
            CIFAR10(root='~/data', train=False,download=True)
        elif(self.dataset=="MNIST"):
            MNIST(root='~/data', train=True, download=True)
            MNIST(root='~/data', train=False, download=True)

    def setup(self, stage: str):
        if(self.dataset=="CIFAR10"):
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

            if(stage == "fit"):
                self.trainset = CIFAR10(root='~/data', train=True,
                                                        download=True, transform=train_transform)
                self.testset = CIFAR10(root='~/data', train=False,
                                                    download=True, transform=test_transform)

        elif(self.dataset=="MNIST"):
            train_transform = transforms.Compose(
                [transforms.ToTensor(),
            ])

            if(stage == "fit"):
                self.trainset = MNIST(root='~/data', train=True,
                                                        download=True, transform=train_transform)
                self.testset = MNIST(root='~/data', train=False,
                                                    download=True, transform=train_transform)
                                                    
        elif(self.dataset=="ImageNet"):
            train_transform = transforms.Compose([
                transforms.Resize((256, 256)),
                transforms.CenterCrop((224,224)),
                transforms.ToTensor(),
            ])
            test_transform = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
            ])

            if(stage == "fit"):
                train_data_raw = ImageFolder('/data/ImageNet/2012/ILSVRC2012_img_train', transform=train_transform)
                val_data_raw = ImageFolder('/data/ImageNet/2012/ILSVRC2012_img_val', transform=test_transform)
                
                labels = list(range(self.num_classes))
                train_indices = [idx for idx, target in enumerate(train_data_raw.targets) if target in labels]
                val_indices = [idx for idx, target in enumerate(val_data_raw.targets) if target in labels]
                train_data_raw = torch.utils.data.Subset(train_data_raw, train_indices)
                val_data_raw = torch.utils.data.Subset(val_data_raw, val_indices)

                # assert False
                # self.trainset, self.testset = random_split(data_raw, [0.9, 0.1])
                self.trainset, self.testset = train_data_raw, val_data_raw


    def train_dataloader(self):
        return DataLoader(self.trainset, batch_size=self.batchsize, shuffle=True, num_workers =self.num_workers)

    def val_dataloader(self):
        return DataLoader(self.testset, batch_size=self.batchsize, shuffle=False, num_workers =self.num_workers)


class ImageClassifier(pl.LightningModule):
    def __init__(self, config, num_classes, input_ch):
        super().__init__()
        # self.save_hyperparameters()
        self.config = config
        self.num_classes = num_classes
        self.encoder = ModelWrapper(config, num_classes,input_ch)
        print(self.encoder)

    def training_step(self, batch, batch_idx):
        # training_step defines the train loop.
        # it is independent of forward
        x, y = batch


        if(self.config['adv']):
            
            benign_inputs, benign_targets_a, benign_targets_b, benign_lam = utils.mixup_data(x, y)
            benign_outputs = self.encoder(benign_inputs)
            loss1 = utils.mixup_criterion(nn.functional.cross_entropy, benign_outputs, benign_targets_a, benign_targets_b, benign_lam)


            advExample = self.generateAdv(x, y, self.config["adv_epsilon"])
            adv_inputs, adv_targets_a, adv_targets_b, adv_lam = utils.mixup_data(advExample, y)
            advZ = self.encoder(adv_inputs)
            loss2 = utils.mixup_criterion(nn.functional.cross_entropy, advZ, adv_targets_a, adv_targets_b, adv_lam)

            self.log("train_loss", (loss1 + loss2) / 2)

            return (loss1 + loss2) / 2
        else:
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
            )
        elif(self.config['optimizer'] == "SGD"):
            optimizer = torch.optim.SGD(
                self.parameters(),
                lr=self.config["lr"],
                momentum=self.config['momentum'],
                weight_decay=self.config['wd'],
            )
        elif(self.config['optimizer'] == "AdamW"):
            optimizer = torch.optim.AdamW(
                self.parameters(),
                lr=self.config["lr"]
            )
        
        if(self.config["lr_scheduler"]):
            train_dataloader = self.trainer.datamodule.train_dataloader() 
            data_per_gpu = len(train_dataloader.dataset) // 4 + 1
            steps_per_epoch = data_per_gpu // self.config["batch_size"] + 1
            scheduler_dict = {
                "scheduler": OneCycleLR(
                    optimizer,
                    self.config["lr"],
                    epochs=self.config["epochs"],
                    steps_per_epoch = steps_per_epoch,
                    final_div_factor = self.config["epochs"],
                ),
                "interval": "step",
            }
            return {"optimizer": optimizer, "lr_scheduler" : scheduler_dict, "monitor": "val_acc"}
        else:
            return {"optimizer": optimizer, "monitor": "val_acc"}
    
    def generateAdv(self, x, y, eps = 0.0314, alpha=0.00784, steps=3):
        with torch.enable_grad():
            atk = torchattacks.PGD(self.encoder, eps=eps, alpha=alpha, steps=steps)
            adv_images = atk(x, y)
        return adv_images
    
    def evaluateRobust(self, x, y):
        adv_images = self.generateAdv(x, y, self.config["adv_epsilon"])
        logits = self.encoder(adv_images)
        preds = torch.argmax(logits, dim=1)
        acc = accuracy(preds, y, num_classes=self.num_classes, task="multiclass")
        self.log("Robust_acc", acc, prog_bar=True, sync_dist=True)
    
    def evaluate(self, batch, stage=None):
        x, y = batch
        logits = self.encoder(x)
        loss = nn.functional.cross_entropy(logits, y)
        preds = torch.argmax(logits, dim=1)
        acc = accuracy(preds, y, num_classes=self.num_classes, task="multiclass")
        
        if stage:
            self.log(f"{stage}_loss", loss, prog_bar=True, sync_dist=True)
            self.log(f"{stage}_acc", acc, prog_bar=True, sync_dist=True)
            if(self.config['adv']):
                self.evaluateRobust(x, y)

    def validation_step(self, batch, batch_idx):
        self.evaluate(batch, "val")

    def test_step(self, batch, batch_idx):
        self.evaluate(batch, "test")

class ModelWrapper(pl.LightningModule):
    def __init__(self, config, num_classes,input_ch):
        super(ModelWrapper, self).__init__()
        self.config = config
        self.cnn = self.create_model(num_classes, input_ch)
        if(config["dataset"] == 'MNIST'):
            self.normalization = nn.Identity()
        else:
            mean, std = utils.getDataNormalization(config["dataset"])
            self.normalization = transforms.Normalize(mean, std)

    def create_model(self, num_classes,input_ch):
        
        resnet_models = {
            'resnet18' : resnet.resnet18,
            'resnet50' : resnet.resnet50,
            'resnet101' : resnet.resnet101
        }

        model = resnet_models[self.config['architecture']](weights=False, num_classes=num_classes)
        if(self.config['dataset'] != 'ImageNet'):
            model.conv1 = nn.Conv2d(input_ch, 64, kernel_size=(1, 1), stride=(1, 1), padding=(1, 1), bias=False)
            model.maxpool = nn.Identity()

        if(self.config["activation"] == 'racun'):
            for name,child in model.named_children():
                if(isinstance(child, nn.Sequential)):
                    for sub_name, sub_child in child.named_children():
                        sub_child.configure_RaCUN(SamplingLayer, self.config["replace_all"])
        
        return model

    def forward(self, x):
        x = self.normalization(x)
        return self.cnn(x)