# from modules import TransformerClassifier
import lightning.pytorch as pl
import torch.nn as nn
import torch
import torchvision.transforms as transforms
from torchvision.datasets import MNIST, CIFAR10, ImageFolder
from torchmetrics.functional import accuracy
from lightning.pytorch.loggers import WandbLogger
from lightning.pytorch.callbacks import ModelCheckpoint, LearningRateMonitor, EarlyStopping
from torch.optim.lr_scheduler import OneCycleLR
from models import BReLU, VariableBReLU, LeakyVariableBReLU, PVariableBReLU, Leaky_BReLU, PBReLU
import wandb
from modules import seed_everything
from setproctitle import setproctitle
from torch.nn import GELU, SiLU, ELU, LeakyReLU, PReLU, ReLU
import torchattacks
import yaml
import argparse

from torchvision.models import vit_b_16, swin_t, maxvit_t, ViT_B_16_Weights, Swin_T_Weights, MaxVit_T_Weights

#############################################################################################################
parser = argparse.ArgumentParser(
    description="""sweep with yaml
    usage: trans.py --yaml [yaml_path] --devices 0 --project_name [pname] --entity [ename]
    """
)
parser.add_argument('--yaml', required=True, help='yaml 파일 경로 입력')
parser.add_argument('--project_name', default="VBReLU_CIFAR10_random_test", help='wandb project name')
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
    
    
activation_functions = {
    'relu' : ReLU,
    'gelu' : GELU,
    'silu' : SiLU,
    'elu' : ELU,
    'LReLU' : LeakyReLU,
    'PReLU' : PReLU,
    'brelu' : BReLU,
    'Vbrelu' : VariableBReLU,
    'leakyVbrelu' : LeakyVariableBReLU,
    'PVbrelu' : PVariableBReLU,
    'leaky_brelu' : Leaky_BReLU,
    'pbrelu' : PBReLU,
}


def choose_dataset(config):
    if config["dataset"] == "CIFAR10":
        config['num_classes'] = 10
        train_transform = transforms.Compose(
            [
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
        ])

        test_transform = transforms.Compose(
            [
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
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
        config['num_classes'] = 100
        transform = transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.CenterCrop((224,224)),
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
        ])
        
        trainset = ImageFolder('/data/ImageNet100/train', transform=transform)
        testset = ImageFolder('/data/ImageNet100/val', transform=transform)
        
        trainloader = torch.utils.data.DataLoader(trainset, batch_size=config['batch_size'], shuffle=True, num_workers =config['num_workers'])
        testloader = torch.utils.data.DataLoader(testset, batch_size=config['batch_size'], shuffle=False, num_workers =config['num_workers'])
        
        data = (trainloader, testloader)


    elif config["dataset"] == "ImageNet":
        config['num_classes'] = 1000
        transform = transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.CenterCrop((224,224)),
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
        ])

        trainset = ImageFolder('/data/ImageNet/2012/ILSVRC2012_img_train', transform=transform)
        testset = ImageFolder('/data/ImageNet/2012/ILSVRC2012_img_val', transform=transform)
        
        trainloader = torch.utils.data.DataLoader(trainset, batch_size=config['batch_size'], shuffle=True, num_workers =config['num_workers'])

        # testset = ImageFolder('dataset/ImageNet/2012/ILSVRC2012_img_val', transform=transform)
        testloader = torch.utils.data.DataLoader(testset, batch_size=config['batch_size'], shuffle=False, num_workers =config['num_workers'])
        
        data = (trainloader, testloader)

        
    elif(config["dataset"]=="MNIST"):
        config['num_classes'] = 10
        transform = transforms.Compose(
            [transforms.ToTensor(),
             transforms.Normalize((0.1307, ), (0.3081, ))
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


# Define the LightningModule
class TransformerClassifier(pl.LightningModule):
    def __init__(self, config):
        super(TransformerClassifier, self).__init__()
        self.save_hyperparameters()
        self.config = config

        # Select the transformer model
        if self.config["architecture"] == 'vit':
            # self.model = vit_b_16(weights=ViT_B_16_Weights.IMAGENET1K_V1)  # Use weights
            self.model = vit_b_16(weights=None)  # No pretrained weights
            self.model.heads.head = nn.Linear(self.model.heads.head.in_features, self.config.num_classes)  # Correct head layer
        elif self.config["architecture"] == 'swin':
            # self.model = swin_t(weights=Swin_T_Weights.IMAGENET1K_V1)
            self.model = swin_t(weights=None)  # No pretrained weights
            self.model.head = nn.Linear(self.model.head.in_features, self.config.num_classes)
        elif self.config["architecture"] == 'maxvit':
            # self.model = maxvit_t(weights=MaxVit_T_Weights.IMAGENET1K_V1)
            self.model = maxvit_t(weights=None)  # No pretrained weights
            self.model.classifier[-1] = nn.Linear(self.model.classifier[-1].in_features, self.config.num_classes)
        else:
            raise ValueError(f"Unsupported model_name: {self.config.architecture}")

        self.replace_activation(self.model, activation_functions['activation'])
        print(self.model)
        

    def forward(self, x):
        return self.model(x)
    
    
    def generateAdv(self, x, y, atkType = "PGD", eps = 0.0314, alpha = 0.00784, steps = 7,
                    c = 1, kappa = 0):
        with torch.enable_grad():
            if atkType == "PGD":
                atk = torchattacks.PGD(self.model, eps=eps, alpha=alpha, steps=steps)
            elif atkType == "FGSM":
                atk = torchattacks.FGSM(self.model, eps=eps)
            elif atkType == "CW":
                atk = torchattacks.CW(self.model, c=c, kappa=kappa, steps=steps)
            adv_images = atk(x, y)
        return adv_images
    

    def training_step(self, batch, batch_idx):
        x, y = batch
        if self.config['adv']:
            logits = self(x)
            pure_loss = nn.functional.cross_entropy(logits, y)
            preds = torch.argmax(logits, dim=1)
            acc = accuracy(preds, y, num_classes=self.config['num_classes'], task="multiclass")
            self.log("train_clean_acc", acc)
            self.log("train_clean_loss", pure_loss)

            advExample = self.generateAdv(x, y, "PGD", self.config['eps'])
            advLogits = self(advExample)
            advLoss = nn.functional.cross_entropy(advLogits, y)
            advPreds = torch.argmax(advLogits, dim=1)
            robustAcc = accuracy(advPreds, y, num_classes=self.config["num_classes"], task="multiclass")
            self.log("train_robust_acc", robustAcc)
            self.log("train_robust_loss", advLoss)
            
            loss = (pure_loss + advLoss) / 2
        
        else:
            logits = self(x)
            loss = nn.functional.cross_entropy(logits, y)
            preds = torch.argmax(logits, dim=1)
            acc = accuracy(preds, y, num_classes=self.config['num_classes'], task="multiclass")
            self.log("train_acc")
            
        self.log("train_loss", loss)
        return loss

    
    def configure_optimizers(self):
        lr = self.config.get('lr') if self.config.get('lr') is not None else 0.01
        if(self.config['optimizer'] == "Adam"):
            b1 = self.config.get('beta1') if self.config.get('beta1') is not None else 0.9
            b2 = self.config.get('beta2') if self.config.get('beta2') is not None else 0.999
            optimizer = torch.optim.Adam(
                self.parameters(),
                lr=lr,
                betas=(b1, b2)
            )
        elif(self.config['optimizer'] == "SGD"):
            wd = self.config.get('wd') if self.config.get('wd') is not None else 5e-4
            momentum = self.config.get('momentum') if self.config.get('momentum') is not None else 0.85
            optimizer = torch.optim.SGD(
                self.parameters(),
                lr=lr,
                momentum=momentum,
                weight_decay=wd,
            )
        elif(self.config['optimizer'] == "AdamW"):
            b1 = self.config.get('beta1') if self.config.get('beta1') is not None else 0.9
            b2 = self.config.get('beta2') if self.config.get('beta2') is not None else 0.999
            optimizer = torch.optim.AdamW(
                self.parameters(),
                lr=lr,
                betas=(b1, b2)
            )
        
        if(self.config["lr_scheduler"]):
            # momentum = self.config.get('momentum') if self.config['optimizer'] == "SGD" else self.config.get('beta1')
            momentum = self.config.get('momentum') if self.config['optimizer'] == "SGD" else 0.85
            steps_per_epoch = self.config["dataloader_len"]
            scheduler_dict = {
                "scheduler": OneCycleLR(
                    optimizer,
                    max_lr=lr,
                    epochs=self.config["epochs"],
                    steps_per_epoch=steps_per_epoch,
                    base_momentum=momentum,
                ),
                "interval": "step",
            }

            return {"optimizer": optimizer, "lr_scheduler" : scheduler_dict, "monitor": "val_acc"}
        else:
            return {"optimizer": optimizer, "monitor": "val_acc"}

    def evaluateRobust(self, x, y):
        adv_images = self.generateAdv(x, y, "PGD", self.config['eps'])
        logits = self(adv_images)
        loss = nn.functional.cross_entropy(logits, y)
        preds = torch.argmax(logits, dim=1)
        acc = accuracy(preds, y, num_classes=self.config["num_classes"], task="multiclass")
        self.log("Robust_acc", acc, prog_bar=True, sync_dist=True)
        self.log("Robust_loss", loss)
    
    
    def validation_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = nn.functional.cross_entropy(preds, y)
        preds = torch.argmax(logits, dim=1)
        acc = accuracy(preds, y, num_classes=self.config['num_classes'], task='multiclass')
        self.log('val_loss', loss, prog_bar=True)
        self.log('val_acc', acc, prog_bar=True)
        if self.config['adv']:
            self.evaluateRobust(x, y)
            
        return loss
    

    def replace_activation(self, module, new_activation_fn):
        """
        Recursively replace all activation functions in the model with the given activation function.
        
        Args:
        - module: The module in which to replace the activation functions (e.g., self.model).
        - new_activation_fn: The new activation function to replace with (e.g., nn.ReLU()).
        """
        for name, child in module.named_children():
            if isinstance(child, (nn.ReLU, nn.GELU, nn.LeakyReLU)):
                # Replace activation function
                setattr(module, name, new_activation_fn())
            else:
                # Recur for child modules
                self.replace_activation(child, new_activation_fn)

# m = TransformerClassifier('vit')
# print(m)

# m.replace_activation(m, BReLU)
# print(m)

def train_model():
    run = wandb.init(project=project_name, entity=entity)
    config = wandb.config
    seed = config.get('seed') if config.get('seed') is not None else 42
    print(f"Seed set {seed}")
    seed_everything(seed)
    
    rpa = '-All' if config.get('replaceAll') else ''
    name_postfix = config['activation'] + rpa + '-' + config['optimizer']
    adver = "-adv" + 'eps:' + str(config.get('eps')) if config.get('adv') else ''
    name = config["dataset"] + "-" + config["architecture"] + "-" + name_postfix + adver
    run.name = name
    
    wandb_logger = WandbLogger(config=config, save_code=False)    # no checkpoint save
    
    train_loader, val_loader = choose_dataset(config=config)
    config.dataloader_len = len(train_loader)
    
    model = TransformerClassifier(config)
    wandb_logger.watch(model, log="all")
    
    variable_act = ["Vbrelu", "leakyVbrelu", "PVbrelu"]
    alpha = f"_a={config.get('alpha')}" if config['activation'] in variable_act else ''
    
    a_dir = '' if alpha=='' else f"/a={config.get('alpha')}"
    dir_path = f"./ckpt_{config['architecture']}/{config['dataset']}/{'all' if config.get('replaceAll') else 'part'}/{config['optimizer']}/{config['activation']}{a_dir}{'/'+str(seed)}"
    file_name = f"{config['activation']}{alpha}{'_ALL' if config.get('replaceAll') else ''}_{config['dataset']}_{config['optimizer']}_"

    callbacks = []
    lr_monitor = LearningRateMonitor(logging_interval='step')
    callbacks.append(lr_monitor)
    
    if config.get('adv') == False:
        checkpoint_callback = ModelCheckpoint(monitor="val_acc", mode="max",
                                            dirpath=dir_path,
                                            filename=file_name + '{epoch}_{val_acc:.4f}')
        callbacks.append(checkpoint_callback)
        # early_stop = EarlyStopping('val_acc', mode='max', patience=8)
        # callbacks.append(early_stop)
        
    else:
        checkpoint_callback = ModelCheckpoint(monitor="Robust_acc", mode="max",
                                            dirpath=dir_path,
                                            filename=file_name + '{epoch}_{Robust_acc:.4f}')
        callbacks.append(checkpoint_callback)    
        # early_stop = EarlyStopping('Robust_acc', mode='max', patience=8)
        # callbacks.append(early_stop)
        
    trainer = pl.Trainer(accelerator='gpu', max_epochs=config['epochs'], logger=wandb_logger,
                         callbacks=callbacks, devices=device_num)
    trainer.fit(model=model, train_dataloaders=train_loader, val_dataloaders=val_loader)
    torch.cuda.empty_cache()
    

def main():
    setproctitle('transformer adv train(jh)')
    resume = sweep_config.get('sweep_id')
    sweep_id = resume if resume else wandb.sweep(sweep_config, entity=entity, project=project_name)
    wandb.agent(sweep_id=sweep_id, function=train_model, project=project_name, entity=entity)
    

main()