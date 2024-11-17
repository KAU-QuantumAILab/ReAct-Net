import yaml
import argparse
from setproctitle import setproctitle

import wandb

import lightning.pytorch as pl
from lightning.pytorch.loggers import WandbLogger
from lightning.pytorch.callbacks import ModelCheckpoint, LearningRateMonitor, EarlyStopping

import torch
import torch.nn as nn
from torch.nn import GELU, SiLU, ELU, LeakyReLU, PReLU, ReLU
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import OneCycleLR

import torchattacks

from torchmetrics.functional import accuracy

import torchvision.transforms as transforms
from torchvision.datasets import MNIST, CIFAR10, ImageFolder
from torchvision.models import VisionTransformer, SwinTransformer
from torchvision.models import *
# from torchvision.models import vit_b_16, swin_t, maxvit_t, ViT_B_16_Weights, Swin_T_Weights, MaxVit_T_Weights, vit_l_16, swin_s, swin_b, Swin_B_Weights
# from torchvision.models import vgg16, efficientnet_b0, VGG16_Weights, EfficientNet_B0_Weights, efficientnet_v2_s, EfficientNet_V2_S_Weights
# from torchvision.models import wide_resnet101_2, wide_resnet50_2, resnet18

from models import BReLU, VariableBReLU, LeakyVariableBReLU, PVariableBReLU, Leaky_BReLU, PBReLU
from modules import seed_everything

#############################################################################################################

parser = argparse.ArgumentParser(
    description="""sweep with yaml
    usage: trans.py --yaml [yaml_path] --devices 0 --project_name [pname] --entity [ename]
    """
)
parser.add_argument('--yaml', required=True, help='yaml 파일 경로 입력')
parser.add_argument('--project_name', default="CIFAR10_Norm_test", help='wandb project name')
parser.add_argument('--entity', default='kau-quantum', help='wandb entity name')
parser.add_argument('--devices', default=0, type=int, help='choose the CUDA(ex: 0, 1, 2, -1)')

args = parser.parse_args()

torch.set_float32_matmul_precision('high')
# torch.autograd.set_detect_anomaly(True)
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
    

def get_num_classes(dataset):
    if dataset in ["CIFAR10", "MNIST"]:
        return 10
    elif dataset in ["ImageNet"]:
        return 1000
    elif dataset in ["TinyImagenet"]:
        return 200
    elif dataset in ["ImageNet100"]:
        return 100
    
    
def create_model(architecture, dataset, num_classes, pretrain):
    if architecture == 'vit':
        if dataset in ["CIFAR10", "TinyImagenet"]:
            return VisionTransformer(
                image_size=32,
                patch_size=4,
                num_layers=12,
                num_heads=12,
                hidden_dim=384,
                mlp_dim=1536,
                num_classes=num_classes
            )
        
        elif dataset in ["ImageNet", "ImageNet100"]:
            weight = ViT_B_16_Weights.IMAGENET1K_V1 if pretrain else None
            model = vit_b_16(weights=weight)
            if num_classes != 1000:
                model.heads.head = nn.Linear(model.heads.head.in_features, num_classes)
                
            return model
    
    elif architecture == 'swin':
        if dataset in ["CIFAR10", "TinyImagenet"]:
            return SwinTransformer(
                patch_size=[2, 2],
                embed_dim=96,
                depths=[2, 6, 4],
                num_heads=[3, 6, 12],
                window_size=[4, 4],
                mlp_ratio=4,
                num_classes=num_classes
            )
        
        elif dataset in ["ImageNet", "ImageNet100"]:
            weight = Swin_B_Weights.IMAGENET1K_V1 if pretrain else None
            model = swin_b(weights=weight)
            if num_classes != 1000:
                model.head = nn.Linear(model.head.in_features, num_classes)
                
            return model
        
    elif 'resnet' in architecture:
        resnet_models = {
            'resnet18' : resnet18,
            'resnet50' : resnet50,
            'resnet101' : resnet101,
            'wide_resnet50' : wide_resnet50_2,
            'wide_resnet101': wide_resnet101_2,
        }
        resnet_weights = {
            'resnet18' : ResNet18_Weights.IMAGENET1K_V1,
            'resnet50' : ResNet50_Weights.IMAGENET1K_V1,
            'resnet101' : ResNet101_Weights.IMAGENET1K_V1,
            'wide_resnet50' : Wide_ResNet50_2_Weights.IMAGENET1K_V1,
            'wide_resnet101': Wide_ResNet50_2_Weights.IMAGENET1K_V1,
        }
        if dataset in ["ImageNet", "ImageNet100"]:
            if pretrain:
                weight = resnet_weights[architecture]
                model = resnet_models[architecture](weights=weight)
                if num_classes != 1000:
                    model.fc = nn.Linear(model.fc.in_features, num_classes)
            else:
                model = resnet_models[architecture](weights=None, num_classes=num_classes)
            
            return model
        
        else:
            if pretrain:
                weight = resnet_weights[architecture]
                model = resnet_models[architecture](weights=weight)
                model.fc = nn.Linear(model.fc.in_features, num_classes)
            
            else:
                model = resnet_models[architecture](weights=None, num_classes=num_classes)
                
            if dataset in ["CIFAR10", "TinyImageNet"]:
                model.conv1 = nn.Conv2d(3, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
                model.maxpool = nn.Identity()
            elif dataset == 'MNIST':
                model.conv1 = nn.Conv2d(1, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
                model.maxpool = nn.Identity()
            
            return model

        
    elif architecture == 'vgg16':
        pass
    
    elif architecture == 'eff':
        pass
            
    
    
class Classifier(pl.LightningModule):
    def __init__(self, config):
        super(Classifier, self).__init__()
        
        self.config = config
        dataset = config['dataset']
        num_classes = self.config['num_classes'] = get_num_classes(dataset)
        architecture = config['architecture']
        pretrain = config.get('pretrain', False)
        
        self.random_scheduler = config.get('random_scheduler')
        if self.random_scheduler:
            # self.alpha_schedule = torch.linspace(1000, 1, self.config['epochs'])    # 선형 감소 (1000, 1)
            # self.alpha_schedule = torch.logspace(3, 0, self.config['epochs'])    # log 감소 (1000, 1)
            self.alpha_schedule = torch.logspace(10, 0, self.config['epochs'], base=2)    # log 감소 (1024, 1)
        
        if config.get('model_norm', True):
            mean, std = getDataNormalization(self.config["dataset"])
            self.normalization = transforms.Normalize(mean, std)
        else:
            self.normalization = nn.Identity()
        
        self.model = create_model(architecture, dataset, num_classes, pretrain)
        self.replace_activation(self.model, self.config['activation'], self.config.get('alpha'))
        
        print(self)
    
    
    def forward(self, x):
        x = self.normalization(x)
        return self.model(x)
    
    
    def on_train_epoch_start(self):
        if self.random_scheduler:
            current_alpha = self.alpha_schedule[self.current_epoch]
            self.change_VBReLU_alpha(current_alpha)
            self.log("current_alpha", current_alpha)
    
    def training_step(self, batch, batch_idx):
        x, y = batch
        
        if(self.config['adv']):
            # all adv train (50%)
            logits = self(x)
            pure_loss = nn.functional.cross_entropy(logits, y)
            preds = torch.argmax(logits, dim=1)
            acc = accuracy(preds, y, num_classes=self.config["num_classes"], task="multiclass")
            self.log("train_clean_acc", acc)
            self.log("train_clean_loss", pure_loss)
            
            advExample = self.generateAdv(x, y, "PGD", self.config['eps'])
            advZ = self(advExample)
            advLoss = nn.functional.cross_entropy(advZ, y)
            adv_preds = torch.argmax(advZ, dim=1)
            robust_acc = accuracy(adv_preds, y, num_classes=self.config["num_classes"], task="multiclass")
            self.log("train_robust_acc", robust_acc)
            self.log("train_robust_loss", advLoss)
            
            loss = (pure_loss + advLoss) / 2
        
        else:
            z = self(x)
            loss = nn.functional.cross_entropy(z, y)
        
        self.log("train_loss", loss)
        return loss
    
        
    def generateAdv(self, x, y, atkType = 'PGD', eps = 0.0314, alpha=0.00784, steps=7):
        with torch.enable_grad():
            if atkType == 'PGD':
                atk = torchattacks.PGD(self, eps=eps, alpha=alpha, steps=steps)
            elif atkType == 'FGSM':
                atk = torchattacks.FGSM(self, eps=eps)
            elif atkType == 'CW':
                c = self.config.get('c', 1)
                kappa = self.config.get('kappa', 0)
                atk = torchattacks.CW(self, c = c, kappa=kappa, steps=steps)
            elif atkType == 'auto':
                atk = torchattacks.AutoAttack(self, norm="Linf", eps=eps, n_classes=self.config['num_classes'])
            elif atkType == 'square':
                atk = torchattacks.Square(self, eps=eps)
            elif atkType == 'eotpgd':
                eot_iter = self.config.get('eot_iter', 2)
                atk = torchattacks.EOTPGD(self, eps=eps, alpha=alpha, steps=steps, eot_iter=eot_iter)
            elif atkType == 'one_pixel':
                atk = torchattacks.OnePixel(self, pixels=5, steps=50, popsize=100)
            elif atkType == "pixle":
                atk = torchattacks.Pixle(self, restarts=100, max_iterations=20)
            adv_images = atk(x, y)
        return adv_images
    
    
    def evaluateRobust(self, x, y):
        attack_type = self.config.get('atk', 'PGD')
        epsilon = self.config.get('eps', 0.0314)
        attack_steps = self.config.get('steps', 7)
        adv_images = self.generateAdv(x=x, y=y, atkType=attack_type, eps=epsilon, steps=attack_steps)
        logits = self(adv_images)
        loss = nn.functional.cross_entropy(logits, y)
        preds = torch.argmax(logits, dim=1)
        acc = accuracy(preds, y, num_classes=self.config["num_classes"], task="multiclass")
        self.log("Robust_acc", acc, prog_bar=True, sync_dist=True)
        self.log("Robust_loss", loss)
        
        
    def evaluate(self, batch, stage=None):
        x, y = batch
        logits = self(x)
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
        loss = self.evaluate(batch, "val")
        
        if self.random_scheduler:
            current_alpha = self.model.get_alpha()    # 현재 알파 저장
            self.change_VBReLU_alpha(1.0)
            self.evaluate(batch, "val_alpha1")
            self.change_VBReLU_alpha(current_alpha)
            
        return loss

    def test_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        clean_loss = nn.functional.cross_entropy(logits, y)
        preds = torch.argmax(logits, dim=1)
        clean_acc = accuracy(preds, y, num_classes=self.config["num_classes"], task="multiclass")
        self.log("test_clean_loss", clean_loss)
        self.log("test_clean_acc", clean_acc)
        
        if self.config['adv']:
            attack_type = self.config.get('atk', 'PGD')
            epsilon = self.config.get('eps', 0.0314)
            attack_steps = self.config.get('steps', 7)
            
            adv_example = self.generateAdv(x, y,
                                           atkType=attack_type,
                                           eps=epsilon,
                                           steps=attack_steps)
            advZ = self(adv_example)
            advLoss = nn.functional.cross_entropy(advZ, y)
            adv_preds = torch.argmax(advZ, dim=1)
            robust_acc = accuracy(adv_preds, y, num_classes=self.config["num_classes"], task="multiclass")
            self.log(f"test_{attack_type}_acc", robust_acc)
            self.log(f"test_{attack_type}_loss", advLoss)
            
        return clean_loss
    
        
    def replace_activation(self, module, new_activation_fn, alpha=None):
        for name, child in module.named_children():
            if isinstance(child, (ReLU, GELU, LeakyReLU, SiLU)):
                if new_activation_fn in ["Vbrelu", "leakyVbrelu", "PVbrelu"]:
                    new_act = activation_functions[new_activation_fn](alpha=alpha)
                else:
                    new_act = activation_functions[new_activation_fn]()
                setattr(module, name, new_act)
            else:
                self.replace_activation(child, new_activation_fn, alpha)
    
    
    def get_alpha(self):
        for name, module in self.named_modules():
            if isinstance(module, VariableBReLU):
                return module.alpha
        
        return 1.0
    
    
    def change_VBReLU_alpha(self, alpha):
        for name, module in self.named_modules():
            if isinstance(module, VariableBReLU):
                module.set_alpha(alpha)
        
                
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



def load_dataset(config):
    data_roots = {
        'MNIST' : ('~/data', '~/data'),
        'CIFAR10' : ('~/data', '~/data'),
        'ImageNet' : ('/data/ImageNet/2012/ILSVRC2012_img_train', '/data/ImageNet/2012/ILSVRC2012_img_val'),
        'ImageNet100' : ('/data/ImageNet100/train', '/data/ImageNet100/val')
    }
    
    dataset = config.get('dataset')
    batch_size = config.get('batch_size')
    num_workers = config.get('num_workers')
    model_norm = config.get('model_norm', True)
    if model_norm == False:
        mean, std = getDataNormalization(dataset)
        
    train_roots, val_roots = data_roots[dataset]
    
    if dataset == 'CIFAR10':
        train_transforms = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor()
        ])
        test_transforms = transforms.Compose([
            transforms.ToTensor()
        ])
        print(f"model_norm : {model_norm}")
        if model_norm == False:
            train_transforms.transforms.append(transforms.Normalize(mean, std))
            test_transforms.transforms.append(transforms.Normalize(mean, std))
        
        print(f"train_transform\n{train_transforms}")
        print(f"test_transform\n{test_transforms}")
        
        train_set = CIFAR10(
            root=train_roots,
            train=True,
            download=True,
            transform=train_transforms
        )
        test_set = CIFAR10(
            root=val_roots,
            train=False,
            download=True,
            transform=test_transforms
        )
        
    
    elif dataset in ["ImageNet", "ImageNet100"]:
        transform = transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.CenterCrop((224,224)),
            transforms.ToTensor(),
        ])
        if model_norm == False:
            transform.transforms.append(transforms.Normalize(mean, std))
        
        train_set = ImageFolder(
            root=train_roots, 
            transform=transform
        )
        test_set = ImageFolder(
            root=val_roots,
            transform=transform
        )
    
    elif dataset == 'MNIST':
        transform = transforms.Compose(
            [transforms.ToTensor(),
        ])
        if model_norm == False:
            transform.transforms.append(transforms.Normalize(mean, std))
            
        train_set = MNIST(
            root=train_roots, 
            train=True,
            download=True, 
            transform=transform)
        test_set = MNIST(
            root=val_roots, 
            train=False,
            download=True, 
            transform=transform)
    
    train_loader = DataLoader(
        dataset=train_set,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True
    )
    test_loader = DataLoader(
        dataset=test_set,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    return train_loader, test_loader


def train_model():
    run = wandb.init(project=project_name, entity=entity)
    config = wandb.config
    seed = config.get('seed', 42)
    print(f"Seed is {seed}")
    seed_everything(seed=seed)
    setproctitle(f"{config['activation']}_{config['architecture']}_{config['dataset']} (jh)")
    
    # run name 지정
    rpa = '-All' if config.get('replaceAll') else ''
    name_postfix = config['activation'] + rpa + '-' + config['optimizer']
    adver = "-adv" + 'eps:' + str(config.get('eps')) if config.get('adv') else ''
    name = config["dataset"] + "-" + config["architecture"] + "-" + name_postfix + adver
    run.name = name
    
    wandb_logger = WandbLogger(config=config, save_code=False)
    
    train_loader, val_loader = load_dataset(config=config)
    config['dataloader_len'] = len(train_loader)
    
    model = Classifier(config=config)
    wandb_logger.watch(model=model, log='all')
    
    variable_act = ["Vbrelu", "leakyVbrelu", "PVbrelu"]
    alpha = f"_a={config.get('alpha')}" if config['activation'] in variable_act else ''
    a_dir = '' if alpha=='' else f"/a={config.get('alpha')}"
    model_norm = 'model_norm' if config.get('model_norm', True) == True else 'data_norm'
    dir_path = f"./{model_norm}/ckpt_{config['architecture']}/{config['dataset']}/{'all' if config.get('replaceAll') else 'part'}/{config['optimizer']}/{config['activation']}{a_dir}{'/'+str(seed)}"
    file_name = f"{config['activation']}{alpha}{'_ALL' if config.get('replaceAll') else ''}_{config['dataset']}_{config['optimizer']}_"

    callbacks = [LearningRateMonitor(logging_interval='step')]
    if config.get('adv') == True:
        checkpoint_callback = ModelCheckpoint(
            monitor="val_acc", 
            mode="max",
            dirpath=dir_path,
            filename= file_name + '{epoch}_{val_acc:.4f}'
        )
        callbacks.append(checkpoint_callback)
        # early_stop = EarlyStopping(
        #     monitor='val_acc',
        #     mode='max',
        #     patience=8
        # )
        # callbacks.append(early_stop)
        
    else:
        checkpoint_callback = ModelCheckpoint(
            monitor="Robust_acc", 
            mode="max",
            dirpath=dir_path,
            filename= file_name + '{epoch}_{Robust_acc:.4f}'
        )
        callbacks.append(checkpoint_callback)
        # early_stop = EarlyStopping(
        #     monitor='Robust_acc',
        #     mode='max',
        #     patience=8
        # )
        # callbacks.append(early_stop)
        
    trainer = pl.Trainer(
        accelerator='gpu',
        max_epochs=config.get('epochs', 100),
        logger=wandb_logger,
        callbacks=callbacks,
        devices=device_num
    )
    trainer.fit(
        model=model,
        train_dataloaders=train_loader,
        val_dataloaders=val_loader
    )
    torch.cuda.empty_cache()
    wandb.finish()
    

def main():
    sweep_id = sweep_config.get(
        'sweep_id', 
        wandb.sweep(
            sweep=sweep_config, 
            entity=entity,
            project=project_name
        ))
    wandb.agent(
        sweep_id=sweep_id,
        function=train_model,
        entity=entity,
        project=project_name
    )
    
main()