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
import torchattacks
import yaml
import numpy as np
from torch.nn import GELU, SiLU, ELU, LeakyReLU, PReLU, ReLU
from setproctitle import setproctitle
from torchvision.models import wide_resnet101_2, wide_resnet50_2

##################################################################################


parser = argparse.ArgumentParser(
    description='reAct sweep with yaml \n usage: nl_sweep_yaml.py --yaml [yaml_path] --devices 0 --project_name [pname] -- entity [ename]\n')

parser.add_argument('--yaml', required=True, help='yaml 파일 경로 입력')
# parser.add_argument('--project_name', default="Brelu", help='wandb project name')
# parser.add_argument('--project_name', default="Brelu_ImageNet_A100", help='wandb project name')
# parser.add_argument('--project_name', default="VBReLU_CIFAR10_adv50", help='wandb project name')
parser.add_argument('--project_name', default="BReLU_CIFAR10_H100", help='wandb project name')
# parser.add_argument('--project_name', default="BReLU_ImageNet100_pgd7", help='wandb project name')
# parser.add_argument('--project_name', default="BReLU_CIFAR10_IAT_seeds", help='wandb project name')
# parser.add_argument('--project_name', default="Brelu_CIFAR-10", help='wandb project name')
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

def seed_everything(seed:int = 1004):
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)  # current gpu seed
    torch.cuda.manual_seed_all(seed) # All gpu seed
    torch.backends.cudnn.deterministic = True  # type: ignore
    torch.backends.cudnn.benchmark = False  # True로 하면 gpu에 적합한 알고리즘을 선택함.



###################################################################################

resnet_models = {
    'resnet18' : resnet.resnet18,
    'resnet50' : resnet.resnet50,
    'resnet101' : resnet.resnet101,
    'wide_resnet50' : wide_resnet50_2,
    'wide_resnet101': wide_resnet101_2
}

activation_functions = {
    'relu' : ReLU,
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
    
    
    def replace_activation(self, module, new_activation_fn, alpha=None):
        for name, child in module.named_children():
            if isinstance(child, ReLU):
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
        
        
        if self.config['architecture'] in ['resnet18', 'resnet50', 'resnet101']:
            if activation == 'relu':
                pass
            
            elif activation == 'Vbrelu' or activation == 'leakyVbrelu' or activation == 'PVbrelu' or activation == 'vbelu':
                act = activation_functions[self.config['activation']]
                alpha = self.config.get('alpha')

                if self.config.get('replaceAll'): model.relu = act(alpha=alpha)
                for name,child in model.named_children():
                    if(isinstance(child, nn.Sequential)):
                        for sub_name, sub_child in child.named_children():
                            sub_child.configure_react(activation_functions[self.config['activation']], replaceAll=self.config.get('replaceAll'), alpha=self.config.get('alpha'))
            
            else:
                act = activation_functions[self.config['activation']]

                if self.config.get('replaceAll'): model.relu = act()
                for name,child in model.named_children():
                    if(isinstance(child, nn.Sequential)):
                        for sub_name, sub_child in child.named_children():
                            sub_child.configure_react(activation_functions[self.config['activation']], replaceAll=self.config.get('replaceAll'))
        else:
            self.replace_activation(model, activation, self.config.get('alpha'))
        
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
        
        self.random_scheduler = config.get("random_scheduler")
        if self.random_scheduler:
            # self.alpha_schedule = torch.linspace(1000, 1, self.config['epochs'])    # 선형 감소 (1000, 1)
            # self.alpha_schedule = torch.logspace(3, 0, self.config['epochs'])    # log 감소 (1000, 1)
            self.alpha_schedule = torch.logspace(10, 0, self.config['epochs'], base=2)    # log 감소 (1024, 1)
        print(self.encoder)
    
    
    def change_VBReLU_alpha(self, alpha):
        for name, module in self.encoder.named_modules():
            if isinstance(module, VariableBReLU):
                module.set_alpha(alpha)

    
    def mixup_data(self, x, y):
        mixup_alpha = 1.0
        lam = np.random.beta(mixup_alpha, mixup_alpha)
        batch_size = x.size()[0]
        index = torch.randperm(batch_size).cuda()
        mixed_x = lam * x + (1 - lam) * x[index, :]
        y_a, y_b = y, y[index]
        return mixed_x, y_a, y_b, lam
    
    def mixup_criterion(self, criterion, pred, y_a, y_b, lam):
        return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)
    
    
    def on_train_epoch_start(self):
        if self.random_scheduler:
            current_alpha = self.alpha_schedule[self.current_epoch]
            self.change_VBReLU_alpha(current_alpha)
            self.log("current_alpha", current_alpha)

    def training_step(self, batch, batch_idx):
        # training_step defines the train loop.
        # it is independent of forward
        x, y = batch
        # Logging to TensorBoard (if installed) by default
        # print(loss)

        if(self.config['adv']):
            # part adv train (20%)
            # advIdx = torch.randint(x.shape[0], (int(x.shape[0] * 0.2),))
            # advExample = self.generateAdv(x[advIdx], y[advIdx])
            # advZ = self.encoder(advExample)
            # advLoss = nn.functional.cross_entropy(advZ, y[advIdx])
            # self.log("train_loss", loss + advLoss)
            # return loss + advLoss
            
            #########################################################################################
            # all adv train (50%)
            logits = self.encoder(x)
            pure_loss = nn.functional.cross_entropy(logits, y)
            preds = torch.argmax(logits, dim=1)
            acc = accuracy(preds, y, num_classes=self.config["num_classes"], task="multiclass")
            self.log("train_clean_acc", acc)
            self.log("train_clean_loss", pure_loss)
            
            advExample = self.generateAdv(x, y, "PGD", self.config['eps'])
            advZ = self.encoder(advExample)
            advLoss = nn.functional.cross_entropy(advZ, y)
            adv_preds = torch.argmax(advZ, dim=1)
            robust_acc = accuracy(adv_preds, y, num_classes=self.config["num_classes"], task="multiclass")
            self.log("train_robust_acc", robust_acc)
            self.log("train_robust_loss", advLoss)
            
            loss = (pure_loss + advLoss) / 2

            ##########################################################################################
            
            
            # # interpolated adversarial training(IAT)
            # # vanilla loss with mixup
            # mixup_x, mixup_y_a, mixup_y_b, mixup_lambda = self.mixup_data(x, y)
            # mixup_output = self.encoder(mixup_x)
            # unperturbed_loss = self.mixup_criterion(nn.functional.cross_entropy, mixup_output, mixup_y_a, mixup_y_b, mixup_lambda)
            # # vanilla loss with no mixup
            # # pred = self.encoder(x)
            # # unperturbed_loss = nn.functional.cross_entropy(pred, y)
            
            # # adv loss
            # advExample = self.generateAdv(x, y, "PGD", self.config['eps'])
            # adv_input, adv_y_a, adv_y_b, adv_lam = self.mixup_data(advExample, y)
            # adv_output = self.encoder(adv_input)
            # perturbed_loss = self.mixup_criterion(nn.functional.cross_entropy, adv_output, adv_y_a, adv_y_b, adv_lam)

            # loss = (unperturbed_loss + perturbed_loss) / 2

        else:
            z = self.encoder(x)
            loss = nn.functional.cross_entropy(z, y)
            
        self.log("train_loss", loss)
        
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
    
    
    def generateAdv(self, x, y, atkType = 'PGD', eps = 0.0314, alpha=0.00784, steps=7):
        with torch.enable_grad():
            if atkType == 'PGD':
                atk = torchattacks.PGD(self.encoder, eps=eps, alpha=alpha, steps=steps)
            elif atkType == 'FGSM':
                atk = torchattacks.FGSM(self.encoder, eps=eps)
            adv_images = atk(x, y)
        return adv_images


    def evaluateRobust(self, x, y):
        adv_images = self.generateAdv(x, y, "PGD", self.config['eps'])
        logits = self.encoder(adv_images)
        loss = nn.functional.cross_entropy(logits, y)
        preds = torch.argmax(logits, dim=1)
        acc = accuracy(preds, y, num_classes=self.config["num_classes"], task="multiclass")
        self.log("Robust_acc", acc, prog_bar=True, sync_dist=True)
        self.log("Robust_loss", loss)

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
        loss = self.evaluate(batch, "val")
        
        if self.random_scheduler:
            current_alpha = self.encoder.get_alpha()    # 현재 알파 저장
            self.change_VBReLU_alpha(1.0)
            self.evaluate(batch, "val_alpha1")
            self.change_VBReLU_alpha(current_alpha)
            
        return loss

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
        
        trainset = ImageFolder('/data/ImageNet100/train', transform=transform)
        testset = ImageFolder('/data/ImageNet100/val', transform=transform)
        
        trainloader = torch.utils.data.DataLoader(trainset, batch_size=config['batch_size'], shuffle=True, num_workers =config['num_workers'])
        testloader = torch.utils.data.DataLoader(testset, batch_size=config['batch_size'], shuffle=False, num_workers =config['num_workers'])
        
        data = (trainloader, testloader)


    elif config["dataset"] == "ImageNet":
        transform = transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.CenterCrop((224,224)),
            transforms.ToTensor(),
        ])

        trainset = ImageFolder('/data/ImageNet/2012/ILSVRC2012_img_train', transform=transform)
        testset = ImageFolder('/data/ImageNet/2012/ILSVRC2012_img_val', transform=transform)
        
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
        data_raw = ImageFolder('/data/tiny-imagenet-200/train', transform=transform)
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


def append_dropout(model, act, rate = 0.5, front = False):
    for name, module in model.named_children():
        if len(list(module.children())) > 0:
            append_dropout(module, act, rate, front)
        if isinstance(module, act):
            if front:
                dropout_relu = nn.Sequential(nn.Dropout2d(p = rate, inplace=False), module)
            else: 
                dropout_relu = nn.Sequential(module, nn.Dropout2d(p = rate, inplace=False))
            setattr(model, name, dropout_relu)


def train_model():
    run = wandb.init(project=project_name, entity=entity)
    config = wandb.config
    seed = config.get('seed')
    seed = seed if seed is not None else 42
    print(f"Seed is {seed}")
    seed_everything(seed)
    setproctitle(f"{config['activation']}_{config['architecture']}_{config['dataset']} (jh)")
    # wandb.define_metric("val_acc", summary="max")
    # wandb.define_metric("Robust_acc", summary="max")
    # name_postfix = "reference" if config['activation'] == 'relu' else "ReAct"
    rpa = '-All' if config.get('replaceAll') else ''
    name_postfix = config['activation'] + rpa + '-' + config['optimizer']
    adver = "-adv" + 'eps:' + str(config.get('eps')) if config.get('adv') else ''
    
    
    # name = config["dataset"] + "-" + config["architecture"] + "-" + name_postfix + "-" + config.optimizer + " lr:" + str(round(config.lr, 4)) + adver
    name = config["dataset"] + "-" + config["architecture"] + "-" + name_postfix + adver
    run.name = name
    # wandb_logger = WandbLogger(config=config, save_code=False, log_model="all")
    wandb_logger = WandbLogger(config=config, save_code=False)    # no checkpoint save

    data = choose_dataset(config=config)


    config.dataloader_len = len(data[0])

    modified_resnet_encoder = LitAutoEncoder(config)

    # dropout 관련
    # d_first = config.get('front')
    # prefix_d = "front" if d_first else "back"
    # dropout_p = config.get('dropout')
    # dropout_p = dropout_p if dropout_p is not None else 0
    
    # if dropout_p != 0:
    #     append_dropout(model=modified_resnet_encoder, act=activation_functions[config['activation']],
    #                    rate=dropout_p, front=d_first)
    #     print("\n\nAfter append Dropout\n")
    #     print(modified_resnet_encoder)
    #     print("\n")


    wandb_logger.watch(modified_resnet_encoder, log="all")

    # file_name = config['activation'] + str(round(config.get('alpha'), 3))
    # file_name = config['activation'] + '-' + config['dataset'] + '-'
    variable_act = ["Vbrelu", "leakyVbrelu", "PVbrelu"]
    alpha = f"_a={config.get('alpha')}" if config['activation'] in variable_act else ''

    a_dir = '' if alpha=='' else f"/a={config.get('alpha')}"
    dir_path = f"./ckpt_{config['architecture']}/{config['dataset']}/{'all' if config.get('replaceAll') else 'part'}/{config['optimizer']}/{config['activation']}{a_dir}{'/'+str(seed)}"
    file_name = f"{config['activation']}{alpha}{'_ALL' if config.get('replaceAll') else ''}_{config['dataset']}_{config['optimizer']}_"
    # dir_path = f"./ckpt_pgd7/{config['dataset']}/{'all' if config.get('replaceAll') else 'part'}/{config['optimizer']}/{config['activation']}/{prefix_d}/{dropout_p}"
    # file_name = f"{config['activation']}{alpha}{'_ALL' if config.get('replaceAll') else ''}_{config['dataset']}_{config['optimizer']}_drop={dropout_p}"
    
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
    
    
    
    # trainer = pl.Trainer(max_epochs = config["epochs"],logger= wandb_logger, callbacks=[checkpoint_callback,lr_monitor], devices = find_usable_cuda_devices(1))
    trainer = pl.Trainer(accelerator = 'gpu', max_epochs = config["epochs"],logger= wandb_logger, callbacks=callbacks, devices = device_num)
    # trainer = pl.Trainer(max_epochs = config["epochs"],logger= wandb_logger, callbacks=[lr_monitor], devices = device_num)
    trainer.fit(model=modified_resnet_encoder, train_dataloaders=data[0], val_dataloaders=data[1])
    torch.cuda.empty_cache()
    # wandb.finish()



def main():
    # setproctitle('pgd7 adv train (jh)')
    resume = sweep_config.get('sweep_id')
    sweep_id = resume if resume else wandb.sweep(sweep_config, project=project_name)
    # sweep_id = "gt3qp3cj"
    wandb.agent(sweep_id=sweep_id, function=train_model, project=project_name, entity=entity)


main()