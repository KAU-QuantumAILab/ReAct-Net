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
import torch.nn.functional as F

from torchvision.models import VisionTransformer, SwinTransformer
from torchvision.models import vit_b_16, swin_t, maxvit_t, ViT_B_16_Weights, Swin_T_Weights, MaxVit_T_Weights, vit_l_16, swin_s, swin_b
from torchvision.models import vgg16, efficientnet_b0, VGG16_Weights, EfficientNet_B0_Weights, efficientnet_v2_s, EfficientNet_V2_S_Weights
#############################################################################################################
parser = argparse.ArgumentParser(
    description="""sweep with yaml
    usage: trans.py --yaml [yaml_path] --devices 0 --project_name [pname] --entity [ename]
    """
)
parser.add_argument('--yaml', required=True, help='yaml 파일 경로 입력')
parser.add_argument('--project_name', default="CIFAR10_transformer", help='wandb project name')
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


def choose_dataset(config):
    if config["dataset"] == "CIFAR10":
        config['num_classes'] = 10
        if config['img_size'] == 32:
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
        elif config['img_size'] == 224:
            train_transform = transforms.Compose(
                [
                transforms.RandomResizedCrop(224),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
            ])

            test_transform = transforms.Compose(
                [
                transforms.Resize(224),
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
        config['num_classes'] = 100
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
        config['num_classes'] = 1000
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

class CustomReLU(nn.Module):
    def __init__(self):
        super(CustomReLU, self).__init__()
        # 기본적으로 inplace를 사용하지 않도록 설정
        self.inplace = False

    def forward(self, x):
        # inplace 옵션을 강제로 False로 설정한 ReLU
        return torch.maximum(x, torch.tensor(0.0, device=x.device))


class StochasticMultiheadAttention(pl.LightningModule):
    def __init__(self, config, embed_dim, num_heads, dropout=0.0, bias=True):
        super(StochasticMultiheadAttention, self).__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.dropout = dropout
        self.head_dim = embed_dim // num_heads
        assert self.head_dim * num_heads == self.embed_dim, "embed_dim must be divisible by num_heads"
        
        self.in_proj = nn.Linear(embed_dim, 3 * embed_dim, bias=bias)
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        
        self.mode = config.get('stochastic_mode')  # 'qk', 'v', or 'both'
        if config.get('att_activation') == 'brelu':
            self.brelu = BReLU()
        else:
            self.brelu = None
            
    
    def forward(self, query, key, value):
        batch_size, tgt_len, _ = query.size()
        src_len = key.size(1)
        
        qkv = self.in_proj(query).chunk(3, dim=-1)
        q, k, v = [x.view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2) for x in qkv]

        attn_weights = torch.matmul(q, k.transpose(-2, -1)) / (self.head_dim ** 0.5)
        attn_weights = F.softmax(attn_weights, dim=-1)
        
        if self.mode in ['qk', 'both'] and isinstance(self.brelu, BReLU):
            attn_weights = self.brelu(attn_weights, attn_weights)
            
        if self.mode in ['v', 'both'] and isinstance(self.brelu, BReLU):
            # V에 BReLU 적용
            v = self.brelu(v)
        
        attn_output = torch.matmul(attn_weights, v)
        
        if self.mode == 'qkv' and isinstance(self.brelu, BReLU):
            attn_output = self.brelu(attn_output)
            
        attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size, tgt_len, self.embed_dim)
        return self.out_proj(attn_output)
        
        
        
        
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
        


class TransformerClassifier(pl.LightningModule):
    def __init__(self, config):
        super(TransformerClassifier, self).__init__()
        
        self.config = config
        img_size = self.config['img_size']
        pretrained = self.config['pretrain']
        patch_size = 4 if img_size == 32 else 16
        
        mean, std = getDataNormalization(self.config["dataset"])
        self.normalization = transforms.Normalize(mean, std)

        # Select the model
        if self.config["architecture"] == 'vit':
            # vit = vit_b_16
            # self.model = vit(weights=ViT_B_16_Weights.IMAGENET1K_V1) if img_size == 224 and pretrained else vit(weights=None, image_size=img_size)
            # self.model.heads.head = nn.Linear(self.model.heads.head.in_features, self.config.num_classes)
            self.model = VisionTransformer(
                image_size=img_size,
                patch_size=patch_size,
                num_layers=12,
                num_heads=12,
                hidden_dim=384,
                mlp_dim=1536,
                num_classes=10
            )
            
        elif self.config["architecture"] == 'swin':
            # swin = swin_t
            # self.model = swin(weights=Swin_T_Weights.IMAGENET1K_V1) if img_size == 224 and pretrained else swin(weights=None)
            # self.model.head = nn.Linear(self.model.head.in_features, self.config.num_classes)
            self.model = SwinTransformer(
                patch_size=[2, 2],
                embed_dim=96,
                depths=[2, 6, 4],
                num_heads=[3, 6, 12],
                window_size=[4, 4],
                mlp_ratio=4,
                num_classes=10
            )
        
        elif self.config["architecture"] == 'vgg16':
            vgg = vgg16
            self.model = vgg(weights=VGG16_Weights.IMAGENET1K_V1) if pretrained else vgg(weights=None)
            if self.config['dataset'] == "CIFAR10" and self.config['img_size'] == 32:
                # 1. MaxPool 레이어를 Identity로 변경 (MaxPooling을 생략)
                # for i in [4, 9, 16, 23, 30]:  # VGG에서 MaxPool 레이어의 위치
                for i in [16, 23, 30]:  # VGG에서 MaxPool 레이어의 위치
                    self.model.features[i] = nn.Identity()  # MaxPool을 Identity로 대체

                # batch norm 없이 진행
                # # 2. BatchNorm2d 추가: Conv2d 뒤에 BatchNorm2d 추가
                # layers = []
                # for i, layer in enumerate(self.model.features):
                #     if isinstance(layer, nn.Conv2d):  # Conv 레이어 뒤에 BatchNorm 추가
                #         layers.append(layer)
                #         layers.append(nn.BatchNorm2d(layer.out_channels))  # BatchNorm 추가
                #     else:
                #         layers.append(layer)
                # # 새롭게 구성된 feature 블록을 덮어쓰기
                # self.model.features = nn.Sequential(*layers)

                # 2. Fully Connected Layer 크기 조정 (4096 -> 1024)
                self.model.classifier[0] = nn.Linear(512 * 7 * 7, 1024)  # in_features는 그대로, out_features를 1024로
                self.model.classifier[3] = nn.Linear(1024, 1024)  # 중간 FC 레이어도 1024로 줄임
                self.model.classifier[6] = nn.Linear(1024, self.config.num_classes)  # 최종 출력 레이어, CIFAR-10 클래스는 10개
                # 3. Dropout을 사용하지 않도록 제거
                self.model.classifier[2] = nn.Identity()  # Dropout 제거
                self.model.classifier[5] = nn.Identity()  # Dropout 제거

        elif self.config["architecture"] == 'efficientnet':
            effnet = efficientnet_b0
            self.model = effnet(weights=EfficientNet_B0_Weights.IMAGENET1K_V1) if pretrained else effnet(weights=None)
            if self.config['dataset'] == "CIFAR10" and self.config['img_size'] == 32:
                self.model.features[0][0] = nn.Conv2d(3, 32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
            self.model.classifier[-1] = nn.Linear(self.model.classifier[-1].in_features, self.config.num_classes)
            # dropout 해제
            self.model.classifier[0] = nn.Identity()

        elif self.config["architecture"] == 'eff2':
            effnet = efficientnet_v2_s
            self.model = effnet(weights=EfficientNet_V2_S_Weights.IMAGENET1K_V1) if pretrained else effnet(weights=None)
            if self.config['dataset'] == "CIFAR10" and self.config['img_size'] == 32:
                self.model.features[0][0] = nn.Conv2d(3, 24, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
            self.model.classifier[-1] = nn.Linear(self.model.classifier[-1].in_features, self.config.num_classes)
            # dropout 해제
            self.model.classifier[0] = nn.Identity()
        
        elif self.config["architecture"] == 'maxvit':
            self.model = maxvit_t(weights=None, input_size=(img_size, img_size))
            self.model.classifier[-1] = nn.Linear(self.model.classifier[-1].in_features, self.config.num_classes)
        else:
            raise ValueError(f"Unsupported model_name: {self.config.architecture}")

        self.replace_activation(self.model, self.config['activation'], self.config.get("alpha"))
        
        if self.config.get("architecture") in ['vit', 'swin'] and self.config.get("mha_change") == True:
            self.replace_attention(self.model)

        print(self.model)
        

    def replace_attention(self, model):
        """ViT의 기본 어텐션을 커스텀 스토캐스틱 어텐션으로 교체"""
        for name, module in model.named_modules():
            if isinstance(module, nn.MultiheadAttention):
                parent = self.get_parent_module(model, name)
                setattr(parent, name.split('.')[-1], 
                        StochasticMultiheadAttention(self.config, 
                                                     module.embed_dim,
                                                     module.num_heads,
                                                     module.dropout,
                                                     module.in_proj_bias is not None))
                
    def get_parent_module(self, model, name):
        """모듈의 부모 모듈을 찾는 헬퍼 함수"""
        parent_name = '.'.join(name.split('.')[:-1])
        if parent_name:
            for n, m in model.named_modules():
                if n == parent_name:
                    return m
        return model
    
    
    
    def forward(self, x):
        x = self.normalization(x)
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
        loss = nn.functional.cross_entropy(logits, y)
        preds = torch.argmax(logits, dim=1)
        acc = accuracy(preds, y, num_classes=self.config['num_classes'], task='multiclass')
        self.log('val_loss', loss, prog_bar=True)
        self.log('val_acc', acc, prog_bar=True)
        if self.config['adv']:
            self.evaluateRobust(x, y)
            
        return loss
    

    def replace_activation(self, module, new_activation_fn, alpha = None):
        """
        Recursively replace all activation functions in the model with the given activation function.
        
        Args:
        - module: The module in which to replace the activation functions (e.g., self.model).
        - new_activation_fn: The new activation function to replace with (e.g., nn.ReLU()).
        """
        for name, child in module.named_children():
            if isinstance(child, (ReLU, GELU, LeakyReLU, SiLU)):
                # Replace activation function
                if new_activation_fn == "relu" and self.config['architecture'] == "eff2":
                    new_act = CustomReLU()
                elif new_activation_fn in ["Vbrelu", "leakyVbrelu", "PVbrelu"]:
                    new_act = activation_functions[new_activation_fn](alpha = alpha)
                else:
                    new_act = activation_functions[new_activation_fn]()
                setattr(module, name, new_act)
            else:
                # Recur for child modules
                self.replace_activation(child, new_activation_fn, alpha)


def train_model():
    run = wandb.init(project=project_name, entity=entity)
    config = wandb.config
    seed = config.get('seed') if config.get('seed') is not None else 42
    print(f"Seed set {seed}")
    seed_everything(seed)
    setproctitle(f"{config['activation']}_{config['architecture']}_{config['dataset']} (jh)")
    
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
    # dir_path = f"/media/qlab/새 볼륨/ckpt_{config['architecture']}/{config['dataset']}/{'all' if config.get('replaceAll') else 'part'}/{config['optimizer']}/{config['activation']}{a_dir}{'/'+str(seed)}"
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
        early_stop = EarlyStopping('val_acc', mode='max', patience=8)
        # callbacks.append(early_stop)
        
    else:
        checkpoint_callback = ModelCheckpoint(monitor="Robust_acc", mode="max",
                                            dirpath=dir_path,
                                            filename=file_name + '{epoch}_{Robust_acc:.4f}')
        callbacks.append(checkpoint_callback)    
        early_stop = EarlyStopping('Robust_acc', mode='max', patience=8)
        # callbacks.append(early_stop)
        
    trainer = pl.Trainer(accelerator='gpu', max_epochs=config['epochs'], logger=wandb_logger,
                         callbacks=callbacks, devices=device_num)
    trainer.fit(model=model, train_dataloaders=train_loader, val_dataloaders=val_loader)
    torch.cuda.empty_cache()
    

def main():
    resume = sweep_config.get('sweep_id')
    sweep_id = resume if resume else wandb.sweep(sweep_config, project=project_name)
    wandb.agent(sweep_id=sweep_id, function=train_model, project=project_name, entity=entity)
    

main()