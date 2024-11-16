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
from torchattacks import FGSM, PGD, CW, Square, EOTPGD, AutoAttack, Pixle, OnePixel
import yaml
import numpy as np
from torch.nn import GELU, SiLU, ELU, LeakyReLU, PReLU, ReLU
from torch.utils.data import Dataset, DataLoader, Subset
from torchattacks.attack import Attack
from torchvision.models import vit_b_16, swin_t, maxvit_t, ViT_B_16_Weights, Swin_T_Weights, MaxVit_T_Weights, vit_l_16, swin_s, swin_b
from torchvision.models import vgg16, efficientnet_b0, VGG16_Weights, EfficientNet_B0_Weights, efficientnet_v2_s, EfficientNet_V2_S_Weights
from torchvision.models import VisionTransformer, SwinTransformer
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

class TransformerClassifier(pl.LightningModule):
    def __init__(self, config):
        super(TransformerClassifier, self).__init__()
        
        self.config = config
        img_size = self.config['img_size']
        pretrained = self.config['pretrain']
        
        mean, std = getDataNormalization(self.config["dataset"])
        self.normalization = transforms.Normalize(mean, std)

        # Select the model
        if self.config["architecture"] == 'vit':
            if img_size == 224:
                vit = vit_b_16
                self.model = vit(weights=ViT_B_16_Weights.IMAGENET1K_V1) if img_size == 224 and pretrained else vit(weights=None, image_size=img_size)
                self.model.heads.head = nn.Linear(self.model.heads.head.in_features, self.config['num_classes'])
            else:   # CIFAR10
                self.model = VisionTransformer(
                image_size=img_size,
                patch_size=4,
                num_layers=12,
                num_heads=12,
                hidden_dim=384,
                mlp_dim=1536,
                num_classes=10
                )
                
        elif self.config["architecture"] == 'swin':
            if img_size == 224:
                swin = swin_t
                self.model = swin(weights=Swin_T_Weights.IMAGENET1K_V1) if img_size == 224 and pretrained else swin(weights=None)
                self.model.head = nn.Linear(self.model.head.in_features, self.config['num_classes'])
            else:   # CIFAR10
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
                self.model.classifier[6] = nn.Linear(1024, self.config['num_classes'])  # 최종 출력 레이어, CIFAR-10 클래스는 10개
                # 3. Dropout을 사용하지 않도록 제거
                self.model.classifier[2] = nn.Identity()  # Dropout 제거
                self.model.classifier[5] = nn.Identity()  # Dropout 제거

        elif self.config["architecture"] == 'efficientnet':
            effnet = efficientnet_b0
            self.model = effnet(weights=EfficientNet_B0_Weights.IMAGENET1K_V1) if pretrained else effnet(weights=None)
            if self.config['dataset'] == "CIFAR10" and self.config['img_size'] == 32:
                self.model.features[0][0] = nn.Conv2d(3, 32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
            self.model.classifier[-1] = nn.Linear(self.model.classifier[-1].in_features, self.config['num_classes'])
            # dropout 해제
            self.model.classifier[0] = nn.Identity()

        elif self.config["architecture"] == 'eff2':
            effnet = efficientnet_v2_s
            self.model = effnet(weights=EfficientNet_V2_S_Weights.IMAGENET1K_V1) if pretrained else effnet(weights=None)
            if self.config['dataset'] == "CIFAR10" and self.config['img_size'] == 32:
                self.model.features[0][0] = nn.Conv2d(3, 24, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
            self.model.classifier[-1] = nn.Linear(self.model.classifier[-1].in_features, self.config['num_classes'])
            # dropout 해제
            self.model.classifier[0] = nn.Identity()
        
        elif self.config["architecture"] == 'maxvit':
            self.model = maxvit_t(weights=None, input_size=(img_size, img_size))
            self.model.classifier[-1] = nn.Linear(self.model.classifier[-1].in_features, self.config['num_classes'])
        else:
            raise ValueError(f"Unsupported model_name: {self.config['architecture']}")

        self.replace_activation(self.model, activation_functions[self.config['activation']])

        print(self.model)
        

    def forward(self, x):
        self.normalization(x)
        return self.model(x)
    
    
    def generateAdv(self, x, y, atkType = 'PGD', eps = 0.0314, alpha=0.00784, steps=7):
        with torch.enable_grad():
            if atkType == 'PGD':
                atk = PGD(self.model, eps=eps, alpha=alpha, steps=steps)
            elif atkType == 'FGSM':
                atk = FGSM(self.model, eps=eps)
            elif atkType == 'CW':
                c = self.config.get('c', 1)
                kappa = self.config.get('kappa', 0)
                atk = CW(self.encoder, c = c, kappa=kappa, steps=steps)
            elif atkType == 'auto':
                atk = AutoAttack(self.encoder, norm="Linf", eps=eps, n_classes=self.config['num_classes'])
            elif atkType == 'square':
                atk = Square(self.encoder, eps=eps)
            elif atkType == 'eotpgd':
                eot_iter = self.config.get('eot_iter', 2)
                atk = EOTPGD(self.encoder, eps=eps, alpha=alpha, steps=steps, eot_iter=eot_iter)
            
            # elif atkType == 'modCW':
            #     c = self.config.get('c') if self.config.get('c') is not None else 1
            #     kappa = self.config.get('kappa') if self.config.get('kappa') is not None else 0
            #     early = self.config.get('early') if self.config.get('kappa') is not None else True
            #     atk = modCW(self.model, c = c, kappa=kappa, steps=steps, early_stop=early)
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
        atkType = self.config.get('atk', 'PGD')
        atk_step = self.config.get('steps', 7)
        adv_images = self.generateAdv(x=x, y=y, atkType=atkType, eps = self.config['eps'], steps=atk_step)
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
    

    def replace_activation(self, module, new_activation_fn):
        """
        Recursively replace all activation functions in the model with the given activation function.
        
        Args:
        - module: The module in which to replace the activation functions (e.g., self.model).
        - new_activation_fn: The new activation function to replace with (e.g., nn.ReLU()).
        """
        for name, child in module.named_children():
            if isinstance(child, (ReLU, GELU, LeakyReLU, SiLU)):
                # Replace activation function
                if new_activation_fn == ReLU and self.config['architecture'] == "eff2":
                    new_act = CustomReLU()
                else:
                    new_act = new_activation_fn()
                setattr(module, name, new_act)
            else:
                # Recur for child modules
                self.replace_activation(child, new_activation_fn)





class modCW(Attack):
    def __init__(self, model, c=1, kappa=0, steps=50, lr=0.01, early_stop=True):
        super().__init__("CW", model)
        self.c = c
        self.kappa = kappa
        self.steps = steps
        self.lr = lr
        self.supported_mode = ["default"]
        self.early_stop = early_stop

    def forward(self, images, labels):
        images = images.clone().detach().to(self.device)
        labels = labels.clone().detach().to(self.device)

        w = self.inverse_tanh_space(images).detach()
        w.requires_grad = True

        best_adv_images = images.clone().detach()
        best_L2 = 1e10 * torch.ones((len(images))).to(self.device)
        prev_cost = 1e10
        dim = len(images.shape)

        MSELoss = nn.MSELoss(reduction="none")
        Flatten = nn.Flatten()

        optimizer = optim.Adam([w], lr=self.lr)

        # Softmax와 loss 추적 리스트
        # softmax_values = []
        # loss_values = []

        # 라벨을 제외한 가장 높은 logit을 가진 클래스를 target으로 설정 (첫 스텝에서만)
        with torch.no_grad():
            outputs = self.get_logits(images)
            target_labels = torch.argmax(outputs * (1 - F.one_hot(labels, num_classes=outputs.shape[1])), dim=1)

        for step in range(self.steps):
            # Get adversarial images
            adv_images = self.tanh_space(w)

            # Calculate loss
            current_L2 = MSELoss(Flatten(adv_images), Flatten(images)).sum(dim=1)
            L2_loss = current_L2.sum()

            outputs = self.get_logits(adv_images)

            # f_loss 계산
            f_loss = self.f(outputs, target_labels).sum()
            # f_loss = self.f(outputs, labels).sum()

            cost = L2_loss + self.c * f_loss

            # Softmax 값 추적
            # softmax_out = F.softmax(outputs, dim=1).detach().cpu().numpy()
            # softmax_values.append(softmax_out)
            # print(softmax_out)
            

            # Loss 값 추적
            # loss_values.append(cost.item())

            optimizer.zero_grad()
            cost.backward()
            optimizer.step()

            # Update adversarial images
            pre = torch.argmax(outputs.detach(), 1)
            condition = (pre == target_labels).float()

            mask = condition * (best_L2 > current_L2.detach())
            best_L2 = mask * current_L2.detach() + (1 - mask) * best_L2

            mask = mask.view([-1] + [1] * (dim - 1))
            best_adv_images = mask * adv_images.detach() + (1 - mask) * best_adv_images

            if step % max(self.steps // 10, 1) == 0:
                if cost.item() > prev_cost and self.early_stop:
                    # return best_adv_images, softmax_values, loss_values
                    return best_adv_images
                prev_cost = cost.item()

        # return best_adv_images, softmax_values, loss_values
        return best_adv_images

    def tanh_space(self, x):
        return 1 / 2 * (torch.tanh(x) + 1)

    def inverse_tanh_space(self, x):
        return self.atanh(torch.clamp(x * 2 - 1, min=-1, max=1))

    def atanh(self, x):
        return 0.5 * torch.log((1 + x) / (1 - x))

    def f(self, outputs, labels):
        one_hot_labels = torch.eye(outputs.shape[1]).to(self.device)[labels]
        other = torch.max((1 - one_hot_labels) * outputs, dim=1)[0]
        real = torch.max(one_hot_labels * outputs, dim=1)[0]

        return torch.clamp((other - real), min=-self.kappa)


class AdversarialDataLoader:
    def __init__(self, dataloader, attack, device):
        """
        dataloader: 원본 데이터셋의 DataLoader
        attack: 적용할 공격
        device: GPU 또는 CPU
        """
        self.dataloader = dataloader
        self.attack = attack
        self.device = device

    def __iter__(self):
        """
        DataLoader의 iterator를 사용하여 배치를 처리
        """
        for imgs, labels in self.dataloader:
            # 이미지와 라벨을 지정된 장치로 이동 (GPU 또는 CPU)
            imgs, labels = imgs.to(self.device), labels.to(self.device)

            imgs.requires_grad_()
            with torch.enable_grad():  # gradient 활성화
                adv_imgs = self.attack(imgs, labels)
            
            # 공격된 이미지와 원본 라벨 반환
            yield adv_imgs, labels

    def __len__(self):
        """
        DataLoader의 크기를 반환
        """
        return len(self.dataloader)


# validation set만 취급
def getAdvData(model, atkConfig, dataset, batch_size, num_workers, device):
    if dataset == "CIFAR10":
        test_transform = transforms.Compose(
            [
                transforms.ToTensor(),
            ])
        raw = CIFAR10(root='~/data', train=False, download=True, transform=test_transform)
        raw_dataloader = DataLoader(raw, batch_size=batch_size, num_workers=num_workers)
    
    elif dataset == "ImageNet100":
        transform = transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.CenterCrop((224,224)),
            transforms.ToTensor(),
        ])
        raw = ImageFolder('~/data/ImageNet100/val', transform=transform)
        raw_dataloader = DataLoader(raw, batch_size=batch_size, num_workers=num_workers)
        
    elif dataset == "ImageNet":
        transform = transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.CenterCrop((224,224)),
            transforms.ToTensor(),
        ])
        raw = ImageFolder('~/data/ImageNet/2012/ILSVRC2012_img_val', transform=transform)
        raw_dataloader = DataLoader(raw, batch_size=batch_size, num_workers=num_workers)
        
    if atkConfig['atk'] == "FGSM":
        eps = atkConfig.get('eps') if atkConfig.get('eps') is not None else 8 / 255
        attack = FGSM(model, eps)
    
    elif atkConfig['atk'] == "PGD":
        eps = atkConfig.get('eps') if atkConfig.get('eps') is not None else 8 / 255
        alpha = atkConfig.get('alpha') if atkConfig.get('alpha') is not None else 2 / 255
        steps = atkConfig.get('steps') if atkConfig.get('steps') is not None else 7
        attack = PGD(model, eps, alpha, steps)
    
    elif atkConfig['atk'] == "CW":
        c = atkConfig.get('c') if atkConfig.get('c') is not None else 1
        kappa = atkConfig.get('kappa') if atkConfig.get('kappa') is not None else 0
        steps = atkConfig.get('steps') if atkConfig.get('steps') is not None else 50
        lr = atkConfig.get('lr') if atkConfig.get('lr') is not None else 0.01
        attack = CW(model, c, kappa, steps, lr)
        
    # adv_dataset = AdversarialDataset(raw, attack)
    adv_dataloader = AdversarialDataLoader(raw_dataloader, attack, device)
    # return DataLoader(adv_dataset, batch_size=batch_size, num_workers=num_workers)
    return adv_dataloader
    
        

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
        
        
        return model

    def forward(self, x):
        x = self.normalization(x)
        return self.cnn(x)



# define the LightningModule
class LitAutoEncoder(pl.LightningModule):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.correct_indices = []
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
        
    def forward(self, x):
        return self.encoder(x)

    def training_step(self, batch, batch_idx):
        # training_step defines the train loop.
        # it is independent of forward
        x, y = batch
        z = self.encoder(x)
        loss = nn.functional.cross_entropy(z, y)
        # Logging to TensorBoard (if installed) by default
        # self.log("train_loss", loss)
        # print(loss)

        if(self.config['adv']):
            advIdx = torch.randint(x.shape[0], (int(x.shape[0] * 0.2),))
            advExample = self.generateAdv(x[advIdx], y[advIdx])
            advZ = self.encoder(advExample)
            advLoss = nn.functional.cross_entropy(advZ, y[advIdx])
            self.log("train_loss", loss + advLoss)
            return loss + advLoss
        else:
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
                atk = PGD(self.encoder, eps=eps, alpha=alpha, steps=steps)
            elif atkType == 'FGSM':
                atk = FGSM(self.encoder, eps=eps)
            elif atkType == 'CW':
                c = self.config.get('c', 1)
                kappa = self.config.get('kappa', 0)
                atk = CW(self.encoder, c = c, kappa=kappa, steps=steps)
            elif atkType == 'auto':
                atk = AutoAttack(self.encoder, norm="Linf", eps=eps, n_classes=self.config['num_classes'])
            elif atkType == 'square':
                atk = Square(self.encoder, eps=eps)
            elif atkType == 'eotpgd':
                eot_iter = self.config.get('eot_iter', 2)
                atk = EOTPGD(self.encoder, eps=eps, alpha=alpha, steps=steps, eot_iter=eot_iter)
            elif atkType == 'one_pixel':
                atk = OnePixel(self.encoder, pixels=5, steps=50, popsize=100)
            elif atkType == "pixle":
                atk = Pixle(self.encoder, restarts=100, max_iterations=20)
            # elif atkType == 'modCW':
            #     c = self.config.get('c', 1)
            #     kappa = self.config.get('kappa', 0)
            #     early = self.config.get('early', True)
            #     atk = modCW(self.encoder, c = c, kappa=kappa, steps=steps, early_stop=early)
            adv_images = atk(x, y)
        return adv_images


    def evaluateRobust(self, x, y):
        atkType = self.config.get('atk', 'PGD')
        atk_step = self.config.get('steps', 7)
        adv_images = self.generateAdv(x=x, y=y, atkType=atkType, eps = self.config['eps'], steps=atk_step)
        logits = self.encoder(adv_images)
        loss = nn.functional.cross_entropy(logits, y)
        preds = torch.argmax(logits, dim=1)
        acc = accuracy(preds, y, num_classes=self.config["num_classes"], task="multiclass")
        self.log("Robust_acc", acc, prog_bar=True, sync_dist=True)
        self.log("Robust_loss", loss, prog_bar=False, sync_dist=True)


    def evaluate(self, batch, batch_idx, stage=None):
        x, y = batch
        logits = self.encoder(x)
        loss = nn.functional.cross_entropy(logits, y)
        preds = torch.argmax(logits, dim=1)
        acc = accuracy(preds, y, num_classes=self.config["num_classes"], task="multiclass")

        # 예측값과 정답 비교 (맞춘 경우 True, 틀린 경우 False)
        is_correct = (preds == y)

        # 맞춘 데이터의 인덱스 계산 및 저장
        # batch_size = x.size(0)
        batch_size = self.config['batch_size']
        # print(f"batch_idx = {batch_idx} * batch_size = {batch_size} = {batch_idx * batch_size}")
        correct_indices_in_batch = [batch_idx * batch_size + i for i, correct in enumerate(is_correct) if correct]
        self.correct_indices.extend(correct_indices_in_batch)


        if stage:
            self.log(f"{stage}_loss", loss, prog_bar=True, sync_dist=True)
            self.log(f"{stage}_acc", acc, prog_bar=True, sync_dist=True)
            if self.config["adv"]:
                self.evaluateRobust(x, y)

        return loss

    def validation_step(self, batch, batch_idx):
        return self.evaluate(batch, batch_idx, "val")
    
    def get_correct_indices(self):
        correct_indices = self.correct_indices
        self.correct_indices = []  # 초기화
        return correct_indices

    def test_step(self, batch, batch_idx):
        return self.evaluate(batch, batch_idx, "test")


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



def load_model(ckpt, config, map_location = None):
    if config['architecture'] in ['resnet18', 'resnet50', 'resnet101']:
        model = LitAutoEncoder.load_from_checkpoint(checkpoint_path=ckpt, config=config, map_location=map_location)
    else:
        model = TransformerClassifier.load_from_checkpoint(checkpoint_path=ckpt, config=config, map_location=map_location)
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
    

# from torchvision.transforms import ToPILImage
# import matplotlib.pyplot as plt

def onePixelAttack(loader):
    dataset = loader.dataset
    transformed_dataset = TransformSubset(dataset)
    return DataLoader(transformed_dataset, batch_size=loader.batch_size, shuffle=False, num_workers=loader.num_workers)

def onePixelAttackWithSubset(loader, idx):
    dataset = loader.dataset
    subset = Subset(dataset=dataset, indices=idx)
    subset = TransformSubset(dataset)
    return DataLoader(subset, batch_size=loader.batch_size, shuffle=False, num_workers=loader.num_workers)
    

