# %%
import os
import torch
# from torchvision import models
import lightning.pytorch as pl
from lightning.pytorch.loggers import WandbLogger, TensorBoardLogger 
from lightning.pytorch.callbacks import ModelCheckpoint, LearningRateMonitor
from modules import  CustomDataModule, ImageClassifier
import wandb
import os
import argparse
import utils


def main(config, WANDBLOG):
    class_num = 100
    name_postfix = "relu" if config['activation'] == 'relu' else "RaCUN"
    dataset = CustomDataModule(config['dataset'], class_num, config['batch_size'], config["num_workers"])
    modified_resnet_encoder = ImageClassifier(config, dataset.num_classes, dataset.input_ch)

    if(WANDBLOG):
        wandb_logger = WandbLogger(project='RaCUN', entity='kau-quantum',
            config=config, save_code=True, log_model="all", name=config["dataset"] + str(class_num) + "-" + config["architecture"] + "-" + name_postfix)
        wandb_logger.watch(modified_resnet_encoder, log="all")
    else:
        tb_logger = TensorBoardLogger(save_dir="logs/")



    # train the model (hint: here are some helpful Trainer arguments for rapid idea iteration)
    # torch.set_float32_matmul_precision('medium')

    lr_monitor = LearningRateMonitor(logging_interval='step')
    checkpoint_callback = ModelCheckpoint(monitor="val_acc", mode="max")

    trainer = pl.Trainer(max_epochs = config["epochs"],logger= wandb_logger if WANDBLOG else tb_logger, callbacks=[checkpoint_callback,lr_monitor],log_every_n_steps= 15 if config["batch_size"] >= 512 else 50)
    trainer.fit(modified_resnet_encoder, dataset)
    # trainer.test(model=modified_resnet_encoder,dataloaders=testloader)

if __name__ == "__main__":
    utils.torch_seed()

    parser = argparse.ArgumentParser(description='RaCUN Network Training')
    parser.add_argument('--model',  help='resnet18 / resnet50 / resnet101 선택 가능')    # 필요한 인수를 추가
    parser.add_argument('--dataset',help='MNIST / CIFAR10 / ImageNet 선택가능')
    parser.add_argument('--racun', action='store_true')
    parser.add_argument('--wandb', action='store_true')
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--epsilon', type=float, default=8/255)
    parser.add_argument('--momentum', type=float, default=0)
    parser.add_argument('--optimizer', default="SGD", help="Adam / SGD / AdamW 선택가능")
    parser.add_argument('--batchsize', type=int, default=256)
    parser.add_argument('--lr_scheduler', action='store_true')
    parser.add_argument('--adv', action='store_true')
    parser.add_argument('--replace_all', action='store_true')
    parser.add_argument('--final_div_fac', type=float, default=1e3)
    parser.add_argument('--wd', type=float, default=5e-4)
    args = parser.parse_args()

    torch.set_float32_matmul_precision('high')

    config = {
    "optimizer" : args.optimizer,
    "lr": args.lr,
    "architecture": args.model,
    "momentum": args.momentum,
    "dataset": args.dataset,
    "epochs": 20,
    "batch_size" : args.batchsize,
    'activation' : "racun" if args.racun else "relu",
    "num_workers" : int(os.cpu_count() / 2),
    "lr_scheduler" : args.lr_scheduler,
    "adv" : args.adv,
    "replace_all" : args.replace_all,
    "adv_epsilon" : args.epsilon,
    "final_div_fac" : args.final_div_fac,
    "wd" : args.wd,
    }
    print(config)

    WANDBLOG = args.wandb

    main(config, WANDBLOG)