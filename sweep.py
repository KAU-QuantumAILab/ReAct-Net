# %%
import os
import torch
# from torchvision import models
import lightning.pytorch as pl
from lightning.pytorch.loggers import WandbLogger, TensorBoardLogger 
from lightning.pytorch.callbacks import ModelCheckpoint, LearningRateMonitor
from modules import  CustomDataModule, ImageClassifier
from lightning.fabric import Fabric
import wandb
import os
import argparse
import utils


def main(config=None, WANDBLOG=None):
    
    dataset = CustomDataModule(config['dataset'], config['batch_size'], config["num_workers"])
    modified_resnet_encoder = ImageClassifier(config, dataset.num_classes, dataset.input_ch)
    if(WANDBLOG):
        wandb_logger = WandbLogger()
        wandb_logger.watch(modified_resnet_encoder.encoder.cnn, log="all")
        
    else:
        tb_logger = TensorBoardLogger(save_dir="logs/")

    lr_monitor = LearningRateMonitor(logging_interval='step')
    checkpoint_callback = ModelCheckpoint(monitor="val_acc", mode="max", save_weights_only=True)

    trainer = pl.Trainer(max_epochs = config["epochs"],logger= wandb_logger if WANDBLOG else tb_logger, callbacks=[lr_monitor])
    trainer.fit(modified_resnet_encoder, dataset)
    # trainer.test(model=modified_resnet_encoder,dataloaders=testloader)

if __name__ == "__main__":
    utils.torch_seed()
    
    torch.set_float32_matmul_precision('high')
    run = wandb.init()
    print("id : ",run.id)
    print("config",wandb.config)
    main(wandb.config, True)

