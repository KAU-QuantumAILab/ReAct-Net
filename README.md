# ReAct-Net

## 1.Prerequisites
```
pip install -r requirements.txt
```

## 2.Training

- ```--model``` : ```resnet18``` / ```resnet50``` / ```resnet101```
- ```--dataset``` : ```CIFAR10``` / ```ImageNet```

#### Optional

- ```--react``` : ResNet의 ReLU 함수를 ReAct Layer로 변경
- ```--wandb``` : WandbLogger 사용
- ```--lr_scheduler``` : [OneCycleLR Scheduler](https://pytorch.org/docs/stable/generated/torch.optim.lr_scheduler.OneCycleLR.html) 사용
<br/>

- ```--lr``` : Learning Rate 변경 (Default ```0.001```)
- ```--optimizer``` : Optimizer 변경 ```SGD``` / ```Adam``` / ```AdamW``` (Default ```Adam```)
- ```--batchsize``` : Batch Size 변경 (Default ```256```)

#### UseAge
```python network_lightning.py --model resnet18 --dataset ImageNet --optimizer SGD```
