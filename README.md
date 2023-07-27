# ReAct-Net

## 1. make yaml file
- ```method``` : ```grid```, ```bayes```, ```random```
- ```name``` : sweep의 이름을 설정해주세요. wandb의 sweeps에 표시되는 이름입니다.
- ```metric```: ```goal```과 ```name```을 설정해주세요.
  - ```goal```: ```maximize```, ```minimize``` 중 하나를 골라주세요.
  - ```name```: metric의 이름을 입력해주세요. (ex: ```val_loss```, ```val_acc```)
- ```parameters```: 튜닝할 하이퍼 파라미터들과 아키텍쳐, 데이터셋을 입력해주세요
  - ```architecture```: ```resnet18```, ```resnet50```, ```resnet101```
  - ```dataset```: ```CIFAR10```, ```ImageNet```
  - ```lr```: learning_rate를 설정해주세요
  - ```activation```: ```sampling```, ```relu```
  - ```optimizer```: ```SGD```, ```Adam```,  ```AdamW```
  - ```momentum```: ```SGD```에 사용할 모멘텀을 설정해주세요
  - ```beta1```: ```Adam``` ```AdamW```에 사용할 beta1을 설정해주세요
  - ```beta2```: ```Adam``` ```AdamW```에 사용할 beta2를 설정해주세요
  - ```batch_size```: batch size를 설정해주세요
  - ```num_workers```: num_workers를 설정해주세요
  - ```lr_scheduler```: lr_scheduler를 사용하면 True, 아니면 False를 설정하세요
  - ```epochs```: 학습 epoch를 설정해주세요
  - ```adv```: adverserial attack을 사용할 지 설정하세요 (True, False)

yaml파일 예시
```
program: nl_sweep_yaml.py
method: grid
name: reaAct_sweep_CIFAR10_resnet50_SGD_adv
metric:
  goal: minimize
  name: val_loss

parameters:
  architecture:
    value: "resnet50"
  dataset:
    value: "CIFAR10"
  lr:
    values: [0.01, 0.001]
  activation:
    values: ["sampling", "relu"]
  optimizer: 
    value: "SGD"
  momentum:
    values: [0, 0.1, 0.5, 0.9]
  batch_size:
    value: 256
  num_workers:
    value: 24
  lr_scheduler:
    values: [True, False]
  epochs:
    value: 200
  adv:
    values: [True]
```

[sweep yaml 파일 작성법](https://docs.wandb.ai/guides/sweeps/define-sweep-configuration)


## 2.nl_sweep_yaml.py 실행법

### option
- ```--yaml```: yaml 파일 경로를 입력하세요
- ```--project_name```: wandb에 저장할 project명을 입력하세요 (Default: ```reAct_sweep_test```)
- ```--entity```: 누구의 wandb에 저장할지 입력하세요 (Default: ```kau-quantum```)
- ```--devices```: 어느 그래픽카드로 돌릴지 번호를 입력하세요. 분산 처리 오류로 -1을 추천 안함 (Default: ```0```) (ex: ```0```, ```1```, ```2```, ```-1```)

### Useage
```
python nl_sweep_yaml.py --yaml ./cifar10_resnet50_SGD_config.yaml --project_name reAct_sweep_resnet50 --devices 0
```
