from modules import load_model, getAdvData, seed_everything, choose_dataset
import torch
import lightning.pytorch as pl

torch.set_float32_matmul_precision('high')
seed_everything(42)
gpu_num = [3]
map_location = 'cuda:3'

atk_config = {
    'atk' : 'PGD',
    'steps' : 7
}

def make_config(**kwargs):
    act = kwargs.get('act')
    replaceAll = True if kwargs.get('replaceALL') == 'all' or kwargs.get('replaceALL') == True else False
    alpha = kwargs.get('alpha')
    adv = kwargs.get('adv')
    atk_type = kwargs.get('atk_type')
    steps = kwargs.get('steps')
    eps = kwargs.get('eps') if kwargs.get('eps') is not None else 0.0314
    opt = kwargs.get('opt')
    seed = kwargs.get('seed')
    cfg = {
        'architecture' : 'resnet18',
        'optimizer' : opt,
        'activation' : act,
        'replaceAll' : replaceAll, 
        'dataset' : 'CIFAR10',
        'batch_size' : 512,
        'num_workers' : 16,
        'adv' : adv,
        'eps' : eps,
        'alpha' : alpha,
        'atk' : atk_type,
        'steps' : steps,
        'seed' : seed
    }
    return cfg


relu_cfg = make_config(act = 'relu', replaceALL = 'all', adv = False, opt="AdamW", atk_type="PGD", eps=0.0314)
brelu_cfg = make_config(act = 'brelu', replaceALL = 'all', adv = False, opt="AdamW", atk_type="PGD", eps=0.0314)
lrelu_cfg = make_config(act = 'LReLU', replaceALL = 'all', adv = False, opt="AdamW", atk_type="PGD", eps=0.0314)

relu_ckpt = './ckpt_pgd7_re/CIFAR10/all/AdamW/relu/42/relu_ALL_CIFAR10_AdamW_epoch=58_Robust_acc=0.4755.ckpt'
brelu_ckpt = './ckpt_pgd7_re/CIFAR10/all/AdamW/brelu/42/brelu_ALL_CIFAR10_AdamW_epoch=91_Robust_acc=0.6268.ckpt'
lrelu_ckpt = './ckpt_pgd7/CIFAR10/all/AdamW/LReLU/42/LReLU_ALL_CIFAR10_AdamW_epoch=55_Robust_acc=0.4792.ckpt'

relu_model = load_model(relu_ckpt, relu_cfg, map_location)
brelu_model = load_model(ckpt=brelu_ckpt, config=brelu_cfg, map_location=map_location)
lrelu_model = load_model(ckpt=lrelu_ckpt, config=lrelu_cfg, map_location=map_location)

adv_example = getAdvData(brelu_model, atk_config, "CIFAR10" ,512, 16)

_, valloader = choose_dataset({"dataset" : "CIFAR10", "batch_size" : 512, "num_workers":16})

# sample = next(iter(adv_example))

# print(sample)

trainer = pl.Trainer(inference_mode=False, devices=gpu_num)

trainer.validate(lrelu_model, adv_example)
# trainer.validate(brelu_model, valloader)


# brelu -> brelu : 61.7%
# brelu -> ReLU : 71.1%
# brelu -> LReLU: 71.5%

# ReLU -> ReLU : 46.3%
# ReLU -> brelu : 64.5%
# ReLU -> LReLU : 63.1%

#LReLU -> LReLU : 46.5%
#LReLU -> BReLU : 62.7%
#LReLU -> ReLU : 62.1%

# 다른 모델로 adv_example을 생성하면 정확도가 높아짐 -> 당연함, 공격 대상이 다르니까
# 가지고 있는 act들 모두 비교하는게 좋을 듯
# brelu 모델을 이용해서 만든 adv_example을 썼을 때 다른 함수의 모델에서 정확도가 더 높음
# 공격을 방해하는 것은 맞는 것 같음
# PGD 7 더 해보고 CW도 해보는 것이 좋을 듯
# 내일 반복문으로 바꿔서 리그전 방식으로 다 공격해보기