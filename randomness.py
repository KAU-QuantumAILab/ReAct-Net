from modules import load_model, seed_everything, choose_dataset
import pandas as pd
import torch
import lightning.pytorch as pl
import os, wandb
from torch.nn import ReLU
from models import BReLU


torch.set_float32_matmul_precision('high')
seed_everything(42)
gpu_num = [1]

# config 파일 생성용
def make_config(**kwargs):
    act = kwargs.get('act')
    replaceAll = True if kwargs.get('replaceALL') == 'all' else False
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

def printConfig(cfg):
    string = f"""
    ===========================================================
    architecture : {cfg['architecture']}
    dataset : {cfg['dataset']}
    activation : {cfg['activation']} => {'relu' if cfg['activation'] == 'brelu' else 'brelu'}
    replaceALL : {cfg['replaceAll']}
    seed : {cfg['seed']}
    
    atk : {cfg['atk']}
    steps : {cfg.get('steps')}
    """
    print(string)
    return 

# relu와 brelu를 바꾸는 함수
# def reluBrelu(model, target):
#     for name, module in model.named_children():
#         if len(list(module.children())) > 0:
#             reluBrelu(module, target)
#         if isinstance(module, target[0]):
#             replace = target[1]()
#             setattr(model, name, replace)
            
            
def reluBrelu(model, target, count, current_count = 0, reverse = False):
    children = list(model.named_children())
    if reverse: children = reversed(children)
    
    for name, module in children:
        if current_count >= count: break
        
        if len(list(module.children())) > 0:
            current_count = reluBrelu(module, target, count, current_count, reverse)
            
        if isinstance(module, target[0]) and current_count < count:
            replace = target[1]()
            setattr(model, name, replace)
            current_count += 1
    
    return current_count

def count_act(model, target_class):
    count = 0
    
    for module in model.modules():
        if isinstance(module, target_class):
            count += 1
    
    return count



# brelu의 랜덤성이 성능에 좋은 영향을 주었는지 확인
# relu에 랜덤성을 주면 더 좋은지 확인
# 먼저 pretrain 모델 load -> 활성화함수 변경 -> 성능평가 (FGSM, PGD)
# attack_test2처럼 파일 경로 받고 split해서 cfg 만들고 load해서 활성화함수 교체 후 test


def randomness_test(ckpt_root_dir = './ckpt/CIFAR10_interpolated', 
             fgsm = True, pgd = True, pgd_step = 3,
             log_wandb = False, project_name='attack_test', csv_file_name = "attak_test.csv"):
    
    pgd_steps = range(1, pgd_step + 1)
    result_dict = dict()
    reverse = [False, True]
    
    for (root, directoriesm, files) in os.walk(ckpt_root_dir):
        for file in files:
            temp_dict = dict()
            
            ckpt_file_path = os.path.join(root, file)
            
            info = root.split('/')
            replaceALL = info[3]
            opt = info[4]
            act = info[5]
            seed = info[6]
            
            change = 'relu' if act == 'brelu' else 'brelu'
            temp_dict['activation'] = f"{act} => {change}"
            temp_dict['replaceALL'] = replaceALL
            temp_dict['optimizer'] = opt
            temp_dict['seed'] = seed
            temp_dict['from'] = act
            temp_dict['to'] = change
            row_name = f"{temp_dict['activation']}_{replaceALL}_{opt}_{seed}"
            
            cfg = make_config(
                        act=act,
                        opt=opt,
                        replaceALL=replaceALL,
                        adv=True,
                        seed=seed
                    )
            
            
            
            _, valloader = choose_dataset(cfg)
            trainer = pl.Trainer(inference_mode=False, devices=gpu_num)
            for rev in reverse:
                model = load_model(ckpt=ckpt_file_path, config=cfg)
                target = (ReLU, BReLU) if model.config['activation'] == 'relu' else (BReLU, ReLU)
                act_count = count_act(model, target[0])
                for change_count in range(act_count + 1):
                    cfg['change_reverse'] = rev
                    cfg['num_change'] = change_count
                    cfg['from'] = act
                    cfg['to'] = change
                    if log_wandb:
                        wandb.init(
                            project=project_name,
                            entity="kau-quantum",
                            name=row_name,
                            config=cfg
                        )
                        
                    reluBrelu(model, target, change_count, change_count - 1, rev)    # relu와 BReLU를 교체
                    print(model)

                    # 우선 FGSM
                    if fgsm:
                        model.config['atk'] = 'FGSM'
                        printConfig(model.config)
                        print(f"    from({act}) : {act_count}, to({change}) : {change_count}\n")
                        fgsm_result = trainer.validate(model, valloader)[0]
                        print("    ===========================================================\n")
                        temp_dict['pure_acc'] = fgsm_result['val_acc']
                        temp_dict['FGSM'] = fgsm_result['Robust_acc']
                        if log_wandb:
                            wandb.log({"pure_acc" : fgsm_result['val_acc'],
                                        "pure_loss": fgsm_result['val_loss'],
                                        "FGSM_acc" : fgsm_result['Robust_acc'],
                                        "FGSM_loss": fgsm_result['Robust_loss']},
                                        )
                    
                    if pgd:
                        model.config['atk'] = 'PGD'
                        for i in pgd_steps:
                            col_name = f"PGD {i}"
                            model.config['steps'] = i
                            printConfig(model.config)
                            print(f"    from({act}) : {act_count}, to({change}) : {change_count}\n")
                            pgd_result = trainer.validate(model, valloader)[0]
                            print("    ===========================================================\n")
                            temp_dict[col_name] = pgd_result['Robust_acc']
                            if log_wandb:
                                wandb.log({"PGD_acc" : pgd_result['Robust_acc'],
                                            "PGD_loss": pgd_result['Robust_loss']})
                        
                    if log_wandb:
                        wandb.finish()
                        
                    result_dict[row_name] = temp_dict
                    df = pd.DataFrame.from_dict(result_dict, orient='index')
                    df.to_csv(csv_file_name)
    
    # 최종 저장
    df = pd.DataFrame.from_dict(result_dict, orient='index')
    # print(df)
    df.to_csv(csv_file_name)
    return                


if __name__=='__main__':
    randomness_test(
        ckpt_root_dir = './ckpt/CIFAR10',
        fgsm = True,
        pgd = True, pgd_step = 20,
        log_wandb = True, project_name = 'relu_Brelu_change_slow',
        csv_file_name = 'relu_Brelu_change_slow.csv'
    )