from modules import load_model, choose_dataset, seed_everything, onePixelAttack, onePixelAttackWithSubset
import lightning.pytorch as pl
import torch
import pandas as pd
import wandb
import os
from setproctitle import setproctitle

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
    eps = kwargs.get('eps')
    opt = kwargs.get('opt')
    seed = kwargs.get('seed')
    c = kwargs.get('c')
    kappa = kwargs.get('kappa')
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
        'seed' : seed,
    }
    if atk_type == 'CW':
        cfg['c'] = c
        cfg['kappa'] = kappa
    return cfg

# fgsm attack test 함수
def fgsm_test(name, opt, ckpt, replace, alpha, seed):
    cfg = make_config(act=name, opt = opt, replaceALL=replace, alpha=alpha, adv=True, atk_type='FGSM', eps=0.0314, seed=seed)

    model = load_model(ckpt=ckpt, config=cfg)
    _, valloader = choose_dataset(config=cfg)
    trainer = pl.Trainer(inference_mode=False, devices = gpu_num)
    result = trainer.validate(model, valloader)
    return result[0]

# pgd attack test 함수
def pgd_test(name, opt, ckpt, replace, alpha, seed, steps = 3):
    cfg = make_config(act=name, opt=opt, replaceALL=replace, alpha=alpha, adv=True, atk_type='PGD', steps=steps, eps=0.0314, seed=seed)

    model = load_model(ckpt=ckpt, config=cfg)
    _, valloader = choose_dataset(config=cfg)
    trainer = pl.Trainer(inference_mode=False, devices = gpu_num)
    result = trainer.validate(model, valloader)
    return result[0]

# RDA Attack test 함수
def randomDotAttackTest(act, opt, ckpt, replace, alpha, temp_dict, log_wandb, seed, steps = 30):
    # load model -> evaluate -> record -> attack -> evaluate (repeat)
    cfg = make_config(act=act, opt=opt, replaceALL=replace, alpha=alpha, adv=False, atk_type='RIA', seed=seed)
    model = load_model(ckpt=ckpt, config=cfg)
    print(f"\n Act:{act}, alpha={alpha}, opt:{opt}, , replaceALL:{replace}\n")
    trainer = pl.Trainer(devices = gpu_num)
    _, valloader = choose_dataset(cfg)
    # num_data = len(valloader.dataset)
    for i in range(0, steps + 1):
        print(f"\n{i} Dots Attack, \t model : Act:{act}, alpha={alpha}, opt:{opt}, , replaceALL:{replace}")
        col_name = f"RDA_{i}"
        result = trainer.validate(model=model, dataloaders=valloader)[0]
        cor_idx = model.get_correct_indices()
        temp_dict[col_name] = result['val_acc']
        if log_wandb:
            wandb.log({"RDA_acc" : result['val_acc'],
                        "RDA_loss": result['val_loss']})
        if result['val_acc'] == 0: break
        valloader = onePixelAttackWithSubset(valloader, cor_idx)
    
    return temp_dict


def CW_test(act, opt, ckpt, replace, alpha, seed, c = 1, kappa = 0, steps = 50):
    cfg = make_config(act=act, opt=opt, replaceALL=replace, alpha=alpha, adv=True, atk_type = 'CW', seed=seed, c=c, kappa=kappa, steps=steps)
    
    model = load_model(ckpt=ckpt, config=cfg)
    _, valloader = choose_dataset(config=cfg)
    trainer = pl.Trainer(inference_mode=False, devices=gpu_num)
    result = trainer.validate(model, valloader)
    return result[0]


# 전체 테스트 묶어놓은 함수
def all_test(ckpt_root_dir = './ckpt/CIFAR10_interpolated', 
             fgsm = True, pgd = True, pgd_step = 3, rda=True, rda_step = 30,
             cw = True, cw_c = 1, cw_kappa = 0, cw_step = 50,
             log_wandb = False, project_name='attack_test', csv_file_name = "attak_test.csv"):
    # alpha 있는 함수 분별하기 위한 리스트
    variable = ['Vbrelu', 'leakyVbrelu', 'PVbrelu']
    # pgd attack 범위
    if isinstance(pgd_step, int):
        pgd_steps = range(1, pgd_step+1)
    elif isinstance(pgd_step, list):
        pgd_steps = pgd_step
    result_dict = dict()                # 최종 결과 모아놓을 딕셔너리

    runs = 0
    now = 0
    for (root, directories, files) in os.walk(ckpt_root_dir):
        for file in files:
            runs += 1
    
    
    # 체크포인트 모아놓은 디렉토리 모두 순회
    for (root, directories, files) in os.walk(ckpt_root_dir):
        for file in files:
            now += 1
            setproctitle(f"{now} / {runs} {now / runs * 100:.2f} %: attack testing")


            temp_dict = dict()          # 해당 체크포인트 결과 모아놓을 딕셔너리
            
            ckpt_file_path = os.path.join(root, file)       # 체크포인트 파일 경로
            
            info = root.split('/')  # ['.', 'ckpt', 'CIFAR10_interpolated', Replace(all / part), Optim, Act]
            replaceALL =replace = info[3]       # 리스트 3번째에서 전체 교체 여부
            # replaceALL = (replace == 'all') # 전체 교체면 True
            opt = info[4]           # optimizer 정보
            act = info[5]           # activation 정보
            seed = info[6]
            alpha = info[-1].lstrip('a=') if act in variable else 'NULL'    # a=5 에서 a= 제거
            
            temp_dict['activation'] = act
            temp_dict['alpha'] = alpha
            temp_dict['replace'] = replace
            temp_dict['optimizer'] = opt
            temp_dict['seed'] = seed
            row_name = f"{act}{'a=' if alpha != 'NULL' else ''}{alpha if alpha != 'NULL' else ''}_{replace}_{opt}_{seed}"
            
            a = float(alpha) if alpha != 'NULL' else None       # config에 쓸 수 있게 문자열에서 숫자로 변경
            # wandb 사용시 저장 위치
            if log_wandb:
                wandb.init(
                        project=project_name,
                        entity="kau-quantum",
                        name=row_name,
                        config=make_config(
                            act=act,
                            opt=opt,
                            replaceALL=replaceALL,
                            adv=True,
                            seed=seed
                        )
                    )
            
            
            if fgsm:
                fgsm_result = fgsm_test(name=act, opt=opt, ckpt=ckpt_file_path, replace=replaceALL, alpha=a, seed=seed)
                temp_dict['pure_acc'] = fgsm_result['val_acc']
                temp_dict['FGSM'] = fgsm_result['Robust_acc']
                if log_wandb:
                    wandb.log({"pure_acc" : fgsm_result['val_acc'],
                                "pure_loss": fgsm_result['val_loss'],
                                "FGSM_acc" : fgsm_result['Robust_acc'],
                                "FGSM_loss": fgsm_result['Robust_loss']},
                                )
                    
            if pgd:
                for i in pgd_steps:
                    col_name = f"PGD {i}"
                    pgd_result = pgd_test(name=act, opt=opt, ckpt=ckpt_file_path, replace=replaceALL, alpha=a, steps=i, seed=seed)
                    temp_dict[col_name] = pgd_result['Robust_acc']
                    if log_wandb:
                        wandb.log({"PGD_acc" : pgd_result['Robust_acc'],
                                    "PGD_loss": pgd_result['Robust_loss']})
                        
            if rda:
                print(f"\n Act:{act}, opt:{opt}, replaceALL:{replaceALL}, seed:{seed}\n")
                temp_dict = randomDotAttackTest(act=act, opt=opt, ckpt=ckpt_file_path, replace=replaceALL, 
                                                alpha=a, temp_dict=temp_dict, log_wandb=log_wandb, steps=rda_step, seed=seed)
                
            if cw:
                cw_result = CW_test(act=act, opt=opt, ckpt=ckpt_file_path, replace=replaceALL, alpha=alpha,
                                    seed=seed, c = cw_c, kappa=cw_kappa, steps=cw_step)
                temp_dict['cw_acc'] = cw_result['Robust_acc']
                if log_wandb:
                    wandb.config['cw_c'] = cw_c
                    wandb.config['cw_kappa'] = cw_kappa
                    wandb.config['cw_step'] = cw_step
                    wandb.log({'CW_acc': cw_result['Robust_acc'],
                               'CW_loss' : cw_result['Robust_loss']})
                
            
            if log_wandb:
                wandb.finish()
            
            result_dict[row_name] = temp_dict
            # 중간 저장
            df = pd.DataFrame.from_dict(result_dict, orient='index')
            # print(df)
            df.to_csv(csv_file_name)
            
    # 최종 저장
    df = pd.DataFrame.from_dict(result_dict, orient='index')
    # print(df)
    df.to_csv(csv_file_name)
    return


if __name__=='__main__':
    # pgd_step = list(range(1, 10)) + list(range(10, 101, 10))
    pgd_step = 7
    all_test(
        ckpt_root_dir = './ckpt_pgd7_re/CIFAR10',
        fgsm = False,
        pgd = True, pgd_step = pgd_step,
        rda = False, rda_step = 100,
        cw=False, cw_c= 100, cw_kappa = 0, cw_step = 40,
        log_wandb = False, project_name = 'Adv50_pgd7_CWc100_test',
        csv_file_name = 'Adv50_pgd7_re_test.csv'
    )
    # print(list(range(1, 10)) + list(range(10, 101, 10)))