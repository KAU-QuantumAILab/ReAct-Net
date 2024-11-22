import os
import argparse
import time
import lightning.pytorch as pl
import torch
import pandas as pd
import wandb
from setproctitle import setproctitle
from test_norm import load_model, load_dataset, seed_everything


torch.set_float32_matmul_precision('high')
seed_everything(42)
# gpu_num = [0]
# map_location = 'cuda:0'


# config 파일 생성용
def make_config(**kwargs):
    cfg = kwargs
    cfg['pretrain'] = False
    cfg['batch_size'] = 256
    cfg['num_workers'] = 16
    if cfg['dataset'] == 'CIFAR10':
        cfg['num_classes'] = 10
    elif cfg['dataset'] == 'ImageNet100':
        cfg['num_classes'] = 100
    elif cfg['dataset'] == 'ImageNet':
        cfg['num_classes'] = 1000
    
    return cfg


def printConfig(cfg):
    string = f"""
    ===========================================================
    architecture : {cfg['architecture']}
    dataset : {cfg['dataset']}
    activation : {cfg['activation']}
    replaceALL : {cfg['replaceALL']}
    seed : {cfg['seed']}
    
    atk : {cfg['atk']}
    steps : {cfg.get('steps')}
    """
    print(string)
    return 


# fgsm attack test 함수
def fgsm_test(name, opt, ckpt, replace, alpha, seed, dataset, model_norm, arch='resnet18'):
    cfg = make_config(
        architecture=arch,
        dataset=dataset,
        activation=name, 
        optimizer = opt, 
        replaceALL=replace, 
        alpha=alpha, 
        adv=True, 
        atk='FGSM', 
        eps=0.0314, 
        seed=seed,
        model_norm=model_norm
    )
    model = load_model(ckpt=ckpt, config=cfg, map_location=map_location)
    _, valloader = load_dataset(config=cfg)
    trainer = pl.Trainer(inference_mode=False, devices = gpu_num)
    printConfig(cfg)
    result = trainer.test(model, valloader)
    return result[0]


# pgd attack test 함수
def pgd_test(name, opt, ckpt, replace, alpha, seed, dataset, model_norm, eps=0.0314, steps = 3, arch='resnet18'):
    cfg = make_config(
        architecture=arch,
        dataset=dataset,
        activation=name,
        optimizer=opt,
        replaceALL=replace,
        alpha=alpha,
        adv=True,
        atk='PGD',
        steps=steps,
        eps=eps,
        seed=seed,
        model_norm=model_norm
    )
    model = load_model(ckpt=ckpt, config=cfg, map_location=map_location)
    _, valloader = load_dataset(config=cfg)
    trainer = pl.Trainer(inference_mode=False, devices = gpu_num)
    printConfig(cfg)
    result = trainer.test(model, valloader)
    return result[0]


# CW attack test 함수
def CW_test(act, opt, ckpt, replace, alpha, seed, dataset, model_norm,
            c = 1, kappa = 0, steps = 50, arch='resnet18'):
    cfg = make_config(
        architecture=arch,
        dataset=dataset,
        activation=act,
        optimizer=opt,
        replaceALL=replace,
        alpha=alpha,
        adv=True,
        atk = 'CW',
        seed=seed,
        c=c,
        kappa=kappa,
        steps=steps,
        model_norm=model_norm
    )
    model = load_model(ckpt=ckpt, config=cfg, map_location=map_location)
    _, valloader = load_dataset(config=cfg)
    trainer = pl.Trainer(inference_mode=False, devices=gpu_num)
    printConfig(cfg)
    result = trainer.test(model, valloader)
    return result[0]


# AutoAttack
def autoattack_test(act, opt, ckpt, replace, alpha, seed, dataset, model_norm, arch='resnet18'):
    cfg = make_config(
        architecture=arch,
        dataset=dataset,
        activation=act,
        optimizer=opt,
        replaceALL=replace,
        alpha=alpha,
        adv=True,
        atk='auto',
        eps=0.0314,
        seed=seed,
        model_norm=model_norm
    )
    model = load_model(ckpt=ckpt, config=cfg, map_location=map_location)
    _, valloader = load_dataset(config=cfg)
    trainer = pl.Trainer(inference_mode=False, devices=gpu_num)
    printConfig(cfg)
    result = trainer.test(model, valloader)[0]
    return result


def square_test(act, opt, ckpt, replace, alpha, seed, dataset, model_norm, arch='resnet18'):
    cfg = make_config(
        architecture=arch,
        dataset=dataset,
        activation=act,
        optimizer=opt,
        replaceALL=replace,
        alpha=alpha,
        adv=True,
        atk='square',
        eps=0.0314,
        seed=seed,
        model_norm=model_norm
    )
    model = load_model(ckpt=ckpt, config=cfg, map_location=map_location)
    _, valloader = load_dataset(config=cfg)
    trainer = pl.Trainer(inference_mode=False, devices=gpu_num)
    printConfig(cfg)
    result = trainer.test(model, valloader)[0]
    return result


def eotpgd_test(act, opt, ckpt, replace, alpha, seed, dataset, model_norm,
                eot_steps = 10, eot_iter=2, arch='resnet18'):
    cfg = make_config(
        architecture=arch,
        dataset=dataset,
        activation=act,
        optimizer=opt,
        replaceALL=replace,
        alpha=alpha,
        adv=True,
        atk='eotpgd',
        steps=eot_steps, 
        eot_iter=eot_iter,
        eps=0.0314,
        seed=seed,
        model_norm=model_norm
    )
    model = load_model(ckpt=ckpt, config=cfg, map_location=map_location)
    _, valloader = load_dataset(config=cfg)
    trainer = pl.Trainer(inference_mode=False, devices = gpu_num)
    printConfig(cfg)
    result = trainer.test(model, valloader)[0]
    return result


def one_pixel_test(act, opt, ckpt, replace, alpha, seed, dataset, model_norm, arch='resnet18'):
    cfg = make_config(
        architecture=arch,
        dataset=dataset,
        activation=act,
        optimizer=opt,
        replaceALL=replace,
        alpha=alpha,
        adv=True,
        atk='one_pixel',
        seed=seed,
        model_norm=model_norm
    )
    model = load_model(ckpt=ckpt, config=cfg, map_location=map_location)
    _, valloader = load_dataset(config=cfg)
    trainer = pl.Trainer(inference_mode=False, devices=gpu_num)
    printConfig(cfg)
    result = trainer.test(model, valloader)[0]
    return result


def pixle_test(act, opt, ckpt, replace, alpha, seed, dataset, model_norm, arch='resnet18'):
    cfg = make_config(
        architecture=arch,
        dataset=dataset,
        activation=act,
        optimizer=opt,
        replaceALL=replace,
        alpha=alpha,
        adv=True,
        atk='pixle',
        seed=seed,
        model_norm=model_norm
    )
    model = load_model(ckpt=ckpt, config=cfg, map_location=map_location)
    _, valloader = load_dataset(config=cfg)
    trainer = pl.Trainer(inference_mode=False, devices=gpu_num)
    printConfig(cfg)
    result = trainer.test(model, valloader)[0]
    return result


def extract_config_from_path(path_info):
    # 가능한 모델과 데이터셋 목록
    VALID_ARCHITECTURES = ['resnet18', 'wide_resnet28_10']  # 필요한 모델 추가
    VALID_DATASETS = ['CIFAR10', 'ImageNet', 'ImageNet100']  # 필요한 데이터셋 추가
    
    config = {}
    
    # architecture 찾기
    for part in path_info:
        if part.startswith('ckpt_'):
            config['architecture'] = part[5:]
            arch = part
            break
    
    # dataset 찾기
    for dataset in VALID_DATASETS:
        if dataset in path_info:
            config['dataset'] = dataset
            break
    
    # replaceALL 찾기 ('all' 또는 'part')
    config['replaceAll'] = 'all' in path_info
    config['model_norm'] = 'model_norm' in path_info
    
    # 나머지 정보들은 상대적 위치가 고정되어 있다고 가정
    for part in path_info:
        if part in ['AdamW', 'SGD', 'Adam']:  # 가능한 optimizer 목록
            config['optimizer'] = part
        elif part.isdigit():  # seed 찾기
            config['seed'] = int(part)
        elif part not in ['', '.','data_norm', 'model_norm', f'{arch}', 'CIFAR10', 'all', 'part']:
            # 위에서 처리한 항목들을 제외한 나머지를 activation으로 간주
            config['activation'] = part
    
    return config


# 전체 테스트 묶어놓은 함수
def all_test(ckpt_root_dir = './ckpt/CIFAR10_interpolated',
             fgsm = True, 
             pgd = True, pgd_step = 3,
             cw = True, cw_c = 1, cw_kappa = 0, cw_step = 50,
             autoattack = True,
             square = True,
             eotpgd = True, eot_steps = 10, eot_iter = 2,
             one_pixel = True,
             pixle = True,
             log_wandb = False, project_name='attack_test', csv_file_name = "attak_test.csv"):
    # alpha 있는 함수 분별하기 위한 리스트
    variable = ['Vbrelu', 'leakyVbrelu', 'PVbrelu']
    # pgd attack 범위
    pgd_steps = pgd_step if isinstance(pgd_step, list) else [pgd_step]
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
            
            info = root.split('/')  # ['.', 'ckpt', 'CIFAR10', Replace(all / part), Optim, Act]
            info = extract_config_from_path(info)
            arch = info['architecture']
            dataset = info['dataset']
            replaceALL = info['replaceAll']       # 리스트 3번째에서 전체 교체 여부
            opt = info['optimizer']           # optimizer 정보
            act = info['activation']           # activation 정보
            seed = info['seed']
            alpha = info['alpha'].lstrip('a=') if act in variable else 'NULL'    # a=5 에서 a= 제거
            model_norm = info['model_norm']
            
            temp_dict['model_norm'] = model_norm
            temp_dict['dataset'] = dataset
            temp_dict['activation'] = act
            temp_dict['alpha'] = alpha
            temp_dict['replace'] = replaceALL
            temp_dict['optimizer'] = opt
            temp_dict['seed'] = seed
            row_name = f"{arch}_{act}_{dataset}_{'all' if replaceALL else 'part'}{'_a=' if alpha != 'NULL' else ''}{alpha if alpha != 'NULL' else ''}_{opt}_{seed}"
            
            a = float(alpha) if alpha != 'NULL' else None       # config에 쓸 수 있게 문자열에서 숫자로 변경
            # wandb 사용시 저장 위치
            if log_wandb:
                wandb.init(
                        project=project_name,
                        entity="kau-quantum",
                        name=row_name,
                        config=make_config(
                            model_norm=model_norm,
                            dataset=dataset,
                            act=act,
                            opt=opt,
                            replaceALL=replaceALL,
                            adv=True,
                            seed=seed,
                            arch=arch
                        )
                    )
            
            
            if fgsm:
                fgsm_result = fgsm_test(
                    dataset=dataset,
                    name=act,
                    opt=opt,
                    ckpt=ckpt_file_path,
                    replace=replaceALL,
                    alpha=alpha,
                    seed=seed,
                    arch=arch,
                    model_norm=model_norm)
                temp_dict['clean_acc'] = fgsm_result['test_clean_acc']
                temp_dict['FGSM_acc'] = fgsm_result['test_FGSM_acc']
                if log_wandb:
                    wandb.log({"clean_acc" : fgsm_result['test_clean_acc'],
                                "FGSM_acc" : fgsm_result['test_FGSM_acc'],
                                })
                    
            if pgd:
                for i in pgd_steps:
                    col_name = f"PGD_{i}_acc"
                    pgd_result = pgd_test(
                        dataset=dataset,
                        name=act,
                        opt=opt, 
                        ckpt=ckpt_file_path, 
                        replace=replaceALL, 
                        alpha=alpha, 
                        steps=i, 
                        seed=seed, 
                        arch=arch,
                        model_norm=model_norm)
                    temp_dict[col_name] = pgd_result['test_PGD_acc']
                    if log_wandb:
                        wandb.log({col_name : pgd_result['test_PGD_acc']})
                        
                
            if cw:
                cw_result = CW_test(
                    dataset=dataset,
                    act=act, 
                    opt=opt, 
                    ckpt=ckpt_file_path, 
                    replace=replaceALL, 
                    alpha=alpha,
                    seed=seed, 
                    c = cw_c, 
                    kappa=cw_kappa, 
                    steps=cw_step, 
                    arch=arch,
                    model_norm=model_norm)
                temp_dict['cw_acc'] = cw_result['test_CW_acc']
                if log_wandb:
                    wandb.config['cw_c'] = cw_c
                    wandb.config['cw_kappa'] = cw_kappa
                    wandb.config['cw_step'] = cw_step
                    wandb.log({'CW_acc': cw_result['test_CW_acc']})
            
            
            if autoattack:
                autoattack_result = autoattack_test(
                    dataset=dataset,
                    act=act, 
                    opt=opt, 
                    ckpt=ckpt_file_path, 
                    replace=replaceALL,
                    alpha=alpha, 
                    seed=seed, 
                    arch=arch,
                    model_norm=model_norm)
                temp_dict['autoattack_acc'] = autoattack_result['test_auto_acc']
                if log_wandb:
                    wandb.log({'AutoAttack_acc': autoattack_result['test_auto_acc']})
            
            if square:
                square_result = square_test(
                    dataset=dataset,
                    act=act, 
                    opt=opt, 
                    ckpt=ckpt_file_path, 
                    replace=replaceALL,
                    alpha=alpha, 
                    seed=seed, 
                    arch=arch,
                    model_norm=model_norm)
                temp_dict['square_acc'] = square_result['test_square_acc']
                if log_wandb:
                    wandb.log({'Square_acc' : square_result['test_square_acc']})
            
            if one_pixel:
                one_pixel_result = one_pixel_test(
                    dataset=dataset,
                    act=act, 
                    opt=opt, 
                    ckpt=ckpt_file_path, 
                    replace=replaceALL,
                    alpha=alpha, 
                    seed=seed, 
                    arch=arch,
                    model_norm=model_norm)
                temp_dict['one_pixel_acc'] = one_pixel_result['test_one_pixel_acc']
                if log_wandb:
                    wandb.log({'one_pixel_acc' : one_pixel_result['test_one_pixel_acc']})
            
            if pixle:
                pixle_result = pixle_test(
                    dataset=dataset,
                    act=act, 
                    opt=opt, 
                    ckpt=ckpt_file_path, 
                    replace=replaceALL,
                    alpha=alpha, 
                    seed=seed, 
                    arch=arch,
                    model_norm=model_norm)
                temp_dict['pixle_acc'] = pixle_result['test_pixle_acc']
                if log_wandb:
                    wandb.log({'Pixle_acc' : pixle_result['test_pixle_acc']})
            
            if eotpgd:
                eotpgd_result = eotpgd_test(
                    dataset=dataset,
                    act=act, 
                    opt=opt, 
                    ckpt=ckpt_file_path, 
                    replace=replaceALL,
                    alpha=alpha, 
                    seed=seed, 
                    arch=arch,
                    eot_steps=eot_steps, 
                    eot_iter=eot_iter,
                    model_norm=model_norm)
                temp_dict['eotpgd_acc'] = eotpgd_result['test_eotpgd_acc']
                if log_wandb:
                    wandb.log({'EOTPGD_acc' : eotpgd_result['test_eotpgd_acc']})

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
    parser = argparse.ArgumentParser()
    parser.add_argument('--ckpt', help='ckpt root directory')
    parser.add_argument('-f', '--fgsm', dest='fgsm', action='store_true')
    parser.add_argument('-p', '--pgd', dest='pgd', action='store_true')
    parser.add_argument('-c', '--cw', dest='cw', action='store_true')
    parser.add_argument('-a', '--auto', dest='auto', action='store_true')
    parser.add_argument('-s', '--square', dest='square', action='store_true')
    parser.add_argument('-e', '--eotpgd', dest='eotpgd', action='store_true')
    parser.add_argument('-o', '--one_pixel', dest='one_pixel', action='store_true')
    parser.add_argument('-pi', '--pixle', dest='pixle', action='store_true')
    parser.add_argument('-w', '--wandb', dest='wandb', action='store_true')
    parser.add_argument('--project', default="WRN28-10_Attack_test", help="wandb project name")
    parser.add_argument('--csv', default=f"{time.strftime('%Y.%m.%d - %H:%M:%S')}", help='save csv file name')
    parser.add_argument('--devices', default=0, type=int)
    
    args = parser.parse_args()
    
    global gpu_num, map_location
    gpu_num = [args.devices]
    map_location = f"cuda:{args.devices}"
    
    pgd_step = [20, 100]
    all_test(
        ckpt_root_dir = args.ckpt,
        fgsm = args.fgsm,
        pgd = args.pgd, pgd_step = pgd_step,
        cw=args.cw, cw_c= 1, cw_kappa = 0, cw_step = 40,
        autoattack=args.auto,
        square=args.square,
        eotpgd=args.eotpgd, eot_steps=20, eot_iter=5,
        one_pixel=args.one_pixel,
        pixle=args.pixle,
        log_wandb = args.wandb, project_name = args.project,
        csv_file_name = f'{args.csv}.csv'
    )