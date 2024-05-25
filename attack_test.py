from modules import load_model, choose_dataset, seed_everything
import lightning.pytorch as pl
import torch
import pandas as pd


torch.set_float32_matmul_precision('high')
seed_everything(42)

# key iteration -> make config from act_name
def fgsm_test(name, ckpt, replace, alpha):
    cfg = {
    'architecture' : 'resnet18',
    'activation' : name,
    'replaceAll' : replace, 
    'dataset' : 'CIFAR10',
    'batch_size' : 256,
    'num_workers' : 24,
    'adv' : True,
    'eps' : 0.0314,
    'alpha' : alpha,
    'atk' : 'FGSM',
    }
    model = load_model(ckpt=ckpt, config=cfg)
    _, valloader = choose_dataset(config=config)
    trainer = pl.Trainer(inference_mode=False)
    result = trainer.validate(model, valloader)
    return result[0]


def pgd_test(name, ckpt, replace, alpha, steps = 3):
    cfg = {
    'architecture' : 'resnet18',
    'activation' : name,
    'replaceAll' : replace, 
    'dataset' : 'CIFAR10',
    'batch_size' : 256,
    'num_workers' : 24,
    'adv' : True,
    'eps' : 0.0314,
    'alpha' : alpha,
    'atk' : 'PGD',
    'steps' : steps
    }

    model = load_model(ckpt=ckpt, config=cfg)
    _, valloader = choose_dataset(config=config)
    trainer = pl.Trainer(inference_mode=False)
    result = trainer.validate(model, valloader)
    return result[0]


def all_test(fgsm = True, pgd_step = 3, file_name = "attak_test.csv"):
    optim = ['SGD', 'Adam', 'AdamW']
    variable = ['Vbrelu', 'leakyVbrelu', 'PVbrelu']
    replace = [(ckpt_path, False), (all_ckpt_path, True)]
    pgd_steps = range(1, pgd_step+1)
    result_dict = dict()

    for ckpt_var, replaceALL in replace:
        for act_name, ckpt_dict in ckpt_var.items():
            if act_name not in variable:
                for opt in optim:
                    temp_dict = dict()
                    temp_dict['optimizer'] = opt
                    row_name = f"{act_name}{'(ALL)' if replaceALL else ''}-{opt}"
                    checkpoint = ckpt_dict[opt]

                    if fgsm:
                        fgsm_result = fgsm_test(act_name, checkpoint, replaceALL, None)
                        temp_dict['pure_acc'] = fgsm_result['val_acc']
                        temp_dict['FGSM'] = fgsm_result['Robust_acc']

                    if pgd_step != 0:
                        for i in pgd_steps:
                            col_name = f"PGD {i}"
                            pgd_result = pgd_test(act_name, checkpoint, replaceALL, None, i)
                            temp_dict[col_name] = pgd_result['Robust_acc']
                    
                    result_dict[row_name] = temp_dict

            else:
                alpha = [1.2, 5]
                for a in alpha:
                    for opt in optim:
                        temp_dict = dict()
                        temp_dict['optimizer'] = opt
                        alpha_string = f"_a={a}"
                        row_name = f"{act_name}{alpha_string}{'(ALL)' if replaceALL else ''}-{opt}"
                        checkpoint = ckpt_dict[a][opt]

                        if fgsm:
                            fgsm_result = fgsm_test(act_name, checkpoint, replaceALL, a)
                            temp_dict['pure_acc'] = fgsm_result['val_acc']
                            temp_dict['FGSM'] = fgsm_result['Robust_acc']

                        if pgd_step != 0:
                            for i in pgd_steps:
                                col_name = f"PGD {i}"
                                pgd_result = pgd_test(act_name, checkpoint, replaceALL, a, i)
                                temp_dict[col_name] = pgd_result['Robust_acc']
                        
                        result_dict[row_name] = temp_dict

    df = pd.DataFrame.from_dict(result_dict, orient='index')
    print(df)
    df.to_csv(file_name)
    return


config = {
    'architecture' : 'resnet18',
    'activation' : 'gelu',
    'replaceAll' : False, 
    'dataset' : 'CIFAR10',
    'batch_size' : 256,
    'num_workers' : 24,
    'adv' : True,
    'eps' : 0.0314,
    'alpha' : 5,
    'atk' : 'PGD',
    'steps' : 3
}


# old ver
# PGD 3Step result
# BottleNeck
# [SGD, Adam, AdamW]
# ckpt_path = {
#     'relu': {
#         'SGD': 'ckpt/CIFAR10/relu/relu_ALL_CIFAR10_SGD_epoch=97_val_acc=0.9467.ckpt',
#         'Adam': 'ckpt/CIFAR10/relu/relu_ALL_CIFAR10_Adam_epoch=93_val_acc=0.9407.ckpt',
#         'AdamW': 'ckpt/CIFAR10/relu/relu_ALL_CIFAR10_AdamW_epoch=96_val_acc=0.9448.ckpt',
#     },

#     'gelu' : {
#         'SGD': 'ckpt/CIFAR10/gelu/gelu_CIFAR10_SGD_epoch=95_val_acc=0.9482.ckpt',
#         'Adam': 'ckpt/CIFAR10/gelu/gelu-CIFAR10-val_acc=0.9403.ckpt',
#         'AdamW': 'ckpt/CIFAR10/gelu/gelu-CIFAR10-val_acc=0.9429.ckpt',
#         },

#     'silu' : {
#         'SGD': 'ckpt/CIFAR10/silu/silu-CIFAR10-val_acc=0.9481.ckpt',
#         'Adam': 'ckpt/CIFAR10/silu/silu-CIFAR10-val_acc=0.9401.ckpt',
#         'AdamW': 'ckpt/CIFAR10/silu/silu-CIFAR10-val_acc=0.9438.ckpt',
#         },

#     'elu' : {
#         'SGD': 'ckpt/CIFAR10/elu/elu-CIFAR10-val_acc=0.9519.ckpt',
#         'Adam': 'ckpt/CIFAR10/elu/elu-CIFAR10-val_acc=0.9390.ckpt',
#         'AdamW': 'ckpt/CIFAR10/elu/elu-CIFAR10-val_acc=0.9461.ckpt',
#         },

#     'LReLU' : {
#         'SGD': 'ckpt/CIFAR10/LReLU/LReLU-CIFAR10-val_acc=0.9501.ckpt',
#         'Adam': 'ckpt/CIFAR10/LReLU/LReLU-CIFAR10-val_acc=0.9414.ckpt',
#         'AdamW': 'ckpt/CIFAR10/LReLU/LReLU-CIFAR10-val_acc=0.9448.ckpt',
#         },

#     'PReLU' : {
#         'SGD': 'ckpt/CIFAR10/PReLU/PReLU-CIFAR10-val_acc=0.9428.ckpt',
#         'Adam': 'ckpt/CIFAR10/PReLU/PReLU-CIFAR10-val_acc=0.9374.ckpt',
#         'AdamW': 'ckpt/CIFAR10/PReLU/PReLU-CIFAR10-val_acc=0.9379.ckpt',
#         },


#     'brelu' : {
#         'SGD': 'ckpt/CIFAR10/brelu/brelu-CIFAR10-val_acc=0.9329.ckpt',
#         'Adam': 'ckpt/CIFAR10/brelu/brelu-CIFAR10-val_acc=0.9416.ckpt',
#         'AdamW': 'ckpt/CIFAR10/brelu/brelu-CIFAR10-val_acc=0.9391.ckpt',
#         },

#     'leaky_brelu' : {
#         'SGD': 'ckpt/CIFAR10/leaky_brelu/leaky-CIFAR10-val_acc=0.9370.ckpt',
#         'Adam': 'ckpt/CIFAR10/leaky_brelu/leaky-CIFAR10-val_acc=0.9412.ckpt',
#         'AdamW': 'ckpt/CIFAR10/leaky_brelu/leaky-CIFAR10-val_acc=0.9403.ckpt',
#         },

#     'pbrelu' : {
#         'SGD': 'ckpt/CIFAR10/pbrelu/pbrelu-CIFAR10-val_acc=0.9419.ckpt',
#         'Adam': 'ckpt/CIFAR10/pbrelu/pbrelu-CIFAR10-val_acc=0.9333.ckpt',
#         'AdamW': 'ckpt/CIFAR10/pbrelu/pbrelu-CIFAR10-val_acc=0.9308.ckpt',
#         },

#     'Vbrelu' : {
#         1.2 : {
#             'SGD': 'ckpt/CIFAR10/Vbrelu/Vbrelu_a=1.2-CIFAR10-val_acc=0.9319.ckpt',
#             'Adam': 'ckpt/CIFAR10/Vbrelu/Vbrelu_a=1.2-CIFAR10-val_acc=0.8901.ckpt',
#             'AdamW': 'ckpt/CIFAR10/Vbrelu/Vbrelua=1.2_CIFAR10_AdamW_epoch=99_val_acc=0.8914.ckpt',
#         },
#         5 : {
#             'SGD': 'ckpt/CIFAR10/Vbrelu/Vbrelu_a=5-CIFAR10-val_acc=0.9414.ckpt',
#             'Adam': 'ckpt/CIFAR10/Vbrelu/Vbrelu_a=5-CIFAR10-val_acc=0.9402.ckpt',
#             'AdamW': 'ckpt/CIFAR10/Vbrelu/Vbrelua=5_CIFAR10_AdamW_epoch=98_val_acc=0.9211.ckpt',
#         }
#     },

#     'leakyVbrelu' : {
#         1.2 : {
#             'SGD': 'ckpt/CIFAR10/leakyVbrelu/leakyVbrelu_a=1.2-CIFAR10-val_acc=0.9365.ckpt',
#             'Adam': 'ckpt/CIFAR10/leakyVbrelu/leakyVbrelu_a=1.2-CIFAR10-val_acc=0.9041.ckpt',
#             'AdamW': 'ckpt/CIFAR10/leakyVbrelu/leakyVbrelua=1.2_CIFAR10_AdamW_epoch=96_val_acc=0.9184.ckpt',
#         },
#         5 : {
#             'SGD': 'ckpt/CIFAR10/leakyVbrelu/leakyVbrelu_a=5-CIFAR10-val_acc=0.9439.ckpt',
#             'Adam': 'ckpt/CIFAR10/leakyVbrelu/leakyVbrelu_a=5-CIFAR10-val_acc=0.8961.ckpt',
#             'AdamW': 'ckpt/CIFAR10/leakyVbrelu/leakyVbrelua=5_CIFAR10_AdamW_epoch=98_val_acc=0.8925.ckpt',
#         }
#     },

#     'PVbrelu' : {
#         1.2 : {
#             'SGD': 'ckpt/CIFAR10/PVbrelu/PVbrelu_a=1.2-CIFAR10-val_acc=0.9446.ckpt',
#             'Adam': 'ckpt/CIFAR10/PVbrelu/PVbrelu_a=1.2-CIFAR10-val_acc=0.8713.ckpt',
#             'AdamW': 'ckpt/CIFAR10/PVbrelu/PVbrelua=1.2_CIFAR10_AdamW_epoch=97_val_acc=0.8872.ckpt',
#         },
#         5 : {
#             'SGD': 'ckpt/CIFAR10/PVbrelu/PVbrelu_a=5-CIFAR10-val_acc=0.9395.ckpt',
#             'Adam': 'ckpt/CIFAR10/PVbrelu/PVbrelu_a=5-CIFAR10-val_acc=0.8864.ckpt',
#             'AdamW': 'ckpt/CIFAR10/PVbrelu/PVbrelua=5_CIFAR10_AdamW_epoch=95_val_acc=0.8945.ckpt',
#         }
#     },
# }

# # ALL
# all_ckpt_path = {
#     'relu': {
#         'SGD': 'ckpt/CIFAR10/relu/relu_ALL_CIFAR10_SGD_epoch=97_val_acc=0.9467.ckpt',
#         'Adam': 'ckpt/CIFAR10/relu/relu_ALL_CIFAR10_Adam_epoch=93_val_acc=0.9407.ckpt',
#         'AdamW': 'ckpt/CIFAR10/relu/relu_ALL_CIFAR10_AdamW_epoch=96_val_acc=0.9448.ckpt',
#     },

#     'gelu' : {
#         'SGD': 'ckpt/CIFAR10/gelu/gelu_ALL-CIFAR10-val_acc=0.9449.ckpt',
#         'Adam': 'ckpt/CIFAR10/gelu/gelu_ALL-CIFAR10-val_acc=0.9405.ckpt',
#         'AdamW': 'ckpt/CIFAR10/gelu/gelu_ALL-CIFAR10-val_acc=0.9445.ckpt',
#         },

#     'silu' : {
#         'SGD': 'ckpt/CIFAR10/silu/silu_ALL-CIFAR10-val_acc=0.9399.ckpt',
#         'Adam': 'ckpt/CIFAR10/silu/silu_ALL-CIFAR10-val_acc=0.9436.ckpt',
#         'AdamW': 'ckpt/CIFAR10/silu/silu_ALL-CIFAR10-val_acc=0.9473.ckpt',
#         },

#     'elu' : {
#         'SGD': 'ckpt/CIFAR10/elu/elu_ALL-CIFAR10-val_acc=0.9294.ckpt',
#         'Adam': 'ckpt/CIFAR10/elu/elu_ALL-CIFAR10-val_acc=0.9399.ckpt',
#         'AdamW': 'ckpt/CIFAR10/elu/elu_ALL-CIFAR10-val_acc=0.9473.ckpt',
#         },

#     'LReLU' : {
#         'SGD': 'ckpt/CIFAR10/LReLU/LReLU_ALL-CIFAR10-val_acc=0.9504.ckpt',
#         'Adam': 'ckpt/CIFAR10/LReLU/LReLU_ALL-CIFAR10-val_acc=0.9416.ckpt',
#         'AdamW': 'ckpt/CIFAR10/LReLU/LReLU_ALL-CIFAR10-val_acc=0.9443.ckpt',
#         },

#     'PReLU' : {
#         'SGD': 'ckpt/CIFAR10/PReLU/PReLU_ALL-CIFAR10-val_acc=0.9457.ckpt',
#         'Adam': 'ckpt/CIFAR10/PReLU/PReLU_ALL-CIFAR10-val_acc=0.9314.ckpt',
#         'AdamW': 'ckpt/CIFAR10/PReLU/PReLU_ALL-CIFAR10-val_acc=0.9410.ckpt',
#         },


#     'brelu' : {
#         'SGD': 'ckpt/CIFAR10/brelu/brelu_ALL-CIFAR10-val_acc=0.7999.ckpt',
#         'Adam': 'ckpt/CIFAR10/brelu/brelu_ALL-CIFAR10-val_acc=0.9344.ckpt',
#         'AdamW': 'ckpt/CIFAR10/brelu/brelu_ALL-CIFAR10-val_acc=0.9175.ckpt',
#         },

#     'leaky_brelu' : {
#         'SGD': 'ckpt/CIFAR10/leaky_brelu/leaky_ALL-CIFAR10-val_acc=0.8101.ckpt',
#         'Adam': 'ckpt/CIFAR10/leaky_brelu/leaky_ALL-CIFAR10-val_acc=0.9318.ckpt',
#         'AdamW': 'ckpt/CIFAR10/leaky_brelu/leaky_ALL-CIFAR10-val_acc=0.9134.ckpt',
#         },

#     'pbrelu' : {
#         'SGD': 'ckpt/CIFAR10/pbrelu/pbrelu_ALL-CIFAR10-val_acc=0.8081.ckpt',
#         'Adam': 'ckpt/CIFAR10/pbrelu/pbrelu_ALL-CIFAR10-val_acc=0.9285.ckpt',
#         'AdamW': 'ckpt/CIFAR10/pbrelu/pbrelu_ALL-CIFAR10-val_acc=0.9151.ckpt',
#         },

#     # 1.2 [SGD, ADam, AdamW]
#     # 5 [SGD, ADam, AdamW]
#     'Vbrelu' : {
#         1.2 : {
#             'SGD': 'ckpt/CIFAR10/Vbrelu/Vbrelu_a=1.2_ALL-CIFAR10-val_acc=0.8147.ckpt',
#             'Adam': 'ckpt/CIFAR10/Vbrelu/Vbrelu_a=1.2_ALL-CIFAR10-val_acc=0.9260.ckpt',
#             'AdamW': 'ckpt/CIFAR10/Vbrelu/Vbrelua=1.2_ALL_CIFAR10_AdamW_epoch=94_val_acc=0.9012.ckpt',
#         },
#         5 : {
#             'SGD': 'ckpt/CIFAR10/Vbrelu/Vbrelu_a=5_ALL-CIFAR10-val_acc=0.9145.ckpt',
#             'Adam': 'ckpt/CIFAR10/Vbrelu/Vbrelu_a=5_ALL-CIFAR10-val_acc=0.9397.ckpt',
#             'AdamW': 'ckpt/CIFAR10/Vbrelu/Vbrelua=5_ALL_CIFAR10_AdamW_epoch=99_val_acc=0.9202.ckpt',
#         }
#     },

#     'leakyVbrelu' : {
#         1.2 : {
#             'SGD': 'ckpt/CIFAR10/leakyVbrelu/leakyVbrelu_a=1.2_ALL-CIFAR10-val_acc=0.8206.ckpt',
#             'Adam': 'ckpt/CIFAR10/leakyVbrelu/leakyVbrelu_a=1.2_ALL-CIFAR10-val_acc=0.9296.ckpt',
#             'AdamW': 'ckpt/CIFAR10/leakyVbrelu/leakyVbrelua=1.2_ALL_CIFAR10_AdamW_epoch=97_val_acc=0.9062.ckpt',
#         },
#         5 : {
#             'SGD': 'ckpt/CIFAR10/leakyVbrelu/leakyVbrelu_a=5_ALL-CIFAR10-val_acc=0.9150.ckpt',
#             'Adam': 'ckpt/CIFAR10/leakyVbrelu/leakyVbrelu_a=5_ALL-CIFAR10-val_acc=0.8986.ckpt',
#             'AdamW': 'ckpt/CIFAR10/leakyVbrelu/leakyVbrelua=5_ALL_CIFAR10_AdamW_epoch=98_val_acc=0.9234.ckpt',
#         }
#     },

#     'PVbrelu' : {
#         1.2 : {
#             'SGD': 'ckpt/CIFAR10/PVbrelu/PVbrelu_a=1.2_ALL-CIFAR10-val_acc=0.8199.ckpt',
#             'Adam': 'ckpt/CIFAR10/PVbrelu/PVbrelu_a=1.2_ALL-CIFAR10-val_acc=0.8759.ckpt',
#             'AdamW': 'ckpt/CIFAR10/PVbrelu/PVbrelua=1.2_ALL_CIFAR10_AdamW_epoch=96_val_acc=0.9005.ckpt',
#         },
#         5 : {
#             'SGD': 'ckpt/CIFAR10/PVbrelu/PVbrelu_a=5_ALL-CIFAR10-val_acc=0.9180.ckpt',
#             'Adam': 'ckpt/CIFAR10/PVbrelu/PVbrelu_a=5_ALL-CIFAR10-val_acc=0.9145.ckpt',
#             'AdamW': 'ckpt/CIFAR10/PVbrelu/PVbrelua=5_ALL_CIFAR10_AdamW_epoch=96_val_acc=0.9148.ckpt',
#         }
#     },
# }

# ckpt = all_ckpt_path[config['activation']] if config['replaceAll'] else ckpt_path[config['activation']]
# model = load_model(ckpt=ckpt, config=config)
# trainloader, valloader = choose_dataset(config=config)
# # print(model)

# trainer = pl.Trainer(inference_mode=False)
# result = trainer.validate(model, valloader)

# print(result)

ckpt_path = {
    'relu': {
        'SGD': 'ckpt/CIFAR10/relu/relu_ALL_CIFAR10_SGD_epoch=97_val_acc=0.9467.ckpt',
        'Adam': 'ckpt/CIFAR10/relu/relu_ALL_CIFAR10_Adam_epoch=93_val_acc=0.9407.ckpt',
        'AdamW': 'ckpt/CIFAR10/relu/relu_ALL_CIFAR10_AdamW_epoch=96_val_acc=0.9448.ckpt',
    },
    
    'gelu': {
        'SGD': 'ckpt/CIFAR10/gelu/gelu_CIFAR10_SGD_epoch=95_val_acc=0.9482.ckpt',
        'Adam': 'ckpt/CIFAR10/gelu/gelu_CIFAR10_Adam_epoch=99_val_acc=0.9414.ckpt',
        'AdamW': 'ckpt/CIFAR10/gelu/gelu_CIFAR10_AdamW_epoch=99_val_acc=0.9446.ckpt',
    },

    'silu': {
        'SGD': 'ckpt/CIFAR10/silu/silu_CIFAR10_SGD_epoch=99_val_acc=0.9499.ckpt',
        'Adam': 'ckpt/CIFAR10/silu/silu_CIFAR10_Adam_epoch=95_val_acc=0.9415.ckpt',
        'AdamW': 'ckpt/CIFAR10/silu/silu_CIFAR10_AdamW_epoch=98_val_acc=0.9447.ckpt',
    },

    'elu': {
        'SGD': 'ckpt/CIFAR10/elu/elu_CIFAR10_SGD_epoch=98_val_acc=0.9475.ckpt',
        'Adam': 'ckpt/CIFAR10/elu/elu_CIFAR10_Adam_epoch=91_val_acc=0.9402.ckpt',
        'AdamW': 'ckpt/CIFAR10/elu/elu_CIFAR10_AdamW_epoch=98_val_acc=0.9457.ckpt',
    },

    'LReLU': {
        'SGD': 'ckpt/CIFAR10/LReLU/LReLU_CIFAR10_SGD_epoch=96_val_acc=0.9490.ckpt',
        'Adam': 'ckpt/CIFAR10/LReLU/LReLU_CIFAR10_Adam_epoch=93_val_acc=0.9403.ckpt',
        'AdamW': 'ckpt/CIFAR10/LReLU/LReLU_CIFAR10_AdamW_epoch=98_val_acc=0.9444.ckpt',
    },
    
    'PReLU': {
        'SGD': 'ckpt/CIFAR10/PReLU/PReLU_CIFAR10_SGD_epoch=98_val_acc=0.9407.ckpt',
        'Adam': 'ckpt/CIFAR10/PReLU/PReLU_CIFAR10_Adam_epoch=95_val_acc=0.9401.ckpt',
        'AdamW': 'ckpt/CIFAR10/PReLU/PReLU_CIFAR10_AdamW_epoch=94_val_acc=0.9397.ckpt',
    },
    
    'brelu': {
        'SGD': 'ckpt/CIFAR10/brelu/brelu_CIFAR10_SGD_epoch=99_val_acc=0.9329.ckpt',
        'Adam': 'ckpt/CIFAR10/brelu/brelu_CIFAR10_Adam_epoch=93_val_acc=0.9407.ckpt',
        'AdamW': 'ckpt/CIFAR10/brelu/brelu_CIFAR10_AdamW_epoch=92_val_acc=0.9325.ckpt',
    },

    'leaky_brelu': {
        'SGD': 'ckpt/CIFAR10/leaky_brelu/leaky_brelu_CIFAR10_SGD_epoch=99_val_acc=0.9379.ckpt',
        'Adam': 'ckpt/CIFAR10/leaky_brelu/leaky_brelu_CIFAR10_Adam_epoch=98_val_acc=0.9381.ckpt',
        'AdamW': 'ckpt/CIFAR10/leaky_brelu/leaky_brelu_CIFAR10_AdamW_epoch=99_val_acc=0.9380.ckpt',
    },
    
    'pbrelu': {
        'SGD': 'ckpt/CIFAR10/pbrelu/pbrelu_CIFAR10_SGD_epoch=95_val_acc=0.9460.ckpt',
        'Adam': 'ckpt/CIFAR10/pbrelu/pbrelu_CIFAR10_Adam_epoch=97_val_acc=0.9355.ckpt',
        'AdamW': 'ckpt/CIFAR10/pbrelu/pbrelu_CIFAR10_AdamW_epoch=97_val_acc=0.9262.ckpt',
    },

    'Vbrelu': {
        1.2: {
            'SGD': 'ckpt/CIFAR10/Vbrelu/Vbrelu_a=1.2_CIFAR10_SGD_epoch=98_val_acc=0.9333.ckpt',
            'Adam': 'ckpt/CIFAR10/Vbrelu/Vbrelu_a=1.2_CIFAR10_Adam_epoch=96_val_acc=0.9407.ckpt',
            'AdamW': 'ckpt/CIFAR10/Vbrelu/Vbrelu_a=1.2_CIFAR10_AdamW_epoch=99_val_acc=0.9382.ckpt',
        },
        5: {
            'SGD': 'ckpt/CIFAR10/Vbrelu/Vbrelu_a=5_CIFAR10_SGD_epoch=97_val_acc=0.9400.ckpt',
            'Adam': 'ckpt/CIFAR10/Vbrelu/Vbrelu_a=5_CIFAR10_Adam_epoch=96_val_acc=0.9396.ckpt',
            'AdamW': 'ckpt/CIFAR10/Vbrelu/Vbrelu_a=5_CIFAR10_AdamW_epoch=96_val_acc=0.9430.ckpt',
        }
    },

    'leakyVbrelu': {
        1.2: {
            'SGD': 'ckpt/CIFAR10/leakyVbrelu/leakyVbrelu_a=1.2_CIFAR10_SGD_epoch=98_val_acc=0.9388.ckpt',
            'Adam': 'ckpt/CIFAR10/leakyVbrelu/leakyVbrelu_a=1.2_CIFAR10_Adam_epoch=96_val_acc=0.9399.ckpt',
            'AdamW': 'ckpt/CIFAR10/leakyVbrelu/leakyVbrelu_a=1.2_CIFAR10_AdamW_epoch=98_val_acc=0.9373.ckpt',
        },
        5: {
            'SGD': 'ckpt/CIFAR10/leakyVbrelu/leakyVbrelu_a=5_CIFAR10_SGD_epoch=99_val_acc=0.9424.ckpt',
            'Adam': 'ckpt/CIFAR10/leakyVbrelu/leakyVbrelu_a=5_CIFAR10_Adam_epoch=97_val_acc=0.9427.ckpt',
            'AdamW': 'ckpt/CIFAR10/leakyVbrelu/leakyVbrelu_a=5_CIFAR10_AdamW_epoch=99_val_acc=0.9446.ckpt',
        }
    },

    'PVbrelu': {
        1.2: {
            'SGD': 'ckpt/CIFAR10/PVbrelu/PVbrelu_a=1.2_CIFAR10_SGD_epoch=97_val_acc=0.9451.ckpt',
            'Adam': 'ckpt/CIFAR10/PVbrelu/PVbrelu_a=1.2_CIFAR10_Adam_epoch=96_val_acc=0.9352.ckpt',
            'AdamW': 'ckpt/CIFAR10/PVbrelu/PVbrelu_a=1.2_CIFAR10_AdamW_epoch=94_val_acc=0.9318.ckpt',
        },
        5: {
            'SGD': 'ckpt/CIFAR10/PVbrelu/PVbrelu_a=5_CIFAR10_SGD_epoch=95_val_acc=0.9432.ckpt',
            'Adam': 'ckpt/CIFAR10/PVbrelu/PVbrelu_a=5_CIFAR10_Adam_epoch=90_val_acc=0.9380.ckpt',
            'AdamW': 'ckpt/CIFAR10/PVbrelu/PVbrelu_a=5_CIFAR10_AdamW_epoch=98_val_acc=0.9405.ckpt',
        }
    },
}

# ALL
all_ckpt_path = {
    'relu': {
        'SGD': 'ckpt/CIFAR10/relu/relu_ALL_CIFAR10_SGD_epoch=97_val_acc=0.9467.ckpt',
        'Adam': 'ckpt/CIFAR10/relu/relu_ALL_CIFAR10_Adam_epoch=93_val_acc=0.9407.ckpt',
        'AdamW': 'ckpt/CIFAR10/relu/relu_ALL_CIFAR10_AdamW_epoch=96_val_acc=0.9448.ckpt',
    },
    
    'gelu': {
        'SGD': 'ckpt/CIFAR10/gelu/gelu_ALL_CIFAR10_SGD_epoch=99_val_acc=0.9453.ckpt',
        'Adam': 'ckpt/CIFAR10/gelu/gelu_ALL_CIFAR10_Adam_epoch=99_val_acc=0.9435.ckpt',
        'AdamW': 'ckpt/CIFAR10/gelu/gelu_ALL_CIFAR10_AdamW_epoch=97_val_acc=0.9419.ckpt',
    },

    'silu': {
        'SGD': 'ckpt/CIFAR10/silu/silu_ALL_CIFAR10_SGD_epoch=98_val_acc=0.9422.ckpt',
        'Adam': 'ckpt/CIFAR10/silu/silu_ALL_CIFAR10_Adam_epoch=93_val_acc=0.9436.ckpt',
        'AdamW': 'ckpt/CIFAR10/silu/silu_ALL_CIFAR10_AdamW_epoch=96_val_acc=0.9438.ckpt',
    },

    'elu': {
        'SGD': 'ckpt/CIFAR10/elu/elu_ALL_CIFAR10_SGD_epoch=95_val_acc=0.9267.ckpt',
        'Adam': 'ckpt/CIFAR10/elu/elu_ALL_CIFAR10_Adam_epoch=98_val_acc=0.9365.ckpt',
        'AdamW': 'ckpt/CIFAR10/elu/elu_ALL_CIFAR10_AdamW_epoch=99_val_acc=0.9480.ckpt',
    },

    'LReLU': {
        'SGD': 'ckpt/CIFAR10/LReLU/LReLU_ALL_CIFAR10_SGD_epoch=99_val_acc=0.9495.ckpt',
        'Adam': 'ckpt/CIFAR10/LReLU/LReLU_ALL_CIFAR10_Adam_epoch=95_val_acc=0.9409.ckpt',
        'AdamW': 'ckpt/CIFAR10/LReLU/LReLU_ALL_CIFAR10_AdamW_epoch=98_val_acc=0.9418.ckpt',
    },
    
    'PReLU': {
        'SGD': 'ckpt/CIFAR10/PReLU/PReLU_ALL_CIFAR10_SGD_epoch=99_val_acc=0.9451.ckpt',
        'Adam': 'ckpt/CIFAR10/PReLU/PReLU_ALL_CIFAR10_Adam_epoch=96_val_acc=0.9384.ckpt',
        'AdamW': 'ckpt/CIFAR10/PReLU/PReLU_ALL_CIFAR10_AdamW_epoch=96_val_acc=0.9373.ckpt',
    },
    
    'brelu': {
        'SGD': 'ckpt/CIFAR10/brelu/brelu_ALL_CIFAR10_SGD_epoch=99_val_acc=0.7989.ckpt',
        'Adam': 'ckpt/CIFAR10/brelu/brelu_ALL_CIFAR10_Adam_epoch=98_val_acc=0.9336.ckpt',
        'AdamW': 'ckpt/CIFAR10/brelu/brelu_ALL_CIFAR10_AdamW_epoch=95_val_acc=0.9163.ckpt',
    },

    'leaky_brelu': {
        'SGD': 'ckpt/CIFAR10/leaky_brelu/leaky_brelu_ALL_CIFAR10_SGD_epoch=99_val_acc=0.8073.ckpt',
        'Adam': 'ckpt/CIFAR10/leaky_brelu/leaky_brelu_ALL_CIFAR10_Adam_epoch=92_val_acc=0.9361.ckpt',
        'AdamW': 'ckpt/CIFAR10/leaky_brelu/leaky_brelu_ALL_CIFAR10_AdamW_epoch=97_val_acc=0.9139.ckpt',
    },

    'pbrelu': {
        'SGD': 'ckpt/CIFAR10/pbrelu/pbrelu_ALL_CIFAR10_SGD_epoch=97_val_acc=0.8065.ckpt',
        'Adam': 'ckpt/CIFAR10/pbrelu/pbrelu_ALL_CIFAR10_Adam_epoch=96_val_acc=0.9280.ckpt',
        'AdamW': 'ckpt/CIFAR10/pbrelu/pbrelu_ALL_CIFAR10_AdamW_epoch=98_val_acc=0.9177.ckpt',
    },

    'Vbrelu': {
        1.2: {
            'SGD': 'ckpt/CIFAR10/Vbrelu/Vbrelu_a=1.2_ALL_CIFAR10_SGD_epoch=95_val_acc=0.8115.ckpt',
            'Adam': 'ckpt/CIFAR10/Vbrelu/Vbrelu_a=1.2_ALL_CIFAR10_Adam_epoch=91_val_acc=0.9371.ckpt',
            'AdamW': 'ckpt/CIFAR10/Vbrelu/Vbrelu_a=1.2_ALL_CIFAR10_AdamW_epoch=93_val_acc=0.9260.ckpt',
        },
        5: {
            'SGD': 'ckpt/CIFAR10/Vbrelu/Vbrelu_a=5_ALL_CIFAR10_SGD_epoch=94_val_acc=0.9141.ckpt',
            'Adam': 'ckpt/CIFAR10/Vbrelu/Vbrelu_a=5_ALL_CIFAR10_Adam_epoch=98_val_acc=0.9396.ckpt',
            'AdamW': 'ckpt/CIFAR10/Vbrelu/Vbrelu_a=5_ALL_CIFAR10_AdamW_epoch=96_val_acc=0.9421.ckpt',
        }
    },

    'leakyVbrelu': {
        1.2: {
            'SGD': 'ckpt/CIFAR10/leakyVbrelu/leakyVbrelu_a=1.2_ALL_CIFAR10_SGD_epoch=98_val_acc=0.8215.ckpt',
            'Adam': 'ckpt/CIFAR10/leakyVbrelu/leakyVbrelu_a=1.2_ALL_CIFAR10_Adam_epoch=99_val_acc=0.9391.ckpt',
            'AdamW': 'ckpt/CIFAR10/leakyVbrelu/leakyVbrelu_a=1.2_ALL_CIFAR10_AdamW_epoch=96_val_acc=0.9217.ckpt',
        },
        5: {
            'SGD': 'ckpt/CIFAR10/leakyVbrelu/leakyVbrelu_a=5_ALL_CIFAR10_SGD_epoch=98_val_acc=0.9166.ckpt',
            'Adam': 'ckpt/CIFAR10/leakyVbrelu/leakyVbrelu_a=5_ALL_CIFAR10_Adam_epoch=95_val_acc=0.9398.ckpt',
            'AdamW': 'ckpt/CIFAR10/leakyVbrelu/leakyVbrelu_a=5_ALL_CIFAR10_AdamW_epoch=96_val_acc=0.9444.ckpt',
        }
    },

    'PVbrelu': {
        1.2: {
            'SGD': 'ckpt/CIFAR10/PVbrelu/PVbrelu_a=1.2_ALL_CIFAR10_SGD_epoch=99_val_acc=0.8176.ckpt',
            'Adam': 'ckpt/CIFAR10/PVbrelu/PVbrelu_a=1.2_ALL_CIFAR10_Adam_epoch=97_val_acc=0.9338.ckpt',
            'AdamW': 'ckpt/CIFAR10/PVbrelu/PVbrelu_a=1.2_ALL_CIFAR10_AdamW_epoch=98_val_acc=0.9239.ckpt',
        },
        5: {
            'SGD': 'ckpt/CIFAR10/PVbrelu/PVbrelu_a=5_ALL_CIFAR10_SGD_epoch=99_val_acc=0.9147.ckpt',
            'Adam': 'ckpt/CIFAR10/PVbrelu/PVbrelu_a=5_ALL_CIFAR10_Adam_epoch=94_val_acc=0.9343.ckpt',
            'AdamW': 'ckpt/CIFAR10/PVbrelu/PVbrelu_a=5_ALL_CIFAR10_AdamW_epoch=95_val_acc=0.9387.ckpt',
        }
    },
}

all_test(fgsm = True, pgd_step = 10, file_name = 'new_akt.csv')