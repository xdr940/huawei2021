
import networks
from path import Path
import torch.optim as optim
import torch
from datasets import CustomSequence
from torch.utils.data import DataLoader
from utils.official import readlines

def dataset_init(opts):

    #local
    dataset_opt = opts['dataset']
    datasets_dict = {
        "ctmnt": CustomSequence
                     }

    assert dataset_opt['type'] in datasets_dict.keys()
    dataset = datasets_dict[dataset_opt['type']]  # 选择建立哪个类，这里kitti，返回构造函数句柄


    split_path = Path(dataset_opt['split']['path'])
    train_path = split_path / dataset_opt['split']['train_file']
    val_path = split_path / dataset_opt['split']['val_file']
    data_path = Path(dataset_opt['path'])


    X_names = dataset_opt['X_names']
    Y_names = dataset_opt['Y_names']

    #global
    seq_length= opts['seq_length']
    batch_size = opts['batch_size']
    num_workers = opts['num_workers']





    train_seq_idxs = readlines(train_path)
    val_seq_idxs = readlines(val_path)

    num_train_samples = len(train_seq_idxs)
    stat_dict={}
    # train loader
    train_dataset = dataset(
        data_path = data_path,
        seq_idxs=train_seq_idxs,
        X_names = X_names,
        Y_names = Y_names,
        stat_dict=stat_dict,
        mode="train",
        seq_length=seq_length
    )
    train_loader = DataLoader(
        dataset=train_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True
    )
    # val loader
    val_dataset = dataset(
        data_path=data_path,
        seq_idxs=val_seq_idxs,
        X_names=X_names,
        Y_names=Y_names,
        stat_dict=stat_dict,
        mode="val",
        seq_length=seq_length

    )

    val_loader = DataLoader(
        dataset=val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True)

    print("Using split:{}, {}, {}".format(split_path,
                                                                dataset_opt['split']['train_file'],
                                                                dataset_opt['split']['val_file']
                                                                ))
    print("There are {:d} training items and {:d} validation items".format(
        len(train_dataset), len(val_dataset)))

    for k,v in stat_dict.items():
        stat_dict[k] = torch.tensor(v,dtype=torch.float32).to(opts['device'])

    return train_loader, val_loader, stat_dict


def model_init(opt):


    #global opts
    seq_length = opt['seq_length']
    device = opt['device']

    # local opts
    model_opt = opt['model']
    lr = model_opt['lr']
    scheduler_step_size = model_opt['scheduler_step_size']
    load_paths = model_opt['load_paths']
    optimizer_path = model_opt['optimizer_path']

    input_fnum = model_opt['input_fnum']
    output_fnum = model_opt['output_fnum']

    hidden_size = model_opt['hidden_size']
    num_layers = model_opt['num_layers']



    models = {}  # dict

    # models['lstm'] = networks.getLSTM(input_fnum=input_fnum,hidden_size=hidden_size,seq_length=seq_length)

    # models['decoder'] = networks.getDecoder(hidden_size=hidden_size,output_fnum=output_fnum,seq_length=seq_length)
    if model_opt['framework']=='rnn':
        models['rnn'] = networks.getRNN(input_fnum=input_fnum,hidden_size=hidden_size,output_fnum=output_fnum,num_layers=num_layers)
    elif model_opt['framework']=='lstm':
        models['lstm'] = networks.getLSTM(input_fnum=input_fnum,hidden_size=hidden_size,output_fnum=output_fnum,num_layers=num_layers)


    # models['en-de'] = networks.getEnDe(input_fnum=input_fnum,output_fnum=output_fnum,hidden_size=40)

    # model device
    for k, v in models.items():
        models[k].to(device)

    # params to train
    parameters_to_train = []
    for k, v in models.items():
        parameters_to_train += list(v.parameters())

    model_optimizer = optim.Adam(parameters_to_train, lr)

    model_lr_scheduler = optim.lr_scheduler.StepLR(
        model_optimizer,
        scheduler_step_size,
        lr
    )  # end models arch




    print('--> load models:')

    # load models
    for name,model in models.items():
        if name in list(load_paths.keys()):
            path = load_paths[name]
            if not path:
                continue
            model_dict = models[name].state_dict()
            pretrained_dict = torch.load(path)
            pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
            model_dict.update(pretrained_dict)
            models[name].load_state_dict(model_dict)
            print(" ->{}:{}".format(name, path))



    if optimizer_path:
        optimizer_path = Path(optimizer_path)
        optimizer_dict = torch.load(optimizer_path)
        model_optimizer.load_state_dict(optimizer_dict)
        print('optimizer params from {}'.format(optimizer_path))

    else:
        print('optimizer params from scratch')

    return models, model_optimizer, model_lr_scheduler