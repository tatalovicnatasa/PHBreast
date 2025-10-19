
import torch.multiprocessing as mp
import torch.distributed as dist
import argparse
import numpy as np
import torch
import torch.optim as optim
import wandb # tracking experimetns online
import os
import random
from multiprocessing import cpu_count
from GetModel import GetModel # function that implements the model from GetModel.py 
from training import Trainer  # class for training or evaluation from Trainer.py
from utils.dataloaders import MyDataLoader # for datas from dataloaders.py
from utils.readFile import readFile #from readFile.py

def setup(rank, world_size, opt):
    torch.cuda.set_device(opt.gpu_nums[rank])
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    dist.init_process_group("nccl", rank=rank, world_size=world_size)
    main(rank, world_size, opt)

def cleanup():
    dist.destroy_process_group()

def main(rank, world_size, opt): 
    # Setting the GPU 
    if opt.distributed:
        gpu_num = opt.gpu_nums[rank] # more GPUs
    else:
        gpu_num = opt.gpu_num # one GPU
    
    use_cuda = opt.cuda
   
    lr = opt.lr
    epochs = opt.epochs
    batch_size = opt.batch_size
    dataset = opt.Dataset
    n_workers = opt.n_workers
    l1_reg = opt.l1_reg
    
    if n_workers=='max':
        n_workers = cpu_count()
    
    # Set seed    
    manualSeed = opt.seed
    random.seed(manualSeed)
    torch.manual_seed(manualSeed)
    seed=manualSeed
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    
    model = opt.model # ex. phcresnet18 
    n = opt.n         # hipercomplex parameter
    train_dir = opt.train_dir
    num_views = opt.num_views
    
    if dataset == 'CBIS' or dataset == 'INbreast': # based on dataset provided
        num_classes = 1 # binary classification
    elif dataset == 'CBIS_patches':
        num_classes = 5 # this is pretraining and it needs 5 classed
    else:
        RuntimeError('Wrong dataset or not implemented')
    
    train_loader, eval_loader = MyDataLoader(root=train_dir, name=dataset, batch_size=batch_size, num_workers=n_workers, 
                                             distributed=opt.distributed, rank=rank, world_size=world_size)
    
    pretrained_weights = None if opt.evaluate_model else opt.model_state # ako evaluira model, ne ucitava pretrenirane ali hoce posle 
    net = GetModel(str_model=model, n=n, num_classes=num_classes, weights=pretrained_weights,   # prosledjuju se tezine
                    shared=opt.shared, patch_weights=opt.patch_weights)  
    
    if opt.evaluate_model: # EVALUACIJA
        if dataset != 'CBIS_patches' and num_views == 2: # nije pretreniranje u pitanju
            net.add_top_blocks(num_classes=num_classes) # dodavanje top blokova    
        net.load_state_dict(torch.load(opt.model_state, map_location='cpu')) # ucitavanje tezina ipak 
        
    else:
        # Load pretrained weights.
        # In case of four-view models, loading weights is done inside the architecture itself.
        if opt.num_views == 2 and opt.model_state:    #TRENING 
            if opt.patch_weights: # ovo je patc classifier
                print("Loading weights of patch classifier from ", opt.model_state)
                net = GetModel(str_model=model, n=n, num_classes=5)  
                net.load_state_dict(torch.load(opt.model_state, map_location='cpu'))
                net.add_top_blocks(num_classes=num_classes)

            else: # whole-image weights
                net.add_top_blocks(num_classes=num_classes)
                print("Loading weights of pretrained whole-image classifier from ", opt.model_state)
                net.load_state_dict(torch.load(opt.model_state, map_location='cpu'))
        
    # Tracking online 
    if rank == 0:
        wandb.init(project="phbreast-project")
        wandb.config.update(opt, allow_val_change=True)
        wandb.watch(net)
    # Broji parametre
    params = sum(p.numel() for p in net.parameters() if p.requires_grad)
    print(f'[Proc{rank}]Number of parameters:', params)
    print()
        
    checkpoint_folder = 'checkpoints/'
    if not os.path.isdir(checkpoint_folder):
        os.makedirs(checkpoint_folder)
    
    # Initialize optimizers
    if opt.optim == "SGD": 
        optimizer = optim.SGD(net.parameters(), lr=lr, momentum=opt.momentum, weight_decay=opt.weight_decay)
    if opt.optim == "Adam":
        optimizer = optim.Adam(net.parameters(), lr=lr, weight_decay=opt.weight_decay, betas=(0.5, 0.999))
    

    '''Train model'''
    # Kreiranje Trainer objekta
    trainer = Trainer(net, optimizer, epochs=epochs,
                      use_cuda=use_cuda, gpu_num=gpu_num,
                      checkpoint_folder = checkpoint_folder,
                      l1_reg=l1_reg,
                      num_classes=num_classes,
                      num_views=num_views,
                      pos_weight=opt.pos_weight,
                      distributed=opt.distributed,
                      rank=rank,
                      world_size=world_size)
    # Trening ili evaluacija 
    if opt.evaluate_model: 
        trainer.test(eval_loader)
    else:
        trainer.train(train_loader, eval_loader)
    
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--cuda', type=bool, default=True)
    parser.add_argument('--seed', type=int, default=1656079)
    parser.add_argument('--gpu_num', type=int, default=0)
    parser.add_argument('--n_workers', default=1)
    parser.add_argument('--n', type=int, default=2, help="n parameter for PHC layers")
    parser.add_argument('--optim', type=str, default="Adam")
    parser.add_argument('--l1_reg', type=bool, default=False)
    parser.add_argument('--train_dir', type=str, default='./data/', help="Folder containg training data")
    
    parser.add_argument('--Dataset', type=str, default='SVHN', help='CBIS_patches, CBIS, INbreast')
    parser.add_argument('--num_views', type=int, default=2, help='Number of views in input')
    parser.add_argument('--model', type=str, default='resnet20', help='Models: ...')
    parser.add_argument('--batch_size', type=int, default=4)
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--lr', type=float, default=0.00001)
    parser.add_argument('--weight_decay', type=float, default=5e-4)
    parser.add_argument('--momentum', type=float, default=0.9)
    parser.add_argument('--model_state', help='model weights for pretraining or testing')
    parser.add_argument('--pos_weight', type=float, help='pos_weight for BCE in case of unbalanced data')
    parser.add_argument('--shared', type=bool, default=True, help='in case of fourview model: True for shared bottleneck, False for concat version')
    parser.add_argument('--patch_weights', type=bool, default=True, help='True if weights are patch, False if they are whole-image')
    parser.add_argument('--evaluate_model', type=bool, default=False)
    parser.add_argument('--distributed', type=bool, default=False, help='True for distributed training with DistributedDataParallel')
    parser.add_argument('--gpu_nums', help='indices of gpus to use for distributed training')
    
    parser.add_argument('--TextArgs', type=str, default='TrainingArguments.txt', help='Path to text with training settings')

    parse_list=readFile(parser.parse_args().TextArgs)
    
    opt = parser.parse_args(parse_list)
    
    if opt.distributed:
        opt.gpu_nums = list(map(int, opt.gpu_nums.split()))
        # we have 2 gpus
        world_size = 2  
        print(f"DISTRIBUTED TRAINING: spawning {world_size} processes")
        mp.spawn(setup, args=(world_size, opt), nprocs=world_size)
    else:
        main(rank=0, world_size=None, opt=opt)
