import torch.multiprocessing as mp
import torch.distributed as dist
import argparse
import numpy as np
import torch
import torch.optim as optim
import wandb
import os
import random
from multiprocessing import cpu_count
from GetModel import GetModel
from training import Trainer
from utils.dataloaders import MyDataLoader
from utils.readFile import readFile

def setup(rank, world_size, opt):
    torch.cuda.set_device(opt.gpu_nums[rank])
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    dist.init_process_group("nccl", rank=rank, world_size=world_size)
    main(rank, world_size, opt)

def cleanup():
    dist.destroy_process_group()

def main(rank, world_size, opt):
    if opt.distributed:
        gpu_num = opt.gpu_nums[rank]
    else:
        gpu_num = opt.gpu_num
    
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
    
    model = opt.model
    n = opt.n
    train_dir = opt.train_dir
    num_views = opt.num_views

    # PROMENJENO --------------------------------------------------
    if dataset == 'CBIS'or dataset == 'INbreast':
        num_classes = 1
        birads_mode= False
    elif dataset == 'INbreastBIRADS': 
        num_classes = 6
        birads_mode= True
    elif dataset == 'CBIS_patches':
        num_classes = 5
        birads_mode= False
    else:
        RuntimeError('Wrong dataset or not implemented')
    
    if birads_mode:
      print("BIRADS clasiffication", num_classes)
    

    train_loader, eval_loader = MyDataLoader(root=train_dir, name=dataset, batch_size=batch_size, num_workers=n_workers, 
                                             distributed=opt.distributed, rank=rank, world_size=world_size)
    
    pretrained_weights = None if opt.evaluate_model else opt.model_state
    net = GetModel(str_model=model, n=n, num_classes=num_classes, weights=pretrained_weights, 
                    shared=opt.shared, patch_weights=opt.patch_weights)  
    
    if opt.evaluate_model:
        if dataset != 'CBIS_patches' and num_views == 2:
            net.add_top_blocks(num_classes=num_classes)
        net.load_state_dict(torch.load(opt.model_state, map_location='cpu'))
        
    # else:
    #     # Load pretrained weights.
    #     # In case of four-view models, loading weights is done inside the architecture itself.
    #     if opt.num_views == 2 and opt.model_state:    
    #         if opt.patch_weights:
    #             print("Loading weights of patch classifier from ", opt.model_state)
    #             net = GetModel(str_model=model, n=n, num_classes=5)  
    #             net.load_state_dict(torch.load(opt.model_state, map_location='cpu'))
    #             net.add_top_blocks(num_classes=num_classes)

    #         else: # whole-image weights
    #             if birads_mode:
    #               print("Loading weights of binary pretrained classifier, removing last layers", opt.model_state)
    #               # creating original binary model 
    #               net = GetModel(str_model=model, n=n, num_classes=1,weights=None)  
    #               net.add_top_blocks(num_classes=1)
    #               # load binary whole image weights
    #               net.load_state_dict(torch.load(opt.model_state, map_location='cpu'), strict=False)
    #               # replace 
    #               print("Replace the last layer ")
    #               net.linear=torch.nn.Linear(1024, num_classes)
    #               # net.add_top_blocks(num_classes=num_classes)
    #             else:
    #               net = GetModel(str_model=model, n=n, num_classes=1) 
    #               net.add_top_blocks(num_classes=num_classes)
    #               print("Loading weights of pretrained whole-image classifier from ", opt.model_state)
    #               print(f'Number of classes:',num_classes)
    #               net.load_state_dict(torch.load(opt.model_state, map_location='cpu'))
        
    else:
    # TRAINING MODE
      if dataset == 'INbreastBIRADS' and opt.model_state:
          # =====================================================
          # BIRADS TRANSFER LEARNING - KORISTI BINARNE TEŽINE
          # =====================================================
          print("\n" + "="*60)
          print("BIRADS TRANSFER LEARNING MODE")
          print("="*60)
          
          # Korak 1: Učitaj model sa 1 klasom (binarna klasifikacija)
          print("Step 1: Loading base model (binary classification)...")
          net = GetModel(
              str_model=model, 
              n=n, 
              num_classes=1,  # Binarni model
              weights=None,
              shared=opt.shared, 
              patch_weights=False
          )
          
          # Korak 2: Dodaj PHRefiner blokove
          print("Step 2: Adding PHRefiner blocks (layer5, layer6)...")
          net.add_top_blocks(num_classes=1)
          
          # Korak 3: Učitaj binarne težine
          print(f"Step 3: Loading binary weights from: {opt.model_state}")
          try:
              pretrained_dict = torch.load(opt.model_state, map_location='cpu')
              net.load_state_dict(pretrained_dict)
              print("✓ Binary weights loaded successfully!")
          except Exception as e:
              print(f"✗ Error loading weights: {e}")
              raise
          
          # Korak 4: Zameni poslednji FC layer za BIRADS
          print(f"Step 4: Replacing FC layer: 1 class → {num_classes} classes")
          net.linear = torch.nn.Linear(1024, num_classes)
          print("✓ FC layer replaced!")
          
          
      elif opt.model_state and opt.num_views == 2:
          # Stari kod za patch ili whole-image weights (NE za BIRADS)
          if opt.patch_weights:
              print("Loading patch classifier weights...")
              net = GetModel(str_model=model, n=n, num_classes=5)  
              net.load_state_dict(torch.load(opt.model_state, map_location='cpu'))
              net.add_top_blocks(num_classes=num_classes)
          else:
              print("Loading whole-image weights...")
              net = GetModel(str_model=model, n=n, num_classes=num_classes)
              net.add_top_blocks(num_classes=num_classes)
              net.load_state_dict(torch.load(opt.model_state, map_location='cpu'))
      else:
          # Training od nule
          print("Training from scratch...")
          net = GetModel(
              str_model=model, 
              n=n, 
              num_classes=num_classes, 
              weights=None,
              shared=opt.shared, 
              patch_weights=opt.patch_weights
          )
          if num_views == 2 and dataset != 'CBIS_patches':
              net.add_top_blocks(num_classes=num_classes)
  # ----
    if rank == 0:
        wandb.init(project="phbreast-project")
        wandb.config.update(opt, allow_val_change=True)
        wandb.watch(net)
    
    params = sum(p.numel() for p in net.parameters() if p.requires_grad)
    print(f'[Proc{rank}]Number of parameters:', params)
    print()

    checkpoint_folder = '/content/drive/MyDrive/PHBreastBIRADS/checkpoints2/'
    if not os.path.isdir(checkpoint_folder):
        os.makedirs(checkpoint_folder)
    
    # Initialize optimizers
    if opt.optim == "SGD":
        optimizer = optim.SGD(net.parameters(), lr=lr, momentum=opt.momentum, weight_decay=opt.weight_decay)
    if opt.optim == "Adam":
        optimizer = optim.Adam(net.parameters(), lr=lr, weight_decay=opt.weight_decay, betas=(0.5, 0.999))
    

    '''Train model'''
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
    
    parser.add_argument('--Dataset', type=str, default='SVHN', help='CBIS_patches, CBIS, INbreast, INbreastBIRADS')
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

