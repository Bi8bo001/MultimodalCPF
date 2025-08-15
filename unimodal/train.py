from argparse import ArgumentParser
import os
import sys
import torch
from torch_geometric.loader import DataLoader
import pytorch_lightning as pl
from pytorch_lightning import Trainer, loggers
from pytorch_lightning.callbacks import ModelCheckpoint
import shutil

os.chdir(os.path.dirname(os.path.abspath(__file__)))

from utils import Params, seed_worker  # params & reproducible results by seeds
from models.regression import RegressionModel  # unimodal: whole framework 到时候注释掉
from dataloaders.common import filter_by_atom_num
from functools import partial  # frozen params
from distutils.util import strtobool

### multimodal 这个动态import在底下
# from multimodal_fusion.regression_fusion import RegressionModelFusion as RegressionModel

def get_option():
    argparser = ArgumentParser(description='Training the network')
    argparser.add_argument('-p', '--param_file', type=str, default='default.json', help='filename of the parameter JSON')
    args, unknown = argparser.parse_known_args()
    return args

def train():
    args = get_option()
    print('parsed args :')
    print(args)
    try:
        params = Params(f'{args.param_file}')
    except:
        params = Params(f'./params/{args.param_file}')

    parser = ArgumentParser(description='Training the network')
    parser.add_argument('-p', '--param_file', type=str, default=args.param_file, help='Config json file for default params')
    # load the json config and use it as default values.
    boolder = lambda x:bool(strtobool(x))
    typefinder = lambda v: str if v is None else boolder if type(v)==bool else type(v)  # def type of param
    for key in params.dict:
        v = params.dict[key]
        if isinstance(v, (list, tuple)):  # check if v is list/tuple
            parser.add_argument(f"--{key}", type=typefinder(v[0]), default=v, nargs='+')
        else:
            parser.add_argument(f"--{key}", type=typefinder(v), default=v)
    params.__dict__ = parser.parse_args().__dict__  # convert to a uniform form, allows e.g. params.encoder_name
    print(params.dict)

    import models.global_config as config
    config.REPRODUCIBLITY_STATE = getattr(params, 'reproduciblity_state', 0)
    print(f"reproduciblity_state = {config.REPRODUCIBLITY_STATE}")

    # Reproducibility
    seed = getattr(params, 'seed', 123)  # for reproducible
    deterministic = params.encoder_name in ["latticeformer"]  # deterministic kernel
    pl.seed_everything(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False  # cudnn: CUDA Deep Neural Network library, False: prevent auto optimization -> different result
    torch.backends.cudnn.deterministic = deterministic
    torch.backends.cuda.matmul.allow_tf32 = False  # tf32: same as cudnn
    torch.backends.cudnn.allow_tf32 = False
    # torch.backends.cuda.preferred_linalg_library("cusolver") # since torch 1.11, needed to avoid an error by torch.det(), but now det_3x3 is implemented manually.

    # import dataset based on encoder_name
    if params.encoder_name == "latticeformer":
        from dataloaders.dataset_latticeformer import RegressionDatasetMP_Latticeformer as Dataset  # 接入数据处理板块处理好的数据集
        
        ### multimodal
        # from dataloaders.dataset_latticeformer import FusionDatasetMP_Latticeformer as Dataset
        
    else:
        raise NameError(params.encoder_name)
    #* from dataloaders.dataloader import PyMgStructureMP as Dataset

    # Setup datasets
    ddp = getattr(params, "ddp", False)
    max_val = getattr(params, "train_filter_max", 0)  # during training, only keep sample with [min,max] atoms in a unit cell
    min_val = getattr(params, "train_filter_min", 0)
    num_workers = getattr(params, "num_workers", 4)  # num of DataLoader multithread
    num_workers = num_workers if num_workers >= 0 else os.cpu_count()
    target_set = getattr(params, "target_set", None)
    train_filter = partial(filter_by_atom_num, max_val=max_val, min_val=min_val) \
        if max_val > 0 or min_val > 1 else None  # not set filter then None
    if not hasattr(params, "training_data") or params.training_data ==  "default":  # by default use the whole dataset
        train_dataset = Dataset(target_split='train', target_set=target_set, post_filter=train_filter)
    elif params.training_data in ["train_6400", "train_10k"]:
        train_dataset = Dataset(target_split=params.training_data, post_filter=train_filter)
    else:
        raise NameError(params.training_data)

    val_dataset = Dataset(target_split='val', target_set=target_set)  # valid set and test set no filter
    test_dataset = Dataset(target_split='test', target_set=target_set)

    if torch.cuda.device_count() == 1 or not ddp:
        train_loader = DataLoader(train_dataset, batch_size=params.batch_size, shuffle=True, num_workers=num_workers, drop_last=True, 
                                worker_init_fn=seed_worker, pin_memory=True)
        ### debug
        print(f"train_dataset length: {len(train_dataset)}")
        print("Checking training dataset...")
        try:
            first_batch = next(iter(train_loader))
            print(f"   x shape: {first_batch.x.shape}")
            print(f"   pos shape: {first_batch.pos.shape}")
            print(f"   y: {first_batch.y if hasattr(first_batch, 'y') else 'None'}")
            assert first_batch.x.shape[0] > 0, "❌ Empty training graph! Check if filter or data is broken."

            ## check text inserted or not?
            # print(f"   text: {first_batch.text if hasattr(first_batch, 'text') else 'No text'}")

        except Exception as e:
            print(f"Failed to load training data: {e}")
        ### debug over

        val_loader  = DataLoader(val_dataset, batch_size=params.batch_size, shuffle=False, num_workers=num_workers, drop_last=False, pin_memory=True)
        test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=num_workers, drop_last=False)
    else:  # if ddp, should specify the shuffler
        train_sampler = torch.utils.data.distributed.DistributedSampler(
            train_dataset, rank=0, num_replicas=torch.cuda.device_count(), shuffle=True)
        val_sampler = torch.utils.data.distributed.DistributedSampler(
            val_dataset, rank=0, num_replicas=torch.cuda.device_count(), shuffle=False)
        test_sampler = torch.utils.data.distributed.DistributedSampler(
            test_dataset, rank=0, num_replicas=torch.cuda.device_count(), shuffle=False)
        
        train_loader = DataLoader(train_dataset, batch_size=params.batch_size, shuffle=False, num_workers=num_workers, drop_last=True, 
                        worker_init_fn=seed_worker, sampler=train_sampler, pin_memory=True)
        val_loader  = DataLoader(val_dataset, batch_size=params.batch_size, shuffle=False, num_workers=num_workers, drop_last=False, sampler=val_sampler, pin_memory=True)
        test_loader = DataLoader(test_dataset, batch_size=params.batch_size, shuffle=False, num_workers=num_workers, drop_last=False, sampler=test_sampler)

    # Uncomment below to force the updating of the cache files.
    # train_dataset.process()
    # val_dataset.process()
    # test_dataset.process()

    # Setup model and trainer
    ## logger (tensorBoard) 记录日志
    logger = loggers.TensorBoardLogger(params.save_path, name=params.experiment_name, default_hp_metric=False)
    logger.log_hyperparams(params.__dict__, \
        {"hp/val": 1.0, "hp/avr50":1.0, "hp/min_avr50":1.0, "hp/min":1.0, "hp/mean_min10": 1.0}
    )

    ### log + fusion type, text encoder, modality mask?
    # logger.log_hyperparams({**params.__dict__})

    ckpt_dir=logger.log_dir+'/model_checkpoint'
    checkpoint_callback = ModelCheckpoint(save_top_k=params.model_checkpoint_save_top_k,
                                        monitor='val/loss', mode='min', dirpath=ckpt_dir)  # save ckpt with min valid loss

    ## model initialization
    system = RegressionModel(params, train_loader, val_loader)

    ### fusion则import新的regression framework, 也设置成一样的名字让改动最小, 前面根据fusion_type选择import哪一个
    '''
    if params.fusion_type == "none":
        from models.regression import RegressionModel
    else:
        from multimodal_fusion.regression_fusion import RegressionModel
    system = RegressionModel(params, train_loader, val_loader)
    '''
    
    param_num = sum([p.nelement() for p in system.model.parameters()])  # all params
    print(f"Whole: {param_num}, {param_num*4/1024**2} MB")
    param_num = sum([p.nelement() for p in system.model.encoder.layers[0].parameters()])
    print(f"Block: {param_num}, {param_num*4/1024**1} KB")
    
    # initialize mean and std values in crystalformer by forwarding once. 
    ## initialize encoder
    if ddp:
        with torch.no_grad():
            import random
            import numpy
            state = torch.random.get_rng_state(), random.getstate(), numpy.random.get_state()  # save randomness
            system.train()  # entering training mode
            system.cuda().forward(next(iter(train_loader)).cuda())  # forward once for encoder initialization
            system.cpu()
            torch.random.set_rng_state(state[0])    #* usually, resetting torch's state is sufficient
            random.setstate(state[1])  # reset random state as initialized
            numpy.random.set_state(state[2])

    if params.pretrained_model is not None:  # loading pretrained model if provided
        system = RegressionModel.load_from_checkpoint(  
            params.pretrained_model,
            params=params,
            train_loader=train_loader,
            val_loader=val_loader,
            strict=False)
        
    # Train model
    trainer = Trainer(
        logger=logger, 
        devices=torch.cuda.device_count() if ddp else 1,
        strategy='ddp' if ddp else 'auto',
        max_epochs=params.n_epochs,
        default_root_dir=params.save_path,
        enable_checkpointing=True,
        callbacks=[checkpoint_callback],  # model save, early stop, ...
        num_nodes=1,
        limit_train_batches=params.train_percent_check,  # whether onlt partial training
        limit_val_batches=params.val_percent_check,
        fast_dev_run=False,
        deterministic=deterministic)  # whether fixed randomness
    
    ## trainning + time recording(crystalformer主打的耗时少)
    import time
    time_dict = {}
    start_time = time.time()
    trainer.fit(system)  # run
    time_dict['time-train'] = (time.time()-start_time)
    scores = []
    
    ## save ckpt
    # save the last cjpt
    trainer.save_checkpoint(ckpt_dir+'/last.ckpt')  # ensure checkpointing after SWA's BN updating
    #* Validate and test the SWA model if available
    # support stochastic weight averaging(就是paper中的两种model)
    if system.enable_average_model('val-swa'):
        start_time = time.time()
        scores += trainer.validate(model=system, dataloaders=val_loader)
        time_dict['time-val-swa'] = (time.time()-start_time)
    if system.enable_average_model('test-swa'):
        start_time = time.time()
        scores += trainer.test(model=system, dataloaders=test_loader)
        time_dict['time-test-swa'] = (time.time()-start_time)
    system.disable_average_model()

    # Prepare the best model for testing
    ## with minimize valid loss
    if os.path.exists(checkpoint_callback.best_model_path):
        best_model = RegressionModel.load_from_checkpoint(  # load
            checkpoint_callback.best_model_path,
            params=params,
            train_loader=train_loader,
            val_loader=val_loader)
        system.model = best_model.model  # replace the last one
        system.disable_average_model()  # close SWA
        del best_model  # delete the middle product
        trainer.save_checkpoint(ckpt_dir+'/best.ckpt')  # 标准路径再保存一份

    ## validation set -> loss
    start_time = time.time()
    scores += trainer.validate(model=system, dataloaders=val_loader)
    time_dict['time-val'] = (time.time()-start_time)

    ## test set
    start_time = time.time()
    scores += trainer.test(model=system, dataloaders=test_loader)
    time_dict['time-test'] = (time.time()-start_time)

    ## consuming time + record into log
    print("Running times-----------------------------------------")
    print(f"time-train   : {time_dict['time-train']/(60**2)} h")
    print(f"time-val-swa : {time_dict['time-val-swa']} s")
    print(f"time-test-swa: {time_dict['time-test-swa']/len(test_dataset)*1000} ms")
    print(f"time-val     : {time_dict['time-val']} s")
    print(f"time-test    : {time_dict['time-test']/len(test_dataset)*1000} ms")
    with open(f'{logger.log_dir}/time.txt', 'w') as f:
        print(f"time-train   : {time_dict['time-train']/(60**2)} h", file=f)
        print(f"time-val-swa : {time_dict['time-val-swa']} s", file=f)
        print(f"time-test-swa: {time_dict['time-test-swa']/len(test_dataset)*1000} ms", file=f)
        print(f"time-val     : {time_dict['time-val']} s", file=f)
        print(f"time-test    : {time_dict['time-test']/len(test_dataset)*1000} ms", file=f)
        for score in scores:
            for key in score:
                print(f"{key}\t:{score[key]}", file=f)        

    logger.finalize('success')  # to properly output all test scores in a TB log.

if __name__ == '__main__':
    train()
