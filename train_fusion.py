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
#from models.regression import RegressionModel  # unimodal: whole framework
from dataloaders.common import filter_by_atom_num
from functools import partial  # frozen params
from distutils.util import strtobool

### multimodal dynamic import
from multimodal_fusion.regression_fusion import RegressionModelFusion as RegressionModel
from multimodal_fusion.encoder_selector import get_dataset_and_encoder
# from multimodal_fusion.dataset_fusion import collate_fn_fusion_tokenlevel

def get_option():
    argparser = ArgumentParser(description='Training the network')
    argparser.add_argument('-p', '--param_file', type=str, default='default.json', help='filename of the parameter JSON')
    args, unknown = argparser.parse_known_args()
    return args

def count_params(module):
    return sum(p.numel() for p in module.parameters() if p.requires_grad)

def num_params(system):
    # model components
    if hasattr(system, "structure_encoder"):
        struct_params = count_params(system.structure_encoder)
        print(f"[Structure Encoder] Params: {struct_params:,} ({struct_params * 4 / 1024**2:.2f} MB)")
    else:
        struct_params = count_params(system.model)
        print(f"[Unimodal Model] Params: {struct_params:,} ({struct_params * 4 / 1024**2:.2f} MB)")
    if hasattr(system, "text_encoder"):
        text_params = count_params(system.text_encoder)
        print(f"[Text Encoder] Params: {text_params:,} ({text_params * 4 / 1024**2:.2f} MB)")
        # Frozen check
        if all(not p.requires_grad for p in system.text_encoder.parameters()):
            print("Text encoder is frozen.")
    else:
        text_params = 0
    if hasattr(system, "fusion_module"):
        fusion_params = count_params(system.fusion_module)
        print(f"[Fusion Module] Params: {fusion_params:,} ({fusion_params * 4 / 1024**2:.2f} MB)")
    else:
        fusion_params = 0
    total_params = struct_params + text_params + fusion_params
    return total_params


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

    ### multimodal: add assertion

    ## fusion_type & text_encoder
    assert params.fusion_type in ['none', 'sum', 'concat', 'gated', 'cross_attn'], \
        f"Invalid fusion_type: {params.fusion_type}"
    if params.fusion_type != "none":
        assert params.text_encoder_name in ['matscibert', 'bert'], \
            f"Unsupported text_encoder_name: {params.text_encoder_name}"

    ## struct_mask_prob/text_mask_prob
    assert 0.0 <= params.struct_mask_prob <= 1.0, \
        f"struct_mask_prob should be in [0,1], got {params.struct_mask_prob}"
    assert 0.0 <= params.text_mask_prob <= 1.0, \
        f"text_mask_prob should be in [0,1], got {params.text_mask_prob}"

    # loss weights
    if hasattr(params, "fusion_loss_weights"):
        assert isinstance(params.fusion_loss_weights, list) and len(params.fusion_loss_weights) == 3, \
            "fusion_loss_weights must be a list of 3 floats: [fusion, struct, text]"

    # use_amp
    assert isinstance(params.use_amp, bool), f"use_amp must be boolean, got {type(params.use_amp)}"

    # path
    if params.pretrained_model is not None:
        assert os.path.exists(params.pretrained_model), \
            f"Pretrained model not found at {params.pretrained_model}"
    if getattr(params, "struct_encoder_ckpt", None) is not None:
        assert os.path.exists(params.struct_encoder_ckpt), \
            f"Structure encoder ckpt not found at {params.struct_encoder_ckpt}"
    if not params.freeze_text_encoder and getattr(params, "text_encoder_ckpt", None) is not None:
        assert os.path.exists(params.text_encoder_ckpt), \
            f"Text encoder ckpt not found at {params.text_encoder_ckpt}"

    print(params.dict)


    import models.global_config as config
    config.REPRODUCIBLITY_STATE = getattr(params, 'reproduciblity_state', 0)
    print(f"reproduciblity_state = {config.REPRODUCIBLITY_STATE}")

    # Reproducibility
    seed = getattr(params, 'seed', 123)  # for reproducible
    # deterministic = params.encoder_name in ["latticeformer"]  # deterministic kernel
    
    ###
    deterministic = False  # Force disable deterministic mode to allow torch.bincount
    
    pl.seed_everything(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False  # cudnn: CUDA Deep Neural Network library, False: prevent auto optimization -> different result
    torch.backends.cudnn.deterministic = deterministic
    torch.backends.cuda.matmul.allow_tf32 = False  # tf32: same as cudnn
    torch.backends.cudnn.allow_tf32 = False
    #* torch.backends.cuda.preferred_linalg_library("cusolver") # since torch 1.11, needed to avoid an error by torch.det(), but now det_3x3 is implemented manually.

    # allow non-deterministic ops like bincount
    # torch.use_deterministic_algorithms(False)

    ### multimodal: 
    ## import dataset based on encoder_name
    Dataset, StructureEncoder = get_dataset_and_encoder(params)
    # print(f"Dataset: {type(Dataset)}")
    # # print(Dataset)
    # # print("=====================111")
    # print(f"StructureEncoder: {type(StructureEncoder)}")
    # print(StructureEncoder)
    # # print("=====================222")
    # # input()
    assert isinstance(Dataset, type), f"Dataset should be a class, got {type(Dataset)}"
    assert isinstance(StructureEncoder, type), "StructureEncoder must be a class, not instance."
    print(f"Using Dataset: {Dataset.__name__}, Structure Encoder: {StructureEncoder.__name__}")  # for debugging
    # print("======================")
    # input()

    # Setup datasets
    ddp = getattr(params, "ddp", False)
    max_val = getattr(params, "train_filter_max", 0)  # during training, only keep sample with [min,max] atoms in a unit cell
    min_val = getattr(params, "train_filter_min", 0)
    num_workers = getattr(params, "num_workers", 4)  # num of DataLoader multithread
    num_workers = num_workers if num_workers >= 0 else os.cpu_count()
    #target_set = getattr(params, "target_set", None)
    train_filter = partial(filter_by_atom_num, max_val=max_val, min_val=min_val) \
        if max_val > 0 or min_val > 1 else None  # not set filter then None
    
    ### multimodal param
    if not hasattr(params, "training_data") or params.training_data ==  "default":  # by default use the whole dataset
        train_dataset = Dataset(
            target_split='train',
            #target_set=target_set,
            #post_filter=train_filter,

            ### add
            structure_encoder=StructureEncoder,
            struct_mask_prob=params.struct_mask_prob,
            text_mask_prob=params.text_mask_prob,
            use_text_mask=params.text_mask_prob > 0,
            use_struct_mask=params.struct_mask_prob > 0,
            freeze_text_encoder=params.freeze_text_encoder,
            fusion_type=params.fusion_type,
            modal_dropout_prob=params.modal_dropout_prob,  ### mask training strategy

            # ...
            )
        print(f"[Debug][Train] Dataset size: {len(train_dataset)} | First sample keys: {list(train_dataset[0].keys())}")
        print(f"[Debug][Train] Dataset text embedding shape: {train_dataset[0].text_emb.shape}")

    elif params.training_data in ["train_6400", "train_10k"]:
        train_dataset = Dataset(
            target_split=params.training_data, 
            #post_filter=train_filter,
            structure_encoder=StructureEncoder  ##
            )
    else:
        raise NameError(f"Unknown training_data setting: {params.training_data}")

    val_dataset = Dataset(
        target_split='val',
        structure_encoder=StructureEncoder,
        struct_mask_prob=params.struct_mask_prob,
        text_mask_prob=params.text_mask_prob,
        use_text_mask=params.text_mask_prob > 0,
        use_struct_mask=params.struct_mask_prob > 0,
        target_field=params.targets,
        freeze_text_encoder=params.freeze_text_encoder,
        fusion_type=params.fusion_type,
    )

    test_dataset = Dataset(
        target_split='test',
        structure_encoder=StructureEncoder,
        struct_mask_prob=params.struct_mask_prob,
        text_mask_prob=params.text_mask_prob,
        use_text_mask=params.text_mask_prob > 0,
        use_struct_mask=params.struct_mask_prob > 0,
        target_field=params.targets,
        freeze_text_encoder=params.freeze_text_encoder,
        fusion_type=params.fusion_type,
    )

    # if params.fusion_type == "cross_attn":
    #     collate_fn = collate_fn_fusion_tokenlevel
    # else:
    #     collate_fn = None

    if torch.cuda.device_count() == 1 or not ddp:
        # print(f"[Sanity Check] DataLoader will use collate_fn: {collate_fn}")
        # print(f"[Sanity Check] collate_fn type: {type(collate_fn)}")
        # print(f"[Sanity Check] collate_fn repr: {repr(collate_fn)}")
        # print(f"[Sanity Check] Using collate_fn: {collate_fn}")

        train_loader = DataLoader(
            train_dataset, 
            batch_size=params.batch_size, 
            shuffle=True, 
            num_workers=num_workers, 
            drop_last=True, 
            worker_init_fn=seed_worker, 
            pin_memory=True, 
            # collate_fn=collate_fn_fusion_tokenlevel
            )

        ### debug
        print(f"[Debug] train_dataset length: {len(train_dataset)}")
        print("[Debug] Checking training dataset...")


        try:
            # print(f"[Sanity Check] train_loader actual collate_fn: {train_loader.collate_fn.__name__}")
            first_batch = next(iter(train_loader))
            print(f"[Debug] x shape: {first_batch.x.shape}")
            print(f"[Debug] pos shape: {first_batch.pos.shape}")
            print(f"[Debug] y: {first_batch.y if hasattr(first_batch, 'y') else 'None'}")
            assert first_batch.x.shape[0] > 0, " Empty training graph! Check if filter or data is broken."

            masked_struct = (first_batch.x.abs().sum(dim=1) == 0).sum().item()
            masked_text = (first_batch.text_emb.abs().sum(dim=(1, 2)) == 0).sum().item()
            print(f"[Batch DEBUG] struct masked: {masked_struct}/{len(first_batch.y)}")
            print(f"[Batch DEBUG] text masked: {masked_text}/{len(first_batch.y)}")
            ## multimodal
            ## check text inserted or not?
            # print(f"text: {first_batch.text if hasattr(first_batch, 'text') else 'No text'}")

        except Exception as e:
            print(f"Failed to load training data: {e}")
        ### debug over

        val_loader  = DataLoader(
            val_dataset, 
            batch_size=params.batch_size, 
            shuffle=False, 
            num_workers=num_workers, 
            drop_last=False, 
            pin_memory=True, 
            # collate_fn=collate_fn
            )
        test_loader = DataLoader(
            test_dataset, 
            batch_size=1, 
            shuffle=False, 
            num_workers=num_workers, 
            drop_last=False, 
            # collate_fn=collate_fn
            )
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

    #* Uncomment below to force the updating of the cache files.
    # train_dataset.process()
    # val_dataset.process()
    # test_dataset.process()

    #* Setup model and trainer
    ## logger (tensorBoard) 记录日志
    logger = loggers.TensorBoardLogger(params.save_path, name=params.experiment_name, default_hp_metric=False)
    logger.log_hyperparams(params.__dict__, \
        {"hp/val": 1.0, "hp/avr50":1.0, "hp/min_avr50":1.0, "hp/min":1.0, "hp/mean_min10": 1.0}
    )

    ## multimodal
    ## log + fusion type, text encoder, modality mask?
    logger.log_hyperparams({**params.__dict__})

    ckpt_dir=logger.log_dir+'/model_checkpoint'
    checkpoint_callback = ModelCheckpoint(save_top_k=params.model_checkpoint_save_top_k,
                                        monitor='val/loss', mode='min', dirpath=ckpt_dir)  # save ckpt with min valid loss

    ## model initialization

    ### multimodal: add StructureEncoder
    ### fusion则import新的regression framework, 也设置成一样的名字让改动最小, 前面根据fusion_type选择import哪一个
    if params.fusion_type == "none":
        from models.regression import RegressionModel
        system = RegressionModel(params, train_loader, val_loader)
    else:
        from multimodal_fusion.regression_fusion import RegressionModelFusion as RegressionModel
        system = RegressionModel(
            params, 
            StructureEncoder, 
            train_loader, 
            val_loader,
            dataset_mean=train_dataset.target_mean,
            dataset_std=train_dataset.target_std
            )


    ### multimodal
    ### parameter statistics (compatible with unimodal & multimodal)

    total_params = num_params(system)
    print(f"[Total Model] Params: {total_params:,} ({total_params * 4 / 1024**2:.2f} MB)")

    # optional: First encoder block stats (if available)
    if hasattr(system, "structure_encoder") and hasattr(system.structure_encoder, "encoder"):
        if hasattr(system.structure_encoder.encoder, "layers"):
            first_block = system.structure_encoder.encoder.layers[0]
            block_params = count_params(first_block)
            print(f"[1st Encoder Block] Params: {block_params:,} ({block_params * 4 / 1024:.2f} KB)")


    # initialize mean and std values in crystalformer by forwarding once. 
    ## initialize encoder

    ## multimodal: 只有crystalformer + ddp 才需要
    if ddp and params.encoder_name == "latticeformer":
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

    # ckpt loading

    ### multimodal 每个模态都可能有可能有不同的pretrained的model
    ## 1. multimodal model ckpt(cross-attention/gated + 2 encoders + prediction head...)
    if params.pretrained_model is not None:
        # lightning only
        print(f"Loading pretrained checkpoint from {params.pretrained_model}")
        system = RegressionModel.load_from_checkpoint(
            params.pretrained_model,
            params=params,
            structure_encoder=StructureEncoder,
            train_loader=train_loader,
            val_loader=val_loader,
            strict=False)
    ## 2. initialize structure encoder with ckpt only
    elif hasattr(system, "structure_encoder") and getattr(params, "struct_encoder_ckpt", None):
        ckpt_path = params.struct_encoder_ckpt
        print(f"Initializing structure encoder from: {ckpt_path}")
        state = torch.load(ckpt_path, map_location='cpu')
        if "state_dict" in state:
            state = state["state_dict"]
        state = {k.replace("model.", ""): v for k, v in state.items() if "model." in k}
        system.structure_encoder.load_state_dict(state, strict=False)    


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
        deterministic=deterministic,  # whether fixed randomness

        ### add
        accelerator="gpu" if torch.cuda.is_available() else "cpu",
        precision=16 if getattr(params, "use_amp", False) else 32,
        log_every_n_steps=params.log_every_n_steps
        )

    
    ## trainning + time recording(crystalformer主打的耗时少)
    ### 
    import time
    time_dict = {}
    scores = []

    print("\n>> Start training ...\n")  ##
    start_time = time.time()
    trainer.fit(system)  # run
    # trainer.fit(system, train_dataloaders=train_loader, val_dataloaders=val_loader)  # run
    time_dict['time-train'] = time.time() - start_time
    
    ## save the last ckpt
    last_ckpt_path = os.path.join(ckpt_dir, 'last.ckpt')  ##
    os.makedirs(ckpt_dir, exist_ok=True)  ##
    trainer.save_checkpoint(last_ckpt_path)  # ensure checkpointing after SWA's BN updating
    print(f"\n>> Saved final ckpt to {last_ckpt_path}")

    ## swa
    # support stochastic weight averaging(就是paper中的两种model)
    if hasattr(system, "enable_average_model"):  ##
        # validation with SWA model
        if system.enable_average_model('val-swa'):
            print(">> Evaluating with val-swa model ...")
            start_time = time.time()
            val_score = trainer.validate(model=system, dataloaders=val_loader)
            time_dict['time-val-swa'] = time.time() - start_time
            scores += val_score

        # test with SWA model
        if system.enable_average_model('test-swa'):
            print(">> Evaluating with test-swa model ...")
            start_time = time.time()
            test_score = trainer.test(model=system, dataloaders=test_loader)
            time_dict['time-test-swa'] = time.time() - start_time
            scores += test_score

        # reset to normal model
        system.disable_average_model()

    # prepare the best model for testing
    ## with minimize valid loss
    if os.path.exists(checkpoint_callback.best_model_path):
        print(f">> Loading best model from {checkpoint_callback.best_model_path}")
        best_model = RegressionModel.load_from_checkpoint(  # load
            checkpoint_callback.best_model_path,
            params=params,
            train_loader=train_loader,
            val_loader=val_loader,
            structure_encoder=StructureEncoder,  ##
            dataset_mean=train_dataset.target_mean,  ##
            dataset_std=train_dataset.target_std,  ##
            strict=False)  ## 
        system.model = best_model.model  # replace the last one

        ### multimodal
        if hasattr(system, "structure_encoder"):
            system.structure_encoder = best_model.structure_encoder
        if hasattr(system, "text_encoder"):
            system.text_encoder = best_model.text_encoder
        if hasattr(system, "fusion_module"):
            system.fusion_module = best_model.fusion_module
        system.disable_average_model()
        trainer.save_checkpoint(os.path.join(ckpt_dir, 'best.ckpt'))

    ### final val & test

    ## validation set -> loss
    start_time = time.time()
    # scores += trainer.validate(model=system, dataloaders=val_loader)
    val_score = trainer.validate(model=system, dataloaders=val_loader)
    if val_score is not None:
        scores += val_score
    time_dict['time-val'] = (time.time()-start_time)

    ## test set
    start_time = time.time()
    test_score = trainer.test(model=system, dataloaders=test_loader)
    if test_score is not None:
        scores += test_score
    # scores += trainer.test(model=system, dataloaders=test_loader)
    time_dict['time-test'] = (time.time()-start_time)

    ## consuming time + record into log
    print("========= Training & Evaluation Summary =========")
    for k, v in time_dict.items():
        print(f"{k}: {v:.2f} seconds")
    if scores:
        for score in scores:
            for key, val in score.items():
                print(f"[{key}]: {val:.4f}")
    with open(os.path.join(logger.log_dir, 'time.txt'), 'w') as f:
        for k, v in time_dict.items():
            print(f"{k}: {v:.4f}", file=f)
        for score in scores:
            for key, val in score.items():
                print(f"{key}\t: {val:.6f}", file=f)

    logger.finalize('success')  # to properly output all test scores in a TB log.

if __name__ == '__main__':
    train()
