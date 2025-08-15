
### train_fusion.py -> unimodal + multimodal
### regression.py -> unimodal only
### regression_fusion.py -> multimodal only

import torch
import torch.nn.functional as F
import torch.nn as nn
from pytorch_lightning.core import LightningModule
import torch.optim as optim
from losses.regression_loss import regression_loss  # eval as unimodal
from models.latticeformer import Latticeformer   # default structure encoder(crystalformer)
from torch.optim import lr_scheduler
import numpy
from torch.optim import swa_utils
from typing import Callable

### multimodal
from multimodal_fusion.text_encoder import build_text_encoder
from multimodal_fusion.fusion_block import FusionBlock
from multimodal_fusion.fusion_loss import fusion_regression_loss
from torch_scatter import scatter_mean

# ... also need to import structure encoder if replacing it


class AvgFn:  # for swa model; allows custom averaging behavior (e.g., how and when to update parameters)
    def __call__(self, averaged_model_parameter, model_parameter, num_averaged):
        return averaged_model_parameter + \
            (model_parameter - averaged_model_parameter) / (num_averaged + 1)
    
class RegressionModelFusion(LightningModule):

    ### multimodal

    def __init__(
        self, 
        params, 
        structure_encoder, 
        train_loader, 
        val_loader, 
        dataset_mean=None, 
        dataset_std=None):
        super().__init__()

        ## hyperparams
        self.params = params
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.save_hyperparameters(params.__dict__)
        self.automatic_optimization = False
        self.clip_norm = getattr(params, "clip_norm", 0.0)
        self.clip_grad = getattr(params, "clip_grad", 0.0)

        ## targets
        self.targets = self.params.targets
        if isinstance(self.targets, str):
            self.targets = [self.targets]
        self.val_history = []

        ## structure_encoder
        self.structure_encoder = structure_encoder(params)
        
        ## text_encoder, 默认冻结参数
        self.text_encoder = build_text_encoder(params)
        if getattr(params, "freeze_text_encoder", True):  # default frozen
            for param in self.text_encoder.parameters():
                param.requires_grad = False

        ## fusion model & predictor head
        self.fusion_module = FusionBlock(params.fusion_type, params.model_dim)
        fusion_pred_dim = self.fusion_module.output_dim
        
        #self.predictor = nn.Linear(fusion_pred_dim, len(self.targets))
        self.predictor = nn.ModuleDict({
            "fusion": nn.Linear(fusion_pred_dim, len(self.targets)),
            "struct": nn.Linear(params.model_dim, len(self.targets)),
            "text": nn.Linear(params.model_dim, len(self.targets)),
        })

        ## normalization param
        '''
        target_std = torch.ones((len(self.targets)), dtype=torch.float32)
        target_mean = torch.zeros((len(self.targets)), dtype=torch.float32)
        self.register_buffer('target_std', target_std)
        self.register_buffer('target_mean', target_mean)
        self.normalize_targets = getattr(params, "normalize_targets", "no")
        # if self.normalize_targets in ("scale_bias", "bias", "scale"):
        #     self.update_target_normalizers()
        '''
        self.normalize_targets = getattr(params, "normalize_targets", "no")

        if isinstance(dataset_mean, float):  # single target
            target_mean = torch.tensor([dataset_mean], dtype=torch.float32)
            target_std = torch.tensor([dataset_std], dtype=torch.float32)
        else:  # multi targets
            target_mean = torch.tensor(dataset_mean, dtype=torch.float32)
            target_std = torch.tensor(dataset_std, dtype=torch.float32)

        self.register_buffer('target_mean', target_mean)
        self.register_buffer('target_std', target_std)


        # swa ignore
        self.swa_epochs = getattr(params, "swa_epochs", 0)
        if self.swa_epochs > 0:
            #* In DDP, classes can't have function pointers as members, so define avg_fn as a class.
            self.swa_model = swa_utils.AveragedModel(self.get_trainable_model())
        else:
            self.swa_model = None
        self.use_average_model: bool = False
        self.logging_key: str = None
        self.validation_step_outputs = []

    def get_trainable_model(self):
        return nn.ModuleDict({
            "structure_encoder": self.structure_encoder,
            "fusion_module": self.fusion_module,
            "predictor": self.predictor,
        })

    def get_trainable_parameters(self):
        params = list(self.structure_encoder.parameters()) + \
                list(self.fusion_module.parameters()) + \
                list(self.predictor.parameters())
        if not getattr(self.params, "freeze_text_encoder", True):
            params += list(self.text_encoder.parameters())
        return params
    
    def get_trainable_modules(self):
        modules = [self.structure_encoder, self.fusion_module, self.predictor]
        if not getattr(self.params, "freeze_text_encoder", True):
            modules.append(self.text_encoder)
        return modules

    # ignore these swa-related functions for now
    def enable_average_model(self, logging_key:str=None) -> bool:
        if self.swa_model is not None:
            self.logging_key = logging_key
            self.use_average_model = True
            return True
        return False
        
    def disable_average_model(self):
        self.logging_key = None
        self.use_average_model = False
    
    '''
    # normalize targets to the same scale since different targets may have different value ranges
    # this ensures loss is balanced across targets, especially in multi-task settings

    def update_target_normalizers(self):
        all_targets = []
        
        for data in self.train_loader.dataset:
            targets = []
            for target in self.targets:
                val = getattr(data, target, None)
                if val is None:
                    raise AttributeError(f"Target {target} not found in dataset sample.")
                targets.append(val.view(1) if val.dim()==0 else val)
            targets = torch.cat(targets).view(1, -1)  # shape: [1, num_targets]
            all_targets.append(targets)
            
        target_vals = torch.cat(all_targets, dim=0).T  # shape: [num_targets, num_samples]

        if "bias" in self.normalize_targets:
            target_mean = torch.mean(target_vals, dim=1)
        else:
            target_mean = torch.zeros((len(self.targets)), dtype=torch.float32)
        
        if "scale" in self.normalize_targets:
            target_std = ((target_vals - target_mean[:, None]) ** 2).mean(dim=1).sqrt()
        else:
            target_std = torch.ones((len(self.targets)), dtype=torch.float32)

        print("Computing normalizing scales and biases for target values ---")
        for i, t in enumerate(self.targets):
            print(f"{t}\t: scale={target_std[i].item():.4f}\t bias={target_mean[i].item():.4f}")
        print("-------------------------------------------------------------")

        self.target_std[:] = target_std.to(self.target_std.device)
        self.target_mean[:] = target_mean.to(self.target_mean.device)
    '''
    
    def load_state_dict(self, state_dict, strict: bool = True):
        new_dict = {}
        for key, val in state_dict.items():
            if key.startswith("model.structure_encoder."):
                new_dict["structure_encoder." + key[len("model.structure_encoder."):]] = val
            elif key.startswith("model.text_encoder."):
                new_dict["text_encoder." + key[len("model.text_encoder."):]] = val
            elif key.startswith("model.fusion_module."):
                new_dict["fusion_module." + key[len("model.fusion_module."):]] = val
            elif key.startswith("model.predictor.fusion."):
                new_dict["predictor.fusion." + key[len("model.predictor.fusion."):]] = val
            elif key.startswith("model.predictor.struct."):
                new_dict["predictor.struct." + key[len("model.predictor.struct."):]] = val
            elif key.startswith("model.predictor.text."):
                new_dict["predictor.text." + key[len("model.predictor.text."):]] = val

            else:
                new_dict[key] = val
        if strict:
            return super().load_state_dict(new_dict, strict)
        else:
            missing, unexpected = super().load_state_dict(new_dict, strict = False)
            print(f"[load_state_dict] Missing keys: {missing}")
            print(f"[load_state_dict] Unexpected keys: {unexpected}")
            return missing, unexpected
    
    def forward(self, batch): 

        # assert batch.x is not None, f"Input batch.x is None for {batch.material_id}"  ## debug
        # #print("[Debug] batch keys:", batch.keys)  ## debug
        # #print("[Debug] batch.x:", batch.x)  ## debug
        # # input() 
        # struct_emb = self.structure_encoder(batch)
        # # input() 

        '''        #text_emb = self.text_encoder(batch.text)
        ## frozen text encoder时用提前处理的text embedding
        # if self.params.freeze_text_encoder and hasattr(batch, "text_emb"):
        if self.params.freeze_text_encoder:##
            # print(f"[Debug] batch keys: {batch.__dict__.keys()}")  ##
            if hasattr(batch, "text_emb"):
                # print(f"[Debug] batch.text_emb shape: {batch.text_emb.shape}")
                text_emb = batch.text_emb
                # print("=========================11111")
                # input()
            else:
                raise ValueError("freeze_text_encoder=True but text_emb not found in batch! Check Dataset + DataLoader.")
        else:
            # print("=========================22222")
            # input()
            text_emb = self.text_encoder(batch.text)
        '''

        # print(f"[Forward] batch.text_emb shape: {getattr(batch, 'text_emb', None).shape}")

        # batch text_emb（pooled/token-level）
        if self.params.freeze_text_encoder:
            if not hasattr(batch, "text_emb"):
                raise ValueError("freeze_text_encoder=True but text_emb not found in batch! Check Dataset + DataLoader.")
            text_emb = batch.text_emb  # token-level or pooled depending on fusion_type
        else:
            text_emb = self.text_encoder(batch.text)  # will return token or pooled depending on setting

        # pooled-level mean pooling only
        if self.params.fusion_type in ["concat", "sum", "gated"]:
            # [B, L, D] -> [B, D]
            if text_emb.dim() == 3:
                text_emb = text_emb.mean(dim=1)
        # # print('=======================222222222222222222222') 
        # # input() 

        # fused = self.fusion_module(struct_emb, text_emb)
        # # print('=======================3333333333333333333333') 
        # # input() 

        ##### cross_attn
        if self.params.fusion_type == "cross_attn":
            # print(f"batch: {batch}")
            # input()
            struct_token, struct_batch = self.structure_encoder(batch, return_tokens=True)
            
            '''            
            print(f"[Debug] struct_token shape: {struct_token.shape}")   # [N_token, D]
            print(f"[Debug] struct_batch shape: {struct_batch.shape}")   # [N_token]
            print(f"[Debug] text_emb type: {type(text_emb)}")
            print(f"[Debug] text_emb shape: {text_emb.shape}")
            '''
            fused = self.fusion_module(struct_token, text_emb, struct_batch=struct_batch)  # struct_token shape: [N_token, D]
            # struct_emb = self.structure_encoder.proj_before_pooling(struct_token)
            # struct_emb = self.structure_encoder.pooling_layer(struct_emb, struct_batch, struct_batch.max().item() + 1)
            struct_proj = self.structure_encoder.proj_before_pooling(struct_token)
            B = struct_batch.max().item() + 1
            struct_emb = scatter_mean(struct_proj, struct_batch, dim=0, dim_size=struct_batch.max().item() + 1)
        else:
            struct_emb = self.structure_encoder(batch)  # [B, D]
            fused = self.fusion_module(struct_emb, text_emb)


        fusion_pred = self.predictor["fusion"](fused)
        struct_pred = self.predictor["struct"](struct_emb)
        text_pred = self.predictor["text"](text_emb)
        return fusion_pred, struct_pred, text_pred

        # print(f"[Debug] struct_emb shape: {struct_emb.shape}")  ## 
        # print(f"[Debug] text_emb shape: {text_emb.shape}")  ##
        # print(f"[Debug] fused shape: {fused.shape}, expected: {self.fusion_module.output_dim}")  ##


    def test_step(self, batch, batch_idx):
        return self.validation_step(batch, batch_idx)

    # pytorch lightning training step
    def training_step(self, batch, batch_idx):
        # print("⚠️⚠️ training_step batch.text_emb shape:", getattr(batch, "text_emb", None).shape)
        # optimizer setting & gradient initialized as zero

        opt = self.optimizers()  # current optimizer
        opt.zero_grad()  # zero grad
 
        # batchNorm(BN layer) frozen
        ## norm_type could be set as 'ln' 'bn' 'no' ...if batchnorm
        freeze_bn_epochs = getattr(self.params, 'freeze_bn_epochs', 0)
        ## for stable training, freeze BN in the last 'freeze_bn_epochs' epochs
        if self.current_epoch + freeze_bn_epochs >= self.params.n_epochs:  # 倒数的'freeze_bn_epochs'个epoch都结BN, enter frozen epoch
            for m in self.structure_encoder.modules():
                if isinstance(m, (nn.BatchNorm1d, nn.SyncBatchNorm)):
                    m.eval()
            for m in self.fusion_module.modules():
                if isinstance(m, (nn.BatchNorm1d, nn.SyncBatchNorm)):
                    m.eval()
                    
        fusion_pred, struct_pred, text_pred = self.forward(batch)

        target = torch.cat([getattr(batch, t).view(-1, 1) for t in self.targets], dim=-1)
        # print(f"[Training Step] pred_fusion: {fusion_pred.shape}, target: {target.shape}")

        if self.normalize_targets in ("scale", "scale_bias"):
            target = (target - self.target_mean) / self.target_std
        elif self.normalize_targets == "bias":
            target = target - self.target_mean

        ### fusion loss
        loss, loss_f, loss_s, loss_t = fusion_regression_loss(
            fusion_pred, struct_pred, text_pred, 
            batch, self.targets, self.target_std, self.target_mean,
            loss_type=self.params.loss_func,
            weights=getattr(self.params, 'fusion_loss_weights', [1.0, 0.0, 0.0])  # 在default_fusion.json加
        )
        loss = loss.mean()
        loss_f, loss_s, loss_t = map(lambda l: l.mean(), [loss_f, loss_s, loss_t])

        # backpropogation
        self.manual_backward(loss)

        if self.clip_norm > 0:  # prevent gradient exploding
            total_norm = nn.utils.clip_grad.clip_grad_norm_(self.get_trainable_parameters(), self.clip_norm)
            self.log('train/total_norm', total_norm, on_step=False, on_epoch=True,
                     prog_bar=False, logger=True, batch_size=batch.batch_size if hasattr(batch, 'batch_size') else None)

        if self.clip_grad > 0:
            nn.utils.clip_grad.clip_grad_value_(self.get_trainable_parameters(), self.clip_grad)

        opt.step()  # optimizer update

        # swa
        swa_enabled = self.swa_epochs + self.current_epoch >= self.params.n_epochs
        if swa_enabled and self.swa_model is not None:
            self.swa_model.update_parameters(self.get_trainable_model())

        # sceduler for learning rate
        sch = self.lr_schedulers()
        if sch is not None and not swa_enabled:
            sch.step()

        #* note: make sure disabling on_step logging, which may frequently
        #* cause unexpected pauses due to the logging IO when the disc is on a NAS.
        
        # log
        bsz = target.shape[0]
        self.log('train/loss', loss, on_step=True, on_epoch=True, logger=True, batch_size=bsz)
        self.log('train/loss_fusion', loss_f, on_step=True, on_epoch=True, logger=True, batch_size=bsz)
        self.log('train/loss_struct', loss_s, on_step=True, on_epoch=True, logger=True, batch_size=bsz)
        self.log('train/loss_text', loss_t, on_step=True, on_epoch=True, logger=True, batch_size=bsz)
        return {'loss': loss}

    # update param of BN in swa model
    ## swa
    def on_train_end(self):
        print("Updating BNs for Stochastic Weight Averaging")
        device = self.swa_model.parameters().__next__().device
        torch.optim.swa_utils.update_bn(self.train_loader, self.swa_model, device)

    # validation for single batch
    def validation_step(self, batch, batch_idx):
        fusion_pred, _, _ = self.forward(batch)

        # this is the standard loss used for validation (e.g., MAE); only training loss is customized
        loss = regression_loss(
            fusion_pred, 
            batch, 
            self.targets, 
            self.target_std, 
            self.target_mean, 
            self.params.loss_func
        )

        outputs = {}
        outputs['val/loss'] = loss.detach().cpu()
        outputs['outputs'] = fusion_pred.detach().cpu()

        for i, t in enumerate(self.targets):
            label = getattr(batch, t).detach().cpu()
            pred = (fusion_pred[:, i] * self.target_std[i] + self.target_mean[i]).detach().cpu()
            mae = torch.abs(pred - label)
            outputs[t] = mae

        self.validation_step_outputs.append(outputs)
        return outputs

    # callback after validation epoch ends
    def on_validation_epoch_end(self):
        outputs = self.validation_step_outputs
        if len(outputs)==0:
            return
        if all('val/loss' in x for x in outputs):
            avg_loss = torch.cat([x['val/loss'] for x in outputs], dim=0).mean()  # 取batch的avg loss as val loss
        else:
            raise KeyError("Missing 'val/loss' in validation_step_outputs")
            
        logging_key = self.logging_key or 'val'
        self.log(f'{logging_key}/loss', avg_loss, prog_bar=True)
        print(f"\n[Epoch {self.current_epoch}] {logging_key}/loss: {avg_loss:.4f}  ", end='')
        
        # target dim record mae
        for t in self.targets:
            if all(t in x for x in outputs):
                mae = torch.cat([x[t] for x in outputs], dim=0).mean()
                self.log(f'{logging_key}/{t}', mae, prog_bar=True)
                print(f"{t}: {mae.item():.4f}  ", end='')
            else:
                print(f"\n[Warning] Missing target {t} in validation outputs")
        print("")

        # record metrics for early stop and convergence tracking
        if self.logging_key is None:
            self.val_history.append(avg_loss.item())
            v = numpy.array(self.val_history, dtype=numpy.float64)
            if len(v) >= 50:
                K = 50  # mean filter width
                k = numpy.ones(K, dtype=numpy.float64)
                m = numpy.convolve(v, k)[:-(K - 1)] / numpy.convolve(numpy.ones_like(v), k)[:-(K - 1)]
                self.log("hp/avr50", m[-1])  # average over 50
            if len(v) >= 10:
                T = 10
                r = numpy.sort(v)[:T]
                self.log("hp/mean_min10", r.mean())
            self.log("hp/min", v.min())
            self.log("hp/val", avg_loss)

            # clear, prevent oom
            self.validation_step_outputs.clear()
            self.logging_key = None
     
    def on_test_epoch_end(self):
        outputs = self.validation_step_outputs
        if len(outputs) == 0:
            return
        avg_loss = torch.cat([x['val/loss'] for x in outputs], dim=0).mean()
        logging_key = self.logging_key if self.logging_key is not None else 'test'
        print(f"\n[test loss] {avg_loss:.4f}")
        self.log(f'{logging_key}/loss', avg_loss)
        for t in self.targets:
            if all(t in x for x in outputs):
                v = torch.cat([x[t] for x in outputs], dim=0).mean()
                self.log(f'{logging_key}/{t}', v)
                print(f"{t}: {v.item():.4f}", end='  ')
        print('')
        self.validation_step_outputs.clear()

    def configure_optimizers(self):
        lr = self.params.lr
        opt = getattr(self.params, 'optimizer', 'adam')
        weight_decay = getattr(self.params, 'weight_decay', 0.0)

        opt_params = [{'params': self.get_trainable_parameters()}]

        opt_args = {'lr': lr}

        if opt == 'adam':
            opt_args['betas'] = self.params.adam_betas
            opt = optim.Adam
        elif opt == 'adamw':
            opt_args['betas'] = self.params.adam_betas
            opt = optim.AdamW
        elif opt == 'sgd':
            opt_args['momentum'] = self.params.momentum
            opt = optim.SGD
        else:
            raise NotImplementedError(f'Unknown optimizer: {self.params.optimizer}')

        sch = None
        if self.params.lr_sch == "const":
            return opt(opt_params, **opt_args)
        elif self.params.lr_sch == "inverse_sqrt_nowarmup":
            decay = self.params.sch_params[0]
            f = lambda t: (decay / (decay + t)) ** 0.5
            opt = opt(opt_params, **opt_args)
            sch = lr_scheduler.LambdaLR(opt, f)
        elif self.params.lr_sch == "inverse_sqrt_nowarmup_dmodel":
            decay = self.params.sch_params[0]
            f = lambda t: self.params.model_dim ** -0.5 * (decay / (decay + t)) ** 0.5
            opt_args['lr'] = lr * f(0)
            opt = opt(opt_params, **opt_args)
            sch = lr_scheduler.LambdaLR(opt, lambda t: f(t) / f(0))
        elif self.params.lr_sch == "inverse_sqrt_warmup":
            warmup_steps = self.params.sch_params[0]
            f = lambda t: self.params.model_dim ** -0.5 * min((t + 1) ** -0.5, (t + 1) * warmup_steps ** -1.5)
            opt_args['lr'] = f(0)
            opt = opt(opt_params, **opt_args)
            sch = lr_scheduler.LambdaLR(opt, lambda t: f(t) / f(0))
        elif self.params.lr_sch == "inverse_sqrt_warmup_lrmax":
            warmup_steps = self.params.sch_params[0]
            f = lambda t: warmup_steps ** 0.5 * min((t + 1) ** -0.5, (t + 1) * warmup_steps ** -1.5)
            opt_args['lr'] = lr * f(0)
            opt = opt(opt_params, **opt_args)
            sch = lr_scheduler.LambdaLR(opt, lambda t: f(t) / f(0))
        elif self.params.lr_sch == "multistep":
            opt = opt(opt_params, **opt_args)
            sch = lr_scheduler.MultiStepLR(opt, milestones=self.params.sch_params, gamma=0.1)
        else:
            raise NotImplementedError(f'Unknown lr_sch: {self.params.lr_sch}')

        return [[opt], [sch]]

    def train_dataloader(self):
        # print("⚠️⚠️ Lightning uses train_dataloader()!!")
        return self.train_loader

    def val_dataloader(self):
        return self.val_loader
