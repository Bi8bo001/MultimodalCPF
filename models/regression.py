
import torch
import torch.nn.functional as F
import torch.nn as nn

from pytorch_lightning.core import LightningModule
import torch.optim as optim
from losses.regression_loss import regression_loss  # 改成加权的loss函数

from models.latticeformer import Latticeformer   # default structure encoder(crystalformer)

## define text encoder & fusion block
''' 
from multimodal_fusion.text_encoder import build_text_encoder
from multimodal_fusion.fusion_block import FusionBlock
from multimodal_fusion.fusion_loss import fusion_regression_loss
from multimodal_fusion.encoder_selector import get_structure_encoder
# ... 还有需要替换的structure encoder 也需要import
'''

from torch.optim import lr_scheduler
import numpy
from torch.optim import swa_utils
from typing import Callable

class AvgFn:  # for SWAmodel, 可以自定义params (如何取avg，多少取avg...)
    def __call__(self, averaged_model_parameter, model_parameter, num_averaged):
        return averaged_model_parameter + \
            (model_parameter - averaged_model_parameter) / (num_averaged + 1)
    
class RegressionModel(LightningModule):
    def __init__(self, params, train_loader, val_loader):
    ## multimodal
    #def _init_(self, params, structure_encoder, train_loader, val_loader):
        super(RegressionModel, self).__init__()

        ## multimodal里面删掉这个
        if params.encoder_name == 'latticeformer':  # structure encoder(Latticeformer)
            self.model = Latticeformer(params)
        
        ### multimodal: 要换掉这个单个的self.model
        '''
        self.structure_encoder = structure_encoder(params)
        self.text_encoder = build_text_encoder(params)
        self.fusion_module = FusionBlock(params.fusion_type, params.model_dim)
        fusion_out_dim = params.model_dim if params.fusion_type in ['sum', 'gated'] else params.model_dim * 2
        self.predictor = nn.Linear(fusion_out_dim, len(params.targets))
        # ...
        '''
        ## 最好是每种encoder都是可选的
        ## build_text_encoder 定义在 multimodal_fusion/text_encoder.py 中，根据 params.text_encoder_type 调用不同模型（BERT, Qwen...）

        else:
            raise Exception(f"Invalid params.encoder_name: {params.encoder_name}")
        
        # swa 暂时先不管
        self.swa_epochs = getattr(params, "swa_epochs", 0)
        if self.swa_epochs > 0:
            #* In DDP, classes can't have function pointers as members, so define avg_fn as a class.
            self.swa_model = swa_utils.AveragedModel(self.model)
        else:
            self.swa_model = None

        # param setting
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.params = params
        self.save_hyperparameters(params.__dict__)
        self.automatic_optimization = False
        self.clip_norm = getattr(params, "clip_norm", 0.0)
        self.clip_grad = getattr(params, "clip_grad", 0.0)

        # 支持list of target
        ## 注意这个部分需要在predictor前def
        self.targets = self.params.targets
        if isinstance(self.targets, str):
            self.targets = [self.targets]
        self.val_history = []

        '''
        out_dim = len(self.targets)
        self.predictor = nn.Linear(fusion_out_dim, out_dim)  # fusion后的维度根据fusion type决定
        '''

        # normalization param
        target_std = torch.ones((len(self.targets)), dtype=torch.float32)
        target_mean = torch.zeros((len(self.targets)), dtype=torch.float32)
        self.register_buffer('target_std', target_std)
        self.register_buffer('target_mean', target_mean)
        self.normalize_targets = getattr(params, "normalize_targets", "no")
        if self.normalize_targets in ("scale_bias", "bias", "scale"):
            self.update_target_normalizers()

        # valid set output setting
        self.use_average_model:bool = False
        self.logging_key:str = None
        self.validation_step_outputs = []

    # swa相关 这2个function先不用管
    def enable_average_model(self, logging_key:str=None) -> bool:
        if self.swa_model is not None:
            self.logging_key = logging_key
            self.use_average_model = True
            return True
        return False
        
    def disable_average_model(self):
        self.logging_key = None
        self.use_average_model = False
    
    # target normalization
    ## 因为不同target的范围值不一样 需要通过normalization让target都在同一个scale 从而让loss也在同一个scale 尤其是在有多个target的时候
    def update_target_normalizers(self):
        target_vals = [getattr(self.train_loader.dataset.data, t) for t in self.targets]
        # data就是那个 data={'x': .., 'pos':.., 'target':.., ...}
        target_vals = torch.stack(target_vals, dim=0)  # 将不同target拼接在一起

        if "bias" in self.normalize_targets:
            target_mean = torch.mean(target_vals, dim=1)
        else:  # by default unbiased
            target_mean = torch.zeros((len(self.targets)), dtype=torch.float32) 
        if "scale" in self.normalize_targets:
            target_std = ((target_vals-target_mean[:, None])**2).mean(dim=1)**0.5
        else:  # by default std=1
            target_std = torch.ones((len(self.targets)), dtype=torch.float32)

        print("Computing normalizing scales and biases for target values ---")
        for i, t in enumerate(self.targets):
            print(f"{t}\t: scale={target_std[i].item()}\t bias={target_mean[i].item()}")
        print("-------------------------")

        self.target_std[:] = target_std.to(self.target_std.device)  # 注册为buffer 不被optimizer优化 但随model保存
        self.target_mean[:] = target_mean.to(self.target_mean.device)

    def load_state_dict(self, state_dict, strict: bool = True):
        # Override for backward compatibility
        new_dict = {}
        for key in state_dict:
            if key.startswith("model.xrd_"):
                # replace 'model' with 'model_xrd'
                new_dict['model_xrd' + key[5:]] = state_dict[key]
            else:
                new_dict[key] = state_dict[key]
            ### multimodal需要改的 因为_init_里面都已经把self.model改成结构文本等等拆开了

        return super().load_state_dict(new_dict, strict)
    
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
            elif key.startswith("model.predictor."):
                new_dict["predictor." + key[len("model.predictor."):]] = val
            else:
                new_dict[key] = val
        return super().load_state_dict(new_dict, strict)
    '''

    # pytorch lightning training的接口 每个miini-batch调用一次
    def training_step(self, batch, batch_idx):

        # optimizer setting & gradient initialized as zero
        opt = self.optimizers()  # 获得当前的optimizer
        opt.zero_grad()  # 清理旧的梯度 归零

        # batchNorm(BN layer) frozen
        ## norm_type可以设置为 'ln' 'bn' 'no' ...如果是batchnorm才适用
        freeze_bn_epochs = getattr(self.params, 'freeze_bn_epochs', 0)
        ## infer的时候用前期稳定的 BN，防止后面batch size变小不稳定
        if self.current_epoch + freeze_bn_epochs >= self.params.n_epochs:  # 倒数的'freeze_bn_epochs'个epoch都结BN, enter frozen epoch
            for m in self.model.modules():
                if isinstance(m, (torch.nn.BatchNorm1d, torch.nn.SyncBatchNorm)):
                    m.eval()

            ### multimodal 需要修改self.model...
            '''
            for m in self.structure_encoder.modules():
                if isinstance(m, (nn.BatchNorm1d, nn.SyncBatchNorm)):
                    m.eval()
            for m in self.fusion_module.modules():
                if isinstance(m, (nn.BatchNorm1d, nn.SyncBatchNorm)):
                    m.eval()
            # text_encoder all frozen，no BN，no eval()
            '''  # 这个最后还是放到外面去吧 后面weight decay的部分也需要调用

        output = self.forward(batch)  # structure-only forward, self.model就是latticeformer(..)

        ### multimodal 需要重新def multimodal的forward, modify 'output'
        '''
        output = self.forward(batch, text_)
        '''

        # loss function
        loss = regression_loss(output, batch, self.targets, self.target_std, self.target_mean, self.params.loss_func)
        loss, bsz = loss.mean(), loss.shape[0]

        ### multimodal: 根据需要重新定义joint loss function for training process
        '''
        output_fusion, output_struct, output_text = self.forward(batch)
        loss = regression_fusion_loss(
            output_fusion, output_struct, output_text, 
            batch, self.targets, self.target_std, self.target_mean,
            loss_type=self.params.loss_func,
            weights=self.params.get('fusion_loss_weights', [1.0, 0.0, 0.0])  # 在default_fusion.json加
        )
        loss, bsz = loss.mean(), loss.shape[0]
        '''

        # backpropogation
        self.manual_backward(loss)
        if self.clip_norm > 0:  # 防止gradient爆炸
            total_norm = nn.utils.clip_grad.clip_grad_norm_(self.model.parameters(), self.clip_norm)  # self.model需要修改

            ## multimodal
            '''
            def get_trainable_parameters(self):
                if hasattr(self, "structure_encoder"):
                    return list(self.structure_encoder.parameters()) + \
                        list(self.text_encoder.parameters()) + \
                        list(self.fusion_module.parameters()) + \
                        list(self.predictor.parameters())
                else:
                    return self.model.parameters()

            total_norm = nn.utils.clip_grad.clip_grad_norm_(self.get_trainable_parameters(), self.clip_norm)
            '''

            self.log('train/total_norm', total_norm, on_step=False, on_epoch=True, \
                prog_bar=False, logger=True, batch_size=bsz)

        if self.clip_grad > 0:
            nn.utils.clip_grad.clip_grad_value_(self.model.parameters(), self.clip_grad)
            '''
            nn.utils.clip_grad.clip_grad_value_(self.get_trainable_parameters(), self.clip_grad)
            '''
        opt.step()  # optimizer update

        # swa
        swa_enabled = self.swa_epochs+self.current_epoch >= self.params.n_epochs
        if swa_enabled:
            self.swa_model.update_parameters(self.model)  # 暂时先不用swa 要用需要改self.model

        # sceduler for learning rate
        sch = self.lr_schedulers()
        if sch is not None and not swa_enabled:
            sch.step()
        
        # loss记录在log中
        output = {'loss': loss}
        #* note: make sure disabling on_step logging, which may frequently
        #* cause unexpected pauses due to the logging IO when the disc is on a NAS.
        self.log('train/loss', loss, on_step=False, on_epoch=True, \
            prog_bar=False, logger=True, batch_size=bsz)  # 可以考虑细分出来更多的loss
        return output

    # update param of BN in swa model
    ## swa相关的暂时都不管
    def on_train_end(self):
        print("Updating BNs for Stochastic Weight Averaging")
        device = self.swa_model.parameters().__next__().device
        torch.optim.swa_utils.update_bn(self.train_loader, self.swa_model, device)

    # validation for single batch
    def validation_step(self, batch, batch_idx):
        output = self.forward(batch)   # multimodal的forward return 3 parts

        ### multimodal
        '''
        if hasattr(self, "fusion_module"):
            fusion_out, _, _ = self.forward(batch)
            output = fusion_out
        else:
            output = self.forward(batch)

        '''

        loss = regression_loss(output, batch, self.targets, self.target_std, self.target_mean, self.params.loss_func)
        # 这里的loss就是validation部分测试mae等等的loss，只有training部分的loss需要改变

        out = {
            'val/loss': loss, 
            'output': output.detach().cpu(),
        }

        for i, t in enumerate(self.targets):
            labels = batch[t]
            out[t] = abs(output[:, i]*self.target_std[i]+self.target_mean[i] - labels).detach().cpu()  # inverse normalization

        self.validation_step_outputs.append(out)
        return out

    # 完成validation之后的callback
    def on_validation_epoch_end(self):
        outputs = self.validation_step_outputs
        avg_loss = torch.cat([x['val/loss'] for x in outputs]).mean()  # 取batch的avg loss as val loss
        
        print(f'\r\rval loss: {avg_loss:.3f} ', end='')
        logging_key = self.logging_key if self.logging_key is not None else 'val'  # loggingkey是swa的
        self.log(f'{logging_key}/loss', avg_loss)

        # 按target dim record mae
        for t in self.targets:
            v = torch.cat([x[t] for x in outputs], dim=0).mean()
            self.log(f'{logging_key}/' + t, v)
            print(f'{t}: {v.item():.3f} ', end='')
        print('   ')

        if self.logging_key is None:
            self.val_history.append(avg_loss.item())
            v = numpy.array(self.val_history, dtype=numpy.float64)
            K = 50  # mean filter width
            T = 10  # mean of top-T scores
            k = numpy.ones(K, dtype=numpy.float64)
            m = numpy.convolve(v, k)[:-(K-1)] / numpy.convolve(numpy.ones_like(v), k)[:-(K-1)]
            r = numpy.sort(v)[:min(len(v),T)]
            self.log("hp/avr50", m[-1])  # 50epoch avg
            self.log("hp/min_avr50", m[~numpy.isnan(m)].min())  # min 50avg, for early stop
            self.log("hp/min", v[~numpy.isnan(v)].min())
            self.log("hp/mean_min10", r.mean())
            self.log("hp/val", avg_loss)
            ## 是crustalformer核心之一 轻量训练 + 快速收敛曲线统计

        self.validation_step_outputs.clear()  # avoid oom

    def test_step(self, batch, batch_idx):
        return self.validation_step(batch, batch_idx)  # test时自动调用
        
    def on_test_epoch_end(self):
        outputs = self.validation_step_outputs
        avg_loss = torch.cat([x['val/loss'] for x in outputs]).mean()
        
        logging_key = self.logging_key if self.logging_key is not None else 'test'
        print(f'\r\rtest loss: {avg_loss:.3f} ', end='')
        self.log(f'{logging_key}/loss', avg_loss)
        for t in self.targets:
            v = torch.cat([x[t] for x in outputs], dim=0).mean()
            key = f'{logging_key}/' + t
            self.log(key, v)
            print(f'{t}: {v.item():.3f} ', end='')
        self.validation_step_outputs.clear()

    def configure_optimizers(self):
        #* opt may be re-initialized if the initial lr is different from lr,
        #* ie, the lr lambda f does not satisfy f(0) = lr.
        lr = self.params.lr
        opt = getattr(self.params, 'optimizer', 'adam')
        weight_decay = getattr(self.params, 'weight_decay', 0.0)

        if weight_decay <= 0:  # no weight decay
            opt_params = [{
                'params': self.model.parameters(),

                '''
                'params': delf.get_trainable_params(),
                '''
            }]
        else:  # weight decay, seperate params into two groups
            nodecay = []
            decay = []
            for m in self.model.modules():

            ### multimodal
            # 新def的get_trainable_modules()来得到所有trainable param
            # 最后和get_trainable_params()一起放在外面
            '''
            def get_trainable_modules(self):
                if hasattr(self, "fusion_modeule"):
                    return [self.structure_encoder, self.fusion_module, self.predictor]
                else:
                    return [self.model]

            for m in self.get_trainable_modules():
            '''

                if isinstance(m, (torch.nn.BatchNorm1d, torch.nn.LayerNorm, torch.nn.SyncBatchNorm)):
                    nodecay.extend(m.parameters(False))
                else:
                    for name, param in m.named_parameters(recurse=False):
                        if "bias" in name:
                            nodecay.append(param)
                        else:
                            decay.append(param)
            opt_params = [
                {'params': nodecay},
                {'params': decay, 'weight_decay': weight_decay},
            ]
            num_nodecay = sum([p.numel() for p in nodecay])
            num_decay = sum([p.numel() for p in decay])
            num_total = sum([p.numel() for p in self.model.parameters()])
            print(f"# nodecay params = {num_nodecay}")
            print(f"# decay params = {num_decay}")
            print(f"# params = {num_total}")
            assert num_decay + num_nodecay == num_total

        opt_args = {
            'lr': lr
        }

        # choosing optimizer
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
            return NotImplementedError(f'Unknown optimizer: {self.params.optimizer}')
        sch = None
        
        if self.params.lr_sch == "const":  # constant lr
            return opt(opt_params, **opt_args)
        elif self.params.lr_sch == "inverse_sqrt_nowarmup":  # no warmup, decay along time
            #* used in the t-fixup paper
            decay = self.params.sch_params[0]
            f = lambda t: (decay / (decay + t))**0.5
            opt = opt(opt_params, **opt_args)
            sch = lr_scheduler.LambdaLR(opt, f)
        elif self.params.lr_sch == "inverse_sqrt_nowarmup_dmodel":  # add model dimension scaling
            decay = self.params.sch_params[0]
            f = lambda t: self.params.model_dim**-0.5*(decay / (decay + t))**0.5
            opt_args['lr'] = lr*f(0)
            opt = opt(opt_params, **opt_args)
            sch = lr_scheduler.LambdaLR(opt, lambda t: f(t)/f(0))
        elif self.params.lr_sch == "inverse_sqrt_warmup":
            #* used in the original transformer paper
            warmup_steps = self.params.sch_params[0]
            f = lambda t: self.params.model_dim**-0.5*min((t+1)**-0.5, (t+1)*warmup_steps**-1.5)
            opt_args['lr'] = f(0)
            opt = opt(opt_params, **opt_args)
            sch = lr_scheduler.LambdaLR(opt, lambda t: f(t)/f(0))
        elif self.params.lr_sch == "inverse_sqrt_warmup_lrmax":  # ensure max lr
            warmup_steps = self.params.sch_params[0]
            f = lambda t: warmup_steps**0.5*min((t+1)**-0.5, (t+1)*warmup_steps**-1.5)
            opt_args['lr'] = lr*f(0)
            opt = opt(opt_params, **opt_args)
            sch = lr_scheduler.LambdaLR(opt, lambda t: f(t)/f(0))
        elif self.params.lr_sch == "multistep":  # reduce lr at specific epochs
            opt = opt(opt_params, **opt_args)
            sch = lr_scheduler.MultiStepLR(opt, milestones=self.params.sch_params, gamma=0.1)
        else:
            return NotImplementedError(f'Unknown lr_sch: {self.params.lr_sch}')
        
        return [[opt], [sch]]

    # unimodal forward
    def forward(self, x):
        if self.use_average_model and self.swa_model is not None:
            return self.swa_model(x)
        return self.model(x)
    
    ### multimodal forward function 兼容unimodal和multimodal
    '''
    def forward(self, batch):
        if hasattr(self, 'fusion_module'):  # if multimodal
            struct_out = self.structure_encoder(batch)
            text_out = self.text_encoder(batch.text)  # ?
            fusion_out = self.fusion_module(struct_out, text_out)
            return self.predictor(fusion_out), struct_out, text_out
        else:
            out = self.model(batch)
            return out, None, None

            ### mask机制应该是在这儿改
            ### 然后param中看怎么加上对应的param
    '''

    def train_dataloader(self):
        return self.train_loader

    def val_dataloader(self):
        return self.val_loader
