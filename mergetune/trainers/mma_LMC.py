import os.path as osp

import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.cuda.amp import GradScaler, autocast
from collections import OrderedDict

from dassl.engine import TRAINER_REGISTRY, TrainerX
from dassl.metrics import compute_accuracy
from dassl.utils import load_pretrained_weights, load_checkpoint
from dassl.optim import build_optimizer, build_lr_scheduler

from clip import clip
from clip.simple_tokenizer import SimpleTokenizer as _Tokenizer
import numpy as np
import tqdm
from .zsclip import ZeroshotCLIP

_tokenizer = _Tokenizer()


def load_clip_to_cpu(cfg):
    backbone_name = cfg.MODEL.BACKBONE.NAME
    url = clip._MODELS[backbone_name]
    model_path = clip._download(url)

    try:
        # loading JIT archive
        model = torch.jit.load(model_path, map_location="cpu").eval()
        state_dict = None

    except RuntimeError:
        state_dict = torch.load(model_path, map_location="cpu")

    model = clip.build_model(state_dict or model.state_dict())

    return model


CUSTOM_TEMPLATES = {
    "OxfordPets": "a photo of a {}, a type of pet.",
    "OxfordFlowers": "a photo of a {}, a type of flower.",
    "FGVCAircraft": "a photo of a {}, a type of aircraft.",
    "DescribableTextures": "a photo of a {}, a type of texture.",
    "EuroSAT": "a centered satellite photo of {}.",
    #"EuroSAT": "a photo of a {}.",
    "StanfordCars": "a photo of a {}.",
    "Food101": "a photo of {}, a type of food.",
    "SUN397": "a photo of a {}.",
    "Caltech101": "a photo of a {}.",
    "UCF101": "a photo of a person doing {}.",
    "ImageNet": "a photo of a {}.",
    "ImageNetSketch": "a photo of a {}.",
    "ImageNetV2": "a photo of a {}.",
    "ImageNetA": "a photo of a {}.",
    "ImageNetR": "a photo of a {}.",
}




class TextEncoder(nn.Module):
    def __init__(self, clip_model):
        super().__init__()
        self.transformer = clip_model.transformer
        self.positional_embedding = clip_model.positional_embedding
        self.ln_final = clip_model.ln_final
        self.text_projection = clip_model.text_projection
        self.dtype = clip_model.dtype

    def forward(self, prompts, tokenized_prompts, retrun_adapater_func=None):
        x = prompts + self.positional_embedding.type(self.dtype)
        x = x.permute(1, 0, 2)  # NLD -> LND
        if retrun_adapater_func == None:
            x = self.transformer(x)
        else:
            x = self.transformer([x, retrun_adapater_func])
        x = x.permute(1, 0, 2)  # LND -> NLD
        x = self.ln_final(x).type(self.dtype)
        # x.shape = [batch_size, n_ctx, transformer.width]
        # take features from the eot embedding (eot_token is the highest number in each sequence)
        x = x[torch.arange(x.shape[0]), tokenized_prompts.argmax(dim=-1)] @ self.text_projection
        return x


class AdapterLearner(nn.Module):
    def __init__(self, cfg, classnames, clip_model):
        super().__init__()

        self.n_cls = len(classnames)
        clip_imsize = clip_model.visual.input_resolution
        cfg_imsize = cfg.INPUT.SIZE[0]
        assert cfg_imsize == clip_imsize, f"cfg_imsize ({cfg_imsize}) must equal to clip_imsize ({clip_imsize})"

        self._build_text_embedding(cfg, classnames, clip_model)

        # build multi-modal adapter
        self.text_adapter_func = lambda x: self.return_text_adapter(index=x)
        self.text_adapter = self._build_adapter(
            clip_model.ln_final.weight.shape[0], 
            len(clip_model.transformer.resblocks), 
            cfg.TRAINER.MMADAPTER.ADAPTER_START,
            cfg.TRAINER.MMADAPTER.ADAPTER_END,
            cfg.TRAINER.MMADAPTER.ADAPTER_DIM,
            clip_model.dtype
        )
        
        self.visual_adapter_func = lambda x: self.return_visual_adapter(index=x)
        self.visual_adapter = self._build_adapter(
            clip_model.visual.ln_post.weight.shape[0],
            len(clip_model.visual.transformer.resblocks), 
            cfg.TRAINER.MMADAPTER.ADAPTER_START,
            cfg.TRAINER.MMADAPTER.ADAPTER_END,
            cfg.TRAINER.MMADAPTER.ADAPTER_DIM,
            clip_model.dtype
        )

        self.shared_adapter = self._build_adapter(
            cfg.TRAINER.MMADAPTER.ADAPTER_DIM,
            len(clip_model.visual.transformer.resblocks), 
            cfg.TRAINER.MMADAPTER.ADAPTER_START,
            cfg.TRAINER.MMADAPTER.ADAPTER_END,
            cfg.TRAINER.MMADAPTER.ADAPTER_DIM,
            clip_model.dtype
        )
        self.adapter_scale = float(cfg.TRAINER.MMADAPTER.ADAPTER_SCALE)

    def return_text_adapter(self, index):
        return self.text_adapter[index], self.shared_adapter[index], self.adapter_scale

    def return_visual_adapter(self, index):
        return self.visual_adapter[index], self.shared_adapter[index], self.adapter_scale


    def _build_text_embedding(self, cfg, classnames, clip_model):
        dtype = clip_model.dtype
        text_ctx_init = cfg.TRAINER.MMADAPTER.TEXT_CTX_INIT

        classnames = [name.replace("_", " ") for name in classnames]
        prompts = [text_ctx_init + " " + name + "." for name in classnames]
        tokenized_prompts = torch.cat([clip.tokenize(p) for p in prompts])

        with torch.no_grad():
            embedding = clip_model.token_embedding(tokenized_prompts).type(dtype)

        self.register_buffer("token_embedding", embedding)
        self.register_buffer("tokenized_prompts", tokenized_prompts)


    def _build_adapter(self, d_model, n_layers, l_start, l_end, mid_dim, dtype):

        adapter = [None] * (n_layers + 1)
        for i in range(l_start, l_end+1):
            if mid_dim == d_model:
                adapter[i] = nn.Sequential(
                    nn.Linear(d_model, mid_dim),
                    nn.ReLU()
                )
            else:
                adapter[i] = nn.Sequential(OrderedDict([
                    ("down", nn.Sequential(nn.Linear(d_model, mid_dim), nn.ReLU())),
                    ("up", nn.Linear(mid_dim, d_model))
                ]))
        adapter = nn.ModuleList([a for a in adapter])
        for m in adapter.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                nn.init.constant_(m.bias, 0)

        if dtype == torch.float16:
            for m in adapter.modules():
                m.half()
    
        return adapter
    
    def forward(self):
        embedding = self.token_embedding
        if self.text_adapter[0] is not None:
            token_embedding = self.text_adapter[0].down(embedding)
            shared_adapter = self.shared_adapter[0]
            token_embedding = shared_adapter(token_embedding)
            token_embedding = self.text_adapter[0].up(token_embedding)
            embedding = embedding + self.adapter_scale * token_embedding
        return embedding, self.text_adapter_func, self.visual_adapter_func


class CustomCLIP(nn.Module):
    def __init__(self, cfg, classnames, clip_model):
        super().__init__()
        self.cfg = cfg
        # self.prompt_learner = PromptLearner(cfg, classnames, clip_model) # fixed coop model
        self.adapter_learner = AdapterLearner(cfg, classnames, clip_model)
        # self.prompt_mid_learner = PromptMidLearner(cfg, classnames, clip_model) # mid coop model, which is learned
        self.tokenized_prompts = self.adapter_learner.tokenized_prompts
        # self.ori_embedding = self.prompt_mid_learner.text_features
        self.image_encoder = clip_model.visual
        self.text_encoder = TextEncoder(clip_model)
        self.logit_scale = clip_model.logit_scale
        self.dtype = clip_model.dtype
        self.text_features_for_inference = None

    def encode_text(self, prompts, tokenized_prompts, text_adapter_func=None):
        if text_adapter_func is not None:
            text_features = self.text_encoder(
                prompts, tokenized_prompts, text_adapter_func
            )
        else:
            text_features = self.text_encoder(
                prompts, tokenized_prompts
            )
        return text_features
    
    def encode_image(self, image, visual_adapter_func=None):
        if visual_adapter_func is not None:
            image_features = self.image_encoder(
                [image.type(self.dtype), visual_adapter_func]
            )
        else:
            image_features = self.image_encoder(
                image.type(self.dtype)
            )
        return image_features

    def forward(self, image, sign=None, text_features_mid=None):
        if sign == 'MMA':  # main loss is from combined prompt_mid_learner
            token_embedding, text_adapter_func, visual_adapter_func = self.adapter_learner()
            tokenized_prompts = self.tokenized_prompts

            if self.adapter_learner.training:
                text_features = self.encode_text(
                    token_embedding, tokenized_prompts, text_adapter_func
                )
            else:
                if self.text_features_for_inference is None:
                    self.text_features_for_inference = self.encode_text(
                        token_embedding, tokenized_prompts, text_adapter_func
                    )   
                text_features = self.text_features_for_inference

            image_features = self.encode_image(image, visual_adapter_func)

            text_features = F.normalize(text_features, dim=-1)
            image_features = F.normalize(image_features, dim=-1)

            logit_scale = self.logit_scale.exp()
            logits = logit_scale * image_features @ text_features.t()

            return logits

        elif sign == 'LMC_MMA':
            # give the text_features_mid to the text_encoder
            token_embedding, text_adapter_func, visual_adapter_func = self.adapter_learner()
            tokenized_prompts = self.tokenized_prompts

            if self.adapter_learner.training:
                text_features = text_features_mid
            else:
                if self.text_features_for_inference is None:
                    self.text_features_for_inference = self.encode_text(
                        token_embedding, tokenized_prompts, text_adapter_func
                    )   
                text_features = self.text_features_for_inference

            image_features = self.encode_image(image, visual_adapter_func)

            text_features = F.normalize(text_features, dim=-1)
            image_features = F.normalize(image_features, dim=-1)

            logit_scale = self.logit_scale.exp()
            logits = logit_scale * image_features @ text_features.t()

            return logits

            

@TRAINER_REGISTRY.register()
class MMA_LMC(TrainerX):

    def check_cfg(self, cfg):
        assert cfg.TRAINER.COOP.PREC in ["fp16", "fp32", "amp"]

    def build_model(self):
        cfg = self.cfg
        classnames = self.dm.dataset.classnames

        print(f"Loading CLIP (backbone: {cfg.MODEL.BACKBONE.NAME})")
        clip_model = load_clip_to_cpu(cfg)
        
        if cfg.TRAINER.COOP.PREC == "fp32" or cfg.TRAINER.COOP.PREC == "amp":
            # CLIP's default precision is fp16
            clip_model.float()
        
        print("Building original CLIP")
        self.clip_orig_model = ZeroshotCLIP(cfg)

        print("Building custom CLIP")
        self.model = CustomCLIP(cfg, classnames, clip_model)
        # self.w = cfg.TRAINER.COOP.W

        if self.cfg.RESUME_COOP and self.cfg.RESUME_COOP != 'None':
            print(f"Loading pretrained MMAdapter adapter_learner from {self.cfg.RESUME_COOP}")
            
            # Find the maximum epoch checkpoint
            adapter_dir = osp.join(self.cfg.RESUME_COOP, "adapter_learner")
            if osp.exists(adapter_dir):
                import glob
                checkpoint_files = glob.glob(osp.join(adapter_dir, "model.pth.tar-*"))
                if checkpoint_files:
                    # Extract epoch numbers and find maximum
                    epochs = [int(f.split("-")[-1]) for f in checkpoint_files]
                    max_epoch = max(epochs)
                    checkpoint_path = osp.join(adapter_dir, f"model.pth.tar-{max_epoch}")
                    print(f"Found checkpoint at epoch {max_epoch}: {checkpoint_path}")
                else:
                    # Try model-best.pth.tar as fallback
                    checkpoint_path = osp.join(adapter_dir, "model-best.pth.tar")
                    if not osp.exists(checkpoint_path):
                        raise FileNotFoundError(f"No checkpoint files found in {adapter_dir}")
            else:
                raise FileNotFoundError(f"Adapter directory not found: {adapter_dir}")
            
            if not osp.exists(checkpoint_path):
                raise FileNotFoundError(f"Checkpoint does not exist: {checkpoint_path}")
            
            
            checkpoint = load_checkpoint(checkpoint_path)
            state_dict = checkpoint["state_dict"]
            epoch = checkpoint["epoch"]
            
            # Ignore fixed token vectors
            if "token_embedding" in state_dict:
                del state_dict["token_embedding"]
            if "tokenized_prompts" in state_dict:
                del state_dict["tokenized_prompts"]

            self.model.adapter_learner.load_state_dict(state_dict, strict=False)

        print("Turning off gradients in both the image and the text encoder")
        
        for name, param in self.model.named_parameters():
            if "text_adapter" not in name and "visual_adapter" not in name and "shared_adapter" not in name:
                param.requires_grad_(False)

        # Double check
        num_trainable_params = 0
        enabled = set()
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                enabled.add(name)
                num_trainable_params += param.data.nelement()
        print(f"Parameters to be updated: {enabled}")
        print(f"Number of trainable parameters: {num_trainable_params}")

        if cfg.MODEL.INIT_WEIGHTS:
            load_pretrained_weights(self.model.prompt_learner, cfg.MODEL.INIT_WEIGHTS)

        self.model.to(self.device)
        # NOTE: optimize all trainable parameters (prompt_mid_learner.ctx + adapters)
        self.optim = build_optimizer(self.model.adapter_learner, cfg.OPTIM)
        self.sched = build_lr_scheduler(self.optim, cfg.OPTIM)
        
        # Register both components for saving/loading
        self.register_model("adapter_learner", self.model.adapter_learner, self.optim, self.sched)

        self.scaler = GradScaler() if cfg.TRAINER.COOP.PREC == "amp" else None

        # device_count = torch.cuda.device_count()
        # if device_count > 1:
        #     print(f"Multiple GPUs detected (n_gpus={device_count}), use all of them!")
        #     self.model = nn.DataParallel(self.model)

    def calculate_line_loss(self, feature_mid, image):
        # text_features = self.model.text_encoder(feature_mid, self.model.tokenized_prompts)
        # self.model.prompt_mid_learner.ctx.data.copy_(feature_mid)
        logits = self.model(image, sign='LMC_MMA', text_features_mid=feature_mid)
        return logits



    def forward_backward(self, batch):
        image, label = self.parse_batch_train(batch)
        prec = self.cfg.TRAINER.COOP.PREC
        if prec == "amp":
            with autocast():
                output = self.model(image, sign='MMA')
                loss_main_clip = F.cross_entropy(output, label)

                if self.cfg.TRAINER.COOP.COOP_LMC:
                    line_samples = np.arange(0.1, 1.01, 1.0 / float(self.cfg.TRAINER.COOP.NUM_SAMPLES))

                    # feature_start is MMA adapter weight generated text_features, feature_end is clip generated text_features
                    token_embedding, text_adapter_func, visual_adapter_func = self.model.adapter_learner()
                    tokenized_prompts = self.model.tokenized_prompts

                    if self.model.adapter_learner.training:
                        text_features = self.model.encode_text(
                            token_embedding, tokenized_prompts, text_adapter_func
                        )
                    else:
                        if self.model.text_features_for_inference is None:
                            self.model.text_features_for_inference = self.model.encode_text(
                                token_embedding, tokenized_prompts, text_adapter_func
                            )
                        text_features = self.model.text_features_for_inference

                    feature_start = text_features
                    feature_end = self.clip_orig_model.text_features
                    
                    total_loss_LMC = 0
                    for t in line_samples:
                        feature_mid = (feature_start + (feature_end - feature_start) * t)
                        output_LMC = self.calculate_line_loss(feature_mid, image)
                        loss_LMC = nn.CrossEntropyLoss()(output_LMC, label)
                        total_loss_LMC += loss_LMC / len(line_samples)

                    loss = loss_main_clip + self.cfg.TRAINER.COOP.W_LMC * total_loss_LMC
                else:
                    print("COOP_LMC is False")
                    loss = loss_main_clip

            self.optim.zero_grad()
            self.scaler.scale(loss).backward()
            self.scaler.step(self.optim)
            self.scaler.update()
        else:
            output = self.model(image, sign='MMA')
            loss_main_clip = F.cross_entropy(output, label)

            if self.cfg.TRAINER.COOP.COOP_LMC:
                line_samples = np.arange(0.1, 1.01, 1.0 / float(self.cfg.TRAINER.COOP.NUM_SAMPLES))

                # feature_start is MMA adapter weight generated text_features, feature_end is clip generated text_features
                token_embedding, text_adapter_func, visual_adapter_func = self.model.adapter_learner()
                tokenized_prompts = self.model.tokenized_prompts

                if self.model.adapter_learner.training:
                    text_features = self.model.encode_text(
                        token_embedding, tokenized_prompts, text_adapter_func
                    )
                else:
                    if self.model.text_features_for_inference is None:
                        self.model.text_features_for_inference = self.model.encode_text(
                            token_embedding, tokenized_prompts, text_adapter_func
                        )
                    text_features = self.model.text_features_for_inference

                feature_start = text_features
                feature_end = self.clip_orig_model.text_features
                
                total_loss_LMC = 0
                for t in line_samples:
                    feature_mid = (feature_start + (feature_end - feature_start) * t)
                    output_LMC = self.calculate_line_loss(feature_mid, image)
                    loss_LMC = nn.CrossEntropyLoss()(output_LMC, label)
                    total_loss_LMC += loss_LMC / len(line_samples)

                loss = loss_main_clip + self.cfg.TRAINER.COOP.W_LMC * total_loss_LMC
            else:
                print("COOP_LMC is False")
                loss = loss_main_clip
                total_loss_LMC = 0

            self.model_backward_and_update(loss)

        loss_summary = {
            "loss": loss.item(),
            "loss_main_clip": loss_main_clip.item(),
            "loss_LMC": total_loss_LMC.item()*self.cfg.TRAINER.COOP.W_LMC,
            "weight_LMC": self.cfg.TRAINER.COOP.W_LMC,
            "acc": compute_accuracy(output, label)[0].item(),
        }

        if (self.batch_idx + 1) == self.num_batches:
            #self.update_lr()
            self.sched.step()
            #self.sched_.step()
        return loss_summary

    def parse_batch_train(self, batch):
        input = batch["img"]
        label = batch["label"]
        input = input.to(self.device)
        label = label.to(self.device)
        return input, label

    def model_inference(self, input, sign=None):
        return self.model(input, sign=sign)


    @torch.no_grad()
    def test(self, split=None):
        """A generic testing pipeline."""
        self.set_model_mode("eval")
        self.evaluator.reset()
        # self.evaluator_MMA.reset()

        if split is None:
            split = self.cfg.TEST.SPLIT

        if split == "val" and self.val_loader is not None:
            data_loader = self.val_loader
        else:
            split = "test"  # in case val_loader is None
            data_loader = self.test_loader

        print(f"Evaluate on the *{split}* set")

        for batch_idx, batch in enumerate(data_loader):
            input, label = self.parse_batch_test(batch)
            output = self.model_inference(input, sign='MMA')
            self.evaluator.process(output, label)



        results = self.evaluator.evaluate()
        # results_MMA = self.evaluator_MMA.evaluate()

        for k, v in results.items():
            tag = f"{split}/{k}"
            self.write_scalar(tag, v, self.epoch)

        # for k, v in results_MMA.items():
        #     tag = f"{split}/{k}"
        #     self.write_scalar(tag, v, self.epoch)

        # return list(results.values())[0], list(results_MMA.values())[0]
        return list(results.values())[0]




    def load_model(self, directory, epoch=None):

        if not directory:
            print("Note that load_model() is skipped as no pretrained model is given")
            return

        names = self.get_model_names()

        # By default, the best model is loaded
        model_file = "model-best.pth.tar"

        if epoch is not None:
            model_file = "model.pth.tar-" + str(epoch)

        for name in names:
            model_path = osp.join(directory, name, model_file)

            if not osp.exists(model_path):
                raise FileNotFoundError('Model not found at "{}"'.format(model_path))

            checkpoint = load_checkpoint(model_path)
            state_dict = checkpoint["state_dict"]
            epoch = checkpoint["epoch"]

            # Ignore fixed token vectors
            if "token_embedding" in state_dict:
                del state_dict["token_embedding"]
            if "tokenized_prompts" in state_dict:
                del state_dict["tokenized_prompts"]

            print("Loading weights to {} " 'from "{}" (epoch = {})'.format(name, model_path, epoch))
            # set strict=False
            self._models[name].load_state_dict(state_dict, strict=False)




    def load_model_loop(self, directory, epoch=None):
        if not directory:
            print("Note that load_model() is skipped as no pretrained model is given")
            return

        names = self.get_model_names()

        # By default, the best model is loaded
        model_file = "model-best.pth.tar"

        if epoch is not None:
            model_file = "model.pth.tar-" + str(epoch)

        for name in names:
            model_path = osp.join(directory, name, model_file)

            if not osp.exists(model_path):
                raise FileNotFoundError('Model not found at "{}"'.format(model_path))

            checkpoint = load_checkpoint(model_path)
            state_dict = checkpoint["state_dict"]
            epoch_loaded = checkpoint["epoch"]

            # Ignore fixed token vectors
            if "token_embedding" in state_dict:
                del state_dict["token_embedding"]
            if "tokenized_prompts" in state_dict:
                del state_dict["tokenized_prompts"]

            print("Loading weights to {} " 'from "{}" (epoch = {})'.format(name, model_path, epoch_loaded))
            # set strict=False
            self._models[name].load_state_dict(state_dict, strict=False)
            
            # Set model to eval mode
            self._models[name].eval()
            
            # Verify model state
            print(f"Model {name} state after loading: {self._models[name].training}")
            print(f"Model {name} parameters: {sum(p.numel() for p in self._models[name].parameters())}")
