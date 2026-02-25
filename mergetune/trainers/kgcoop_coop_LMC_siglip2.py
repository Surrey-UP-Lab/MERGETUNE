import os.path as osp
import json
import time
import datetime
import itertools
from pathlib import Path
from tqdm import tqdm

import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.cuda.amp import GradScaler, autocast
from collections import OrderedDict

from dassl.engine import TRAINER_REGISTRY, TrainerX
from dassl.metrics import compute_accuracy
from dassl.utils import load_pretrained_weights, load_checkpoint
from dassl.optim import build_optimizer, build_lr_scheduler

# Import for SigLIP
from transformers import AutoModel, AutoProcessor, AutoTokenizer

# Fallback to CLIP if needed
try:
    from clip import clip
    from clip.simple_tokenizer import SimpleTokenizer as _Tokenizer
    _tokenizer = _Tokenizer()
except:
    _tokenizer = None

import numpy as np

# For Optuna (optional)
try:
    import optuna
    OPTUNA_AVAILABLE = True
except ImportError:
    OPTUNA_AVAILABLE = False


class SigLIPWrapper(nn.Module):
    """Wrapper to make SigLIP compatible with CLIP-style interface"""
    def __init__(self, siglip_model, processor):
        super().__init__()
        self.model = siglip_model
        self.processor = processor
        
        # Add compatibility attributes for CLIP-style access
        self.visual = siglip_model.vision_model
        self.token_embedding = siglip_model.text_model.embeddings.token_embedding
        self.transformer = siglip_model.text_model.encoder
        self.positional_embedding = siglip_model.text_model.embeddings.position_embedding.weight
        self.ln_final = siglip_model.text_model.final_layer_norm
        
        # Handle text projection
        if hasattr(siglip_model, 'text_projection'):
            self.text_projection = siglip_model.text_projection
        elif hasattr(siglip_model.text_model, 'text_projection'):
            self.text_projection = siglip_model.text_model.text_projection
        else:
            print("Warning: No text_projection found in SigLIP model, using Identity")
            self.text_projection = nn.Identity()
        
        self.logit_scale = nn.Parameter(torch.ones([]) * 4.6052)
        self._dtype = next(siglip_model.parameters()).dtype
    
    @property
    def dtype(self):
        return self._dtype
    
    def float(self):
        self.model.float()
        self._dtype = torch.float32
        return self
    
    def half(self):
        self.model.half()
        self._dtype = torch.float16
        return self
    
    def encode_text(self, text):
        """CLIP-style text encoding for compatibility"""
        return self.model.get_text_features(text)
    
    def forward(self, *args, **kwargs):
        return self.model(*args, **kwargs)


def load_siglip_to_cpu(cfg):
    """Load SigLIP model from Hugging Face"""
    backbone_name = cfg.MODEL.BACKBONE.NAME
    siglip_models = {
        "SigLIP-B/16": "google/siglip-base-patch16-224",
        "SigLIP-L/16": "google/siglip-large-patch16-256",
        "SigLIP-SO400M/14": "google/siglip-so400m-patch14-384",
        "SigLIP2-B/16": "google/siglip-base-patch16-224",  # Alias for SigLIP 2
        "SigLIP2-L/16": "google/siglip-large-patch16-256",  # SigLIP 2 Large
    }
    
    model_id = siglip_models.get(backbone_name, "google/siglip-base-patch16-224")
    print(f"Loading SigLIP model: {model_id}")
    
    siglip_model = AutoModel.from_pretrained(model_id)
    processor = AutoProcessor.from_pretrained(model_id)
    model = SigLIPWrapper(siglip_model, processor)
    return model


def load_clip_to_cpu(cfg):
    """Load CLIP or SigLIP model"""
    backbone_name = cfg.MODEL.BACKBONE.NAME
    
    if "SigLIP" in backbone_name or "siglip" in backbone_name.lower():
        return load_siglip_to_cpu(cfg)
    
    url = clip._MODELS[backbone_name]
    model_path = clip._download(url)

    try:
        model = torch.jit.load(model_path, map_location="cpu").eval()
        state_dict = None
    except RuntimeError:
        state_dict = torch.load(model_path, map_location="cpu")

    model = clip.build_model(state_dict or model.state_dict())
    return model


def tokenize_siglip(texts, processor, context_length=64):
    """Tokenize text for SigLIP models"""
    if isinstance(texts, str):
        texts = [texts]
    tokens = processor.tokenizer(
        texts,
        padding="max_length",
        max_length=context_length,
        truncation=True,
        return_tensors="pt"
    )
    return tokens.input_ids


def tokenize(texts, model=None):
    """Universal tokenizer"""
    if hasattr(model, 'processor'):
        return tokenize_siglip(texts, model.processor)
    else:
        return clip.tokenize(texts)


CUSTOM_TEMPLATES = {
    "OxfordPets": "a photo of a {}, a type of pet.",
    "OxfordFlowers": "a photo of a {}, a type of flower.",
    "FGVCAircraft": "a photo of a {}, a type of aircraft.",
    "DescribableTextures": "a photo of a {}, a type of texture.",
    "EuroSAT": "a centered satellite photo of {}.",
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
        
        # Check if this is a SigLIP model and get the head layer
        self.is_siglip = hasattr(clip_model, 'processor')
        if self.is_siglip:
            # SigLIP has a head layer that must be applied after pooling
            if hasattr(clip_model.model.text_model, 'head'):
                self.head = clip_model.model.text_model.head
            else:
                self.head = nn.Identity()
        else:
            self.head = None

    def forward(self, prompts, tokenized_prompts):
        if self.is_siglip:
            # SigLIP uses batch_first=True, so no permutation needed
            x = prompts + self.positional_embedding.type(self.dtype)
            if hasattr(self.transformer, 'forward'):
                x = self.transformer(x).last_hidden_state if hasattr(self.transformer(x), 'last_hidden_state') else self.transformer(x)
            else:
                x = self.transformer(x)
            x = self.ln_final(x).type(self.dtype)
            
            # SigLIP uses LAST position (sticky EOS tokenization), not argmax
            # x.shape = [batch_size, n_ctx, transformer.width]
            x = x[:, -1, :]  # Take features from last position
            
            # Apply head layer (crucial for SigLIP)
            x = self.head(x)
        else:
            # Original CLIP architecture
            x = prompts + self.positional_embedding.type(self.dtype)
            x = x.permute(1, 0, 2)  # NLD -> LND
            x = self.transformer(x)
            x = x.permute(1, 0, 2)  # LND -> NLD
            x = self.ln_final(x).type(self.dtype)
            
            # take features from the eot embedding (eot_token is the highest number in each sequence)
            x = x[torch.arange(x.shape[0]), tokenized_prompts.argmax(dim=-1)]
            
            # Apply text projection
            if isinstance(self.text_projection, nn.Identity):
                pass
            elif isinstance(self.text_projection, nn.Linear):
                x = self.text_projection(x)
            else:
                x = x @ self.text_projection

        return x


class PromptLearner(nn.Module):
    def __init__(self, cfg, classnames, clip_model):
        super().__init__()
        n_cls = len(classnames)
        n_ctx = cfg.TRAINER.COOP.N_CTX
        ctx_init = cfg.TRAINER.COOP.CTX_INIT
        dtype = clip_model.dtype
        ctx_dim = clip_model.ln_final.weight.shape[0]
        
        # Handle different image size attributes for SigLIP vs CLIP
        if hasattr(clip_model.visual, 'input_resolution'):
            clip_imsize = clip_model.visual.input_resolution
        elif hasattr(clip_model.visual, 'config'):
            clip_imsize = clip_model.visual.config.image_size
        else:
            clip_imsize = 224
        cfg_imsize = cfg.INPUT.SIZE[0]
        print(f"Model image size: {clip_imsize}, Config image size: {cfg_imsize}")

        if ctx_init:
            # use given words to initialize context vectors
            temp = 'a photo of a'
            ctx_init = temp.replace("_", " ")
            n_ctx = len(ctx_init.split(" "))
            prompt = tokenize(ctx_init, clip_model)
            with torch.no_grad():
                embedding = clip_model.token_embedding(prompt).type(dtype)
            
            ctx_vectors = embedding[0, 1 : 1 + n_ctx, :]
            prompt_prefix = ctx_init

        else:
            # random initialization
            if cfg.TRAINER.COOP.CSC:
                print("Initializing class-specific contexts")
                ctx_vectors = torch.empty(n_cls, n_ctx, ctx_dim, dtype=dtype)
            else:
                print("Initializing a generic context")
                ctx_vectors = torch.empty(n_ctx, ctx_dim, dtype=dtype)
            nn.init.normal_(ctx_vectors, std=0.02)
            prompt_prefix = " ".join(["X"] * n_ctx)


        print(f'Initial context: "{prompt_prefix}"')
        print(f"Number of context words (tokens): {n_ctx}")

        # self.ctx = nn.Parameter(ctx_vectors)  # to be optimized
        # fixed not optimized
        self.register_buffer("ctx", ctx_vectors)


        classnames = [name.replace("_", " ") for name in classnames]
        if _tokenizer is not None:
            name_lens = [len(_tokenizer.encode(name)) for name in classnames]
        else:
            name_lens = [len(name.split()) + 1 for name in classnames]
        prompts = [prompt_prefix + " " + name + "." for name in classnames]

        #print(f"Loading CLIP (backbone: {cfg.MODEL.BACKBONE.NAME})")
        clip_model_ = load_clip_to_cpu(cfg)
        clip_model_.cuda()
        
        #prompts_ = [prompt_prefix + " " + name + "." for name in classnames]        
        temp = CUSTOM_TEMPLATES[cfg.DATASET.NAME]
        prompts_ = [temp.format(c.replace("_", " ")) for c in classnames]
        print(f"Prompts: {prompts_}")
        prompts_ = torch.cat([tokenize(p, clip_model_) for p in prompts_])
        prompts_ = prompts_.cuda()

        with torch.no_grad():
            text_features = clip_model_.encode_text(prompts_)
            text_features = text_features / text_features.norm(dim=-1, keepdim=True)

        self.text_features = text_features


        tokenized_prompts = torch.cat([tokenize(p, clip_model) for p in prompts])
        with torch.no_grad():
            embedding = clip_model.token_embedding(tokenized_prompts).type(dtype)
        # These token vectors will be saved when in save_model(),
        # but they should be ignored in load_model() as we want to use
        # those computed using the current class names
        self.register_buffer("token_prefix", embedding[:, :1, :])  # SOS
        self.register_buffer("token_suffix", embedding[:, 1 + n_ctx :, :])  # CLS, EOS


        self.n_cls = n_cls
        self.n_ctx = n_ctx
        self.tokenized_prompts = tokenized_prompts  # torch.Tensor
        self.name_lens = name_lens
        self.class_token_position = cfg.TRAINER.COOP.CLASS_TOKEN_POSITION

    def forward(self):
        ctx = self.ctx

        if ctx.dim() == 2:
            ctx = ctx.unsqueeze(0).expand(self.n_cls, -1, -1) # torch.Size([100, 4, 512])
        
        prefix = self.token_prefix
        suffix = self.token_suffix

        prompts = torch.cat(
            [
                prefix,  # (n_cls, 1, dim)
                ctx,
                suffix,  # (n_cls, *, dim)
            ],
            dim=1,
        )

        return prompts


class PromptMidLearner(nn.Module):
    def __init__(self, cfg, classnames, clip_model):
        super().__init__()
        n_cls = len(classnames)
        n_ctx = cfg.TRAINER.COOP_CLIP.N_CTX
        ctx_init = cfg.TRAINER.COOP_CLIP.CTX_INIT
        dtype = clip_model.dtype
        ctx_dim = clip_model.ln_final.weight.shape[0]
        
        # Handle different image size attributes for SigLIP vs CLIP
        if hasattr(clip_model.visual, 'input_resolution'):
            clip_imsize = clip_model.visual.input_resolution
        elif hasattr(clip_model.visual, 'config'):
            clip_imsize = clip_model.visual.config.image_size
        else:
            clip_imsize = 224
        cfg_imsize = cfg.INPUT.SIZE[0]
        print(f"Model image size: {clip_imsize}, Config image size: {cfg_imsize}")

        if ctx_init:
            # use given words to initialize context vectors
            temp = 'a photo of a'
            ctx_init = temp.replace("_", " ")
            n_ctx = len(ctx_init.split(" "))
            prompt = tokenize(ctx_init, clip_model)
            with torch.no_grad():
                embedding = clip_model.token_embedding(prompt).type(dtype)
            
            ctx_vectors = embedding[0, 1 : 1 + n_ctx, :]
            prompt_prefix = ctx_init

        else:
            # random initialization
            if cfg.TRAINER.COOP.CSC:
                print("Initializing class-specific contexts")
                ctx_vectors = torch.empty(n_cls, n_ctx, ctx_dim, dtype=dtype)
            else:
                print("Initializing a generic context")
                ctx_vectors = torch.empty(n_ctx, ctx_dim, dtype=dtype)
            nn.init.normal_(ctx_vectors, std=0.02)
            prompt_prefix = " ".join(["X"] * n_ctx)


        print(f'Initial context: "{prompt_prefix}"')
        print(f"Number of context words (tokens): {n_ctx}")

        self.ctx = nn.Parameter(ctx_vectors)  # to be optimized

        # Add bias_vectors parameter (missing from original implementation)
        bias_vectors = torch.empty(1, 512, dtype=dtype)
        nn.init.normal_(bias_vectors, std=0.02)
        self.bias_vectors = nn.Parameter(bias_vectors)


        classnames = [name.replace("_", " ") for name in classnames]
        if _tokenizer is not None:
            name_lens = [len(_tokenizer.encode(name)) for name in classnames]
        else:
            name_lens = [len(name.split()) + 1 for name in classnames]
        prompts = [prompt_prefix + " " + name + "." for name in classnames]

        #print(f"Loading CLIP (backbone: {cfg.MODEL.BACKBONE.NAME})")
        clip_model_ = load_clip_to_cpu(cfg)
        clip_model_.cuda()
        
        #prompts_ = [prompt_prefix + " " + name + "." for name in classnames]        
        temp = CUSTOM_TEMPLATES[cfg.DATASET.NAME]
        prompts_ = [temp.format(c.replace("_", " ")) for c in classnames]
        print(f"Prompts: {prompts_}")
        prompts_ = torch.cat([tokenize(p, clip_model_) for p in prompts_])
        prompts_ = prompts_.cuda()

        with torch.no_grad():
            text_features = clip_model_.encode_text(prompts_)
            text_features = text_features / text_features.norm(dim=-1, keepdim=True)

        self.text_features = text_features

        tokenized_prompts = torch.cat([tokenize(p, clip_model) for p in prompts])
        with torch.no_grad():
            embedding = clip_model.token_embedding(tokenized_prompts).type(dtype)
        # These token vectors will be saved when in save_model(),
        # but they should be ignored in load_model() as we want to use
        # those computed using the current class names
        self.register_buffer("token_prefix", embedding[:, :1, :])  # SOS
        self.register_buffer("token_suffix", embedding[:, 1 + n_ctx :, :])  # CLS, EOS

        self.n_cls = n_cls
        self.n_ctx = n_ctx
        self.tokenized_prompts = tokenized_prompts  # torch.Tensor
        self.name_lens = name_lens
        self.class_token_position = cfg.TRAINER.COOP.CLASS_TOKEN_POSITION

    def forward(self):
        ctx = self.ctx

        if ctx.dim() == 2:
            ctx = ctx.unsqueeze(0).expand(self.n_cls, -1, -1) # torch.Size([100, 4, 512])
        
        prefix = self.token_prefix
        suffix = self.token_suffix


        if self.class_token_position == "end":
            prompts = torch.cat(
                [
                    prefix,  # (n_cls, 1, dim)
                    ctx,
                    suffix,  # (n_cls, *, dim)
                ],
                dim=1,
            )

        elif self.class_token_position == "middle":
            # ... same middle logic as PromptLearner ...
            half_n_ctx = self.n_ctx // 2
            prompts = []
            for i in range(self.n_cls):
                name_len = self.name_lens[i]
                prefix_i = prefix[i : i + 1, :, :]
                class_i = suffix[i : i + 1, :name_len, :]
                suffix_i = suffix[i : i + 1, name_len:, :]
                ctx_i_half1 = ctx[i : i + 1, :half_n_ctx, :]
                ctx_i_half2 = ctx[i : i + 1, half_n_ctx:, :]
                prompt = torch.cat(
                    [
                        prefix_i,
                        ctx_i_half1,
                        class_i,
                        ctx_i_half2,
                        suffix_i,
                    ],
                    dim=1,
                )
                prompts.append(prompt)
            prompts = torch.cat(prompts, dim=0)

        elif self.class_token_position == "front":
            # ... same front logic as PromptLearner ...
            prompts = []
            for i in range(self.n_cls):
                name_len = self.name_lens[i]
                prefix_i = prefix[i : i + 1, :, :]
                class_i = suffix[i : i + 1, :name_len, :]
                suffix_i = suffix[i : i + 1, name_len:, :]
                ctx_i = ctx[i : i + 1, :, :]
                prompt = torch.cat(
                    [
                        prefix_i,
                        class_i,
                        ctx_i,
                        suffix_i,
                    ],
                    dim=1,
                )
                prompts.append(prompt)
            prompts = torch.cat(prompts, dim=0)
        
        else:
            raise ValueError


        return prompts


class CustomCLIP(nn.Module):
    def __init__(self, cfg, classnames, clip_model):
        super().__init__()
        self.cfg = cfg
        self.prompt_learner = PromptLearner(cfg, classnames, clip_model) # fixed coop model
        self.prompt_mid_learner = PromptMidLearner(cfg, classnames, clip_model) # mid coop model, which is learned
        self.tokenized_prompts = self.prompt_learner.tokenized_prompts
        self.ori_embedding = self.prompt_learner.text_features
        self.image_encoder = clip_model.visual
        self.text_encoder = TextEncoder(clip_model)
        self.logit_scale = clip_model.logit_scale
        self.dtype = clip_model.dtype

    def forward(self, image, sign=None, text_features_mid=None):
        if sign is None:
            prompts = self.prompt_mid_learner()
            image_features = self.image_encoder(image.type(self.dtype))
            
            # Handle SigLIP output: extract pooled features from BaseModelOutputWithPooling
            if hasattr(image_features, 'pooler_output'):
                image_features = image_features.pooler_output
            elif hasattr(image_features, 'last_hidden_state'):
                image_features = image_features.last_hidden_state.mean(dim=1)

            tokenized_prompts = self.tokenized_prompts
            text_features = self.text_encoder(prompts, tokenized_prompts) 
            text_features_old = self.ori_embedding


            image_features = image_features / image_features.norm(dim=-1, keepdim=True)
            text_features = text_features / text_features.norm(dim=-1, keepdim=True)
            logit_scale = self.logit_scale.exp()

            logits = logit_scale * image_features @ text_features.t()
            text_features_old = text_features_old / text_features_old.norm(dim=-1, keepdim=True)

            # Calculate score based on the configured loss type
            if self.cfg.TRAINER.COOP.LOSS_TYPE == "cosine":
                # Original cosine similarity loss
                cos = torch.nn.CosineSimilarity(dim=1, eps=1e-07)
                score = cos(text_features, text_features_old)
                score = 1.0 - torch.mean(score)
            else:  # L2 loss
                # L2 distance between text features and original text features
                l2_dist = torch.norm(text_features - text_features_old, p=2, dim=1)
                score = torch.mean(l2_dist)

            return logits, score
        
        else:
            # prompts = self.prompt_mid_learner()
            image_features = self.image_encoder(image.type(self.dtype))
            
            # Handle SigLIP output: extract pooled features from BaseModelOutputWithPooling
            if hasattr(image_features, 'pooler_output'):
                image_features = image_features.pooler_output
            elif hasattr(image_features, 'last_hidden_state'):
                image_features = image_features.last_hidden_state.mean(dim=1)

            tokenized_prompts = self.tokenized_prompts
            text_features = text_features_mid
            
            image_features = image_features / image_features.norm(dim=-1, keepdim=True)
            text_features = text_features / text_features.norm(dim=-1, keepdim=True)
            logit_scale = self.logit_scale.exp()
            
            logits = logit_scale * image_features @ text_features.t()

            return logits
        

@TRAINER_REGISTRY.register()
class KgCoOp_COOP_LMC_SigLIP2(TrainerX):
    """KgCoOp with COOP LMC and SigLIP 2 Support"""

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

        print("Building custom CLIP")
        self.model = CustomCLIP(cfg, classnames, clip_model)
        self.w = cfg.TRAINER.COOP.W


        if self.cfg.RESUME_COOP and self.cfg.RESUME_COOP != 'None':
            print(f"Loading pretrained CoOp prompt_learner from {self.cfg.RESUME_COOP}")
            checkpoint_path_1 = osp.join(self.cfg.RESUME_COOP, "prompt_learner/model.pth.tar-100")
            checkpoint_path_2 = osp.join(self.cfg.RESUME_COOP, "prompt_learner/model-100.pth.tar")
            
            if osp.exists(checkpoint_path_1):
                checkpoint_path = checkpoint_path_1
            elif osp.exists(checkpoint_path_2):
                checkpoint_path = checkpoint_path_2
            else:
                raise FileNotFoundError(f"Neither {checkpoint_path_1} nor {checkpoint_path_2} exists")
            
            checkpoint = load_checkpoint(checkpoint_path)
            state_dict = checkpoint["state_dict"]
            epoch = checkpoint["epoch"]
            
            # Remove token information like in CoOp
            if "token_prefix" in state_dict:
                del state_dict["token_prefix"]
            if "token_suffix" in state_dict:
                del state_dict["token_suffix"]

            self.model.prompt_learner.load_state_dict(state_dict, strict=False)


            # CRITICAL FIX: Initialize prompt_mid_learner with the same weights as prompt_learner
            # This ensures that testing before training gives correct results
            print("Initializing prompt_mid_learner with loaded prompt_learner weights")
            self.model.prompt_mid_learner.ctx.data.copy_(self.model.prompt_learner.ctx.data)

            
        print("Turning off gradients in both the image and the text encoder, and the prompt_learner")
        for name, param in self.model.named_parameters():
            #if "prompt_learner" not in name: # and "adapter" not in name:
            if "prompt_mid_learner.ctx" not in name: 
                param.requires_grad_(False)
            else:
                print(name)

        # if cfg.MODEL.INIT_WEIGHTS:
        #     load_pretrained_weights(self.model.prompt_learner, cfg.MODEL.INIT_WEIGHTS)

        self.model.to(self.device)
        # NOTE: only give prompt_mid_learner to the optimizer
        self.optim = build_optimizer(self.model.prompt_mid_learner, cfg.OPTIM)
        self.sched = build_lr_scheduler(self.optim, cfg.OPTIM)
        self.register_model("prompt_mid_learner", self.model.prompt_mid_learner, self.optim, self.sched)

        self.scaler = GradScaler() if cfg.TRAINER.COOP.PREC == "amp" else None

        # Note that multi-gpu training could be slow because CLIP's size is
        # big, which slows down the copy operation in DataParallel
        # device_count = torch.cuda.device_count()
        # if device_count > 1:
        #     print(f"Multiple GPUs detected (n_gpus={device_count}), use all of them!")
        #     self.model = nn.DataParallel(self.model)

    def calculate_line_loss(self, feature_mid, image):
        logits = self.model(image, sign='LMC', text_features_mid=feature_mid)
        return logits



    def forward_backward(self, batch):
        image, label = self.parse_batch_train(batch)
        prec = self.cfg.TRAINER.COOP.PREC
        if prec == "amp":
            with autocast():
                output, score = self.model(image)
                loss_main_clip = F.cross_entropy(output, label) + self.w * score

                if self.cfg.TRAINER.COOP.COOP_LMC:
                    # print("COOP_LMC is True")

                    line_samples = np.arange(0.1, 1.01, 1.0 / float(self.cfg.TRAINER.COOP.NUM_SAMPLES))

                    # feature_start is coop weight, feature_end is learned weight
                    feature_start = self.model.text_encoder(self.model.prompt_learner(), self.model.tokenized_prompts) 
                    feature_end = self.model.text_encoder(self.model.prompt_mid_learner(), self.model.tokenized_prompts)
                    
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
            output,score = self.model(image)
            loss_main_clip = F.cross_entropy(output, label)+self.w*score

            if self.cfg.TRAINER.COOP.COOP_LMC:
                line_samples = np.arange(0.1, 1.01, 1.0 / float(self.cfg.TRAINER.COOP.NUM_SAMPLES))

                # feature_start is coop weight, feature_end is learned weight
                feature_start = self.model.text_encoder(self.model.prompt_learner(), self.model.tokenized_prompts) 
                feature_end = self.model.text_encoder(self.model.prompt_mid_learner(), self.model.tokenized_prompts)
                
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

            self.model_backward_and_update(loss)

        loss_summary = {
            "loss": loss.item(),
            "loss_main_clip": loss_main_clip.item(),
            "loss_clip": score.item(),
            "loss_LMC": total_loss_LMC.item() if self.cfg.TRAINER.COOP.COOP_LMC else 0.0,
            "weight_LMC": self.cfg.TRAINER.COOP.W_LMC,
            "weight_main_clip": self.w,
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
    
    def parse_batch_test(self, batch):
        input = batch["img"]
        label = batch["label"]
        input = input.to(self.device)
        label = label.to(self.device)
        return input, label


    def after_train(self):
        print("Finished training in KgCoOp_COOPLMC")

        do_test = not self.cfg.TEST.NO_TEST
        if do_test:
            if self.cfg.TEST.FINAL_MODEL == "best_val":
                print("Deploy the model with the best val performance")
                self.load_model(self.output_dir)
            self.test(split="val")

        # Show elapsed time
        elapsed = round(time.time() - self.time_start)
        elapsed = str(datetime.timedelta(seconds=elapsed))
        print("Elapsed: {}".format(elapsed))

        # Close writer
        self.close_writer()

    @torch.no_grad()
    def test(self, split=None):
        """A generic testing pipeline."""
        self.set_model_mode("eval")
        self.evaluator.reset()

        if split is None:
            split = self.cfg.TEST.SPLIT

        if split == "val" and self.val_loader is not None:
            data_loader = self.val_loader
            print("Do evaluation on {} set".format(split))
        else:
            data_loader = self.test_loader
            print("Do evaluation on test set")

        for batch_idx, batch in enumerate(tqdm(data_loader)):
            input, label = self.parse_batch_test(batch)
            output = self.model_inference(input)
            self.evaluator.process(output, label)

        results = self.evaluator.evaluate()

        for k, v in results.items():
            tag = "{}/{}".format(split, k)
            self.write_scalar(tag, v, self.epoch)

        return list(results.values())[0]

    def model_inference(self, input):
        return self.model(input)[0]


    def load_model(self, directory, epoch=None):
        if not directory:
            print("Note that load_model() is skipped as no pretrained model is given")
            return

        names = self.get_model_names()
        print(names)

        # By default, the best model is loaded
        model_file = "model-best.pth.tar"

        if epoch is not None:
            # model_file = "model.pth.tar-" + str(epoch)
            # Try both naming patterns
            model_file = "model.pth.tar-" + str(epoch)  # model.pth.tar-100

        for name in names:
            # model_path = osp.join(directory, name, model_file)
            # Check which file exists
            model_path = osp.join(directory, name, model_file)

            if osp.exists(model_path):
                model_path = model_path
            else:
                raise FileNotFoundError(f'Model not found at "{model_path}"')


            checkpoint = load_checkpoint(model_path)
            state_dict = checkpoint["state_dict"]
            epoch = checkpoint["epoch"]

            # Ignore fixed token vectors
            if "token_prefix" in state_dict:
                del state_dict["token_prefix"]

            if "token_suffix" in state_dict:
                del state_dict["token_suffix"]

            if "token_midfix" in state_dict:
                del state_dict["token_midfix"]

            print("Loading weights to {} " 'from "{}" (epoch = {})'.format(name, model_path, epoch))
            # set strict=False
            self._models[name].load_state_dict(state_dict, strict=False)


    def load_model_loop(self, directory, epoch=None):
        if not directory:
            print("Note that load_model() is skipped as no pretrained model is given")
            return

        names = self.get_model_names()
        model_file = f"model.pth.tar-{epoch}" if epoch is not None else None

        for name in names:
            if model_file:
                model_path = osp.join(directory, name, model_file)
                if not osp.exists(model_path):
                    raise FileNotFoundError('Model not found at "{}"'.format(model_path))

                checkpoint = load_checkpoint(model_path)
                state_dict = checkpoint["state_dict"]
                epoch = checkpoint["epoch"]

                # Ignore fixed token vectors
                if "token_prefix" in state_dict:
                    del state_dict["token_prefix"]

                if "token_suffix" in state_dict:
                    del state_dict["token_suffix"]

                if "token_midfix" in state_dict:
                    del state_dict["token_midfix"]

                # Load model state dict
                print("Loading weights to {} from '{}' (epoch = {})".format(name, model_path, epoch))
                self._models[name].load_state_dict(state_dict, strict=False)




    def load_model_merge(self, directory, lambda_val=None, model_prefix=None):
        if not directory:
            print("Note that load_model_merge() is skipped as no pretrained model is given")
            return

        names = self.get_model_names()
        
        for name in names:
            # Determine which naming pattern to use
            model_files = []
            detected_prefix = None
            
            if osp.exists(directory):
                import glob
                
                if model_prefix:
                    # Use explicitly specified prefix
                    pattern = osp.join(directory, f"{model_prefix}_*.pth")
                    model_files = glob.glob(pattern)
                    detected_prefix = model_prefix
                    if model_files:
                        print(f"Using specified prefix: Found {len(model_files)} model files with pattern {model_prefix}_*.pth")
                    else:
                        print(f"No model files found with specified prefix {model_prefix}_*.pth")
                else:
                    # Auto-detect available pattern
                    # Try clip_coop_ties_lambda pattern first
                    pattern1 = osp.join(directory, f"clip_coop_ties_lambda_*.pth")
                    files1 = glob.glob(pattern1)
                    
                    # Try clip_kgcoop_ties_lambda pattern
                    pattern2 = osp.join(directory, f"clip_kgcoop_ties_lambda_*.pth")
                    files2 = glob.glob(pattern2)
                    
                    if files1:
                        model_files = files1
                        detected_prefix = "clip_coop_ties_lambda"
                        print(f"Auto-detected: Found {len(files1)} model files with pattern clip_coop_ties_lambda_*.pth")
                    elif files2:
                        model_files = files2
                        detected_prefix = "clip_kgcoop_ties_lambda"
                        print(f"Auto-detected: Found {len(files2)} model files with pattern clip_kgcoop_ties_lambda_*.pth")
            
            if not model_files:
                if model_prefix:
                    print(f"No model files found in {directory} with pattern {model_prefix}_*.pth")
                else:
                    print(f"No model files found in {directory} with either pattern:")
                    print(f"  - clip_coop_ties_lambda_*.pth")
                    print(f"  - clip_kgcoop_ties_lambda_*.pth")
                continue
                
            # Select model based on lambda parameter
            if lambda_val is not None:
                # Look for specific lambda value with detected prefix
                target_file = osp.join(directory, f"{detected_prefix}_{lambda_val}.pth")
                if osp.exists(target_file):
                    model_path = target_file
                    print(f"Using specified lambda model: {osp.basename(model_path)}")
                else:
                    print(f"Specified lambda model {detected_prefix}_{lambda_val}.pth not found")
                    # Fall back to first available model
                    model_files.sort()
                    model_path = model_files[0]
                    print(f"Falling back to: {osp.basename(model_path)}")
            else:
                # Sort model files and use the first one (lowest lambda first)
                model_files.sort()
                model_path = model_files[0]
                print(f"Found {len(model_files)} model files, using: {osp.basename(model_path)}")
            
            if not osp.exists(model_path):
                raise FileNotFoundError('Model not found at "{}"'.format(model_path))

            checkpoint = load_checkpoint(model_path)
            state_dict = checkpoint["state_dict"]
            epoch = checkpoint.get("epoch", "unknown")

            # Ignore fixed token vectors
            if "token_prefix" in state_dict:
                del state_dict["token_prefix"]

            if "token_suffix" in state_dict:
                del state_dict["token_suffix"]

            if "token_midfix" in state_dict:
                del state_dict["token_midfix"]

            # Load model state dict
            print("Loading weights to {} from '{}' (epoch = {})".format(name, model_path, epoch))
            self._models[name].load_state_dict(state_dict, strict=False)


    def load_model_merge_dare(self, directory, model_name=None):
        if not directory:
            print("Note that load_model() is skipped as no pretrained model is given")
            return

        names = self.get_model_names()
        print(names)

        # By default, the best model is loaded

        for name in names:
            model_path = osp.join(directory, model_name)
            # Check which file exists

            if not osp.exists(model_path):
                raise FileNotFoundError(f'Model not found at "{model_path}"')

            checkpoint = load_checkpoint(model_path)
            state_dict = checkpoint["state_dict"]
            # epoch = checkpoint["epoch"]

            # Ignore fixed token vectors
            if "token_prefix" in state_dict:
                del state_dict["token_prefix"]

            if "token_suffix" in state_dict:
                del state_dict["token_suffix"]

            if "token_midfix" in state_dict:
                del state_dict["token_midfix"]

            # CRITICAL FIX: Rename parameters to match PromptMidLearner naming
            # DARE models have 'prompts.ctx' but PromptMidLearner expects 'ctx'
            renamed_state_dict = {}
            for key, value in state_dict.items():
                if key == "prompts.ctx":
                    renamed_state_dict["ctx"] = value
                    print(f"  Renamed parameter: {key} -> ctx")
                elif key == "prompts.bias_vectors":
                    renamed_state_dict["bias_vectors"] = value
                    print(f"  Renamed parameter: {key} -> bias_vectors")
                else:
                    # Keep all other parameters with original names
                    renamed_state_dict[key] = value

            print("Loading weights to {} " 'from "{}"'.format(name, model_path))
            # set strict=False
            self._models[name].load_state_dict(renamed_state_dict, strict=False)
