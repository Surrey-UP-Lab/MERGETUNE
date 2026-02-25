import os.path as osp

import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.cuda.amp import GradScaler, autocast

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
        
        # Handle text projection - SigLIP models have it as a Linear layer or weight matrix
        if hasattr(siglip_model, 'text_projection'):
            self.text_projection = siglip_model.text_projection
        elif hasattr(siglip_model.text_model, 'text_projection'):
            self.text_projection = siglip_model.text_model.text_projection
        else:
            # If no projection found, use Identity (SigLIP might not need projection)
            print("Warning: No text_projection found in SigLIP model, using Identity")
            self.text_projection = nn.Identity()
        
        self.logit_scale = nn.Parameter(torch.ones([]) * 4.6052)  # SigLIP default
        
        # Set dtype - this works because we're adding it as a new attribute to our wrapper
        self._dtype = next(siglip_model.parameters()).dtype
    
    @property
    def dtype(self):
        """Return the dtype of the model parameters"""
        return self._dtype
    
    def float(self):
        """Convert model to float32"""
        self.model.float()
        self._dtype = torch.float32
        return self
    
    def half(self):
        """Convert model to float16"""
        self.model.half()
        self._dtype = torch.float16
        return self
    
    def forward(self, *args, **kwargs):
        return self.model(*args, **kwargs)


def load_siglip_to_cpu(cfg):
    """Load SigLIP model from Hugging Face"""
    backbone_name = cfg.MODEL.BACKBONE.NAME
    
    # Map backbone names to SigLIP model identifiers
    siglip_models = {
        "SigLIP-B/16": "google/siglip-base-patch16-224",
        "SigLIP-L/16": "google/siglip-large-patch16-256",
        "SigLIP-SO400M/14": "google/siglip-so400m-patch14-384",
        "SigLIP2-B/16": "google/siglip-base-patch16-224",  # Alias for SigLIP 2
        "SigLIP2-L/16": "google/siglip-large-patch16-256",  # SigLIP 2 Large
    }
    
    model_id = siglip_models.get(backbone_name, "google/siglip-base-patch16-224")
    print(f"Loading SigLIP model: {model_id}")
    
    # Load the model
    siglip_model = AutoModel.from_pretrained(model_id)
    processor = AutoProcessor.from_pretrained(model_id)
    
    # Wrap the model to add CLIP-compatible interface
    model = SigLIPWrapper(siglip_model, processor)
    
    return model


def load_clip_to_cpu(cfg):
    """Fallback to original CLIP loader"""
    backbone_name = cfg.MODEL.BACKBONE.NAME
    
    # Check if it's a SigLIP model
    if "SigLIP" in backbone_name or "siglip" in backbone_name.lower():
        return load_siglip_to_cpu(cfg)
    
    # Original CLIP loading
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


def tokenize_siglip(texts, processor, context_length=64):
    """Tokenize text for SigLIP models"""
    if isinstance(texts, str):
        texts = [texts]
    
    # Use the processor's tokenizer
    tokens = processor.tokenizer(
        texts,
        padding="max_length",
        max_length=context_length,
        truncation=True,
        return_tensors="pt"
    )
    return tokens.input_ids


def tokenize(texts, model=None):
    """Universal tokenizer that works with both CLIP and SigLIP"""
    if hasattr(model, 'processor'):
        # SigLIP model
        return tokenize_siglip(texts, model.processor)
    else:
        # CLIP model
        return clip.tokenize(texts)


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
            # For SigLIP transformer, check if it expects attention_mask
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
            clip_imsize = 224  # default
        
        cfg_imsize = cfg.INPUT.SIZE[0]
        print(f"Model image size: {clip_imsize}, Config image size: {cfg_imsize}")
        # assert cfg_imsize == clip_imsize, f"cfg_imsize ({cfg_imsize}) must equal to clip_imsize ({clip_imsize})"

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

        classnames = [name.replace("_", " ") for name in classnames]
        
        # Handle tokenizer for name length calculation
        if _tokenizer is not None:
            name_lens = [len(_tokenizer.encode(name)) for name in classnames]
        else:
            # Approximate for SigLIP
            name_lens = [len(name.split()) + 1 for name in classnames]
        
        prompts = [prompt_prefix + " " + name + "." for name in classnames]

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
            ctx = ctx.unsqueeze(0).expand(self.n_cls, -1, -1)

        prefix = self.token_prefix
        suffix = self.token_suffix

        if self.class_token_position == "end":
            prompts = torch.cat(
                [
                    prefix,  # (n_cls, 1, dim)
                    ctx,     # (n_cls, n_ctx, dim)
                    suffix,  # (n_cls, *, dim)
                ],
                dim=1,
            )

        elif self.class_token_position == "middle":
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
                        prefix_i,     # (1, 1, dim)
                        ctx_i_half1,  # (1, n_ctx//2, dim)
                        class_i,      # (1, name_len, dim)
                        ctx_i_half2,  # (1, n_ctx//2, dim)
                        suffix_i,     # (1, *, dim)
                    ],
                    dim=1,
                )
                prompts.append(prompt)
            prompts = torch.cat(prompts, dim=0)

        elif self.class_token_position == "front":
            prompts = []
            for i in range(self.n_cls):
                name_len = self.name_lens[i]
                prefix_i = prefix[i : i + 1, :, :]
                class_i = suffix[i : i + 1, :name_len, :]
                suffix_i = suffix[i : i + 1, name_len:, :]
                ctx_i = ctx[i : i + 1, :, :]
                prompt = torch.cat(
                    [
                        prefix_i,  # (1, 1, dim)
                        class_i,   # (1, name_len, dim)
                        ctx_i,     # (1, n_ctx, dim)
                        suffix_i,  # (1, *, dim)
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
        self.prompt_learner = PromptLearner(cfg, classnames, clip_model)
        self.tokenized_prompts = self.prompt_learner.tokenized_prompts
        self.image_encoder = clip_model.visual
        self.text_encoder = TextEncoder(clip_model)
        self.logit_scale = clip_model.logit_scale
        self.dtype = clip_model.dtype

    def forward(self, image):
        image_features = self.image_encoder(image.type(self.dtype))
        
        # Handle SigLIP output: extract pooled features from BaseModelOutputWithPooling
        if hasattr(image_features, 'pooler_output'):
            # SigLIP returns BaseModelOutputWithPooling with pooler_output
            image_features = image_features.pooler_output
        elif hasattr(image_features, 'last_hidden_state'):
            # Fallback: use mean pooling of last hidden state
            image_features = image_features.last_hidden_state.mean(dim=1)
        # else: it's already a tensor (CLIP case)

        prompts = self.prompt_learner()
        tokenized_prompts = self.tokenized_prompts
        text_features = self.text_encoder(prompts, tokenized_prompts)

        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)

        logit_scale = self.logit_scale.exp()
        logits = logit_scale * image_features @ text_features.t()

        return logits


@TRAINER_REGISTRY.register()
class CoOp_SigLIP2(TrainerX):
    """Context Optimization (CoOp) with SigLIP 2 Support.

    Learning to Prompt for Vision-Language Models
    https://arxiv.org/abs/2109.01134
    
    Modified to support SigLIP models from Hugging Face.
    """

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

        print("Turning off gradients in both the image and the text encoder")
        for name, param in self.model.named_parameters():
            if "prompt_learner" not in name:
                param.requires_grad_(False)

        if cfg.MODEL.INIT_WEIGHTS:
            load_pretrained_weights(self.model.prompt_learner, cfg.MODEL.INIT_WEIGHTS)

        self.model.to(self.device)
        # NOTE: only give prompt_learner to the optimizer
        self.optim = build_optimizer(self.model.prompt_learner, cfg.OPTIM)
        self.sched = build_lr_scheduler(self.optim, cfg.OPTIM)
        self.register_model("prompt_learner", self.model.prompt_learner, self.optim, self.sched)

        self.scaler = GradScaler() if cfg.TRAINER.COOP.PREC == "amp" else None

        # Note that multi-gpu training could be slow because CLIP's size is
        # big, which slows down the copy operation in DataParallel
        device_count = torch.cuda.device_count()
        if device_count > 1:
            print(f"Multiple GPUs detected (n_gpus={device_count}), use all of them!")
            self.model = nn.DataParallel(self.model)

    def forward_backward(self, batch):
        image, label = self.parse_batch_train(batch)
        
        prec = self.cfg.TRAINER.COOP.PREC
        if prec == "amp":
            with autocast():
                output = self.model(image)
                loss = F.cross_entropy(output, label)
            self.optim.zero_grad()
            self.scaler.scale(loss).backward()
            self.scaler.step(self.optim)
            self.scaler.update()
        else:
            output = self.model(image)
            loss = F.cross_entropy(output, label)
            self.model_backward_and_update(loss)

        loss_summary = {
            "loss": loss.item(),
            "acc": compute_accuracy(output, label)[0].item(),
        }

        if (self.batch_idx + 1) == self.num_batches:
            self.update_lr()

        return loss_summary

    def parse_batch_train(self, batch):
        input = batch["img"]
        label = batch["label"]
        input = input.to(self.device)
        label = label.to(self.device)
        return input, label

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
            if "token_prefix" in state_dict:
                del state_dict["token_prefix"]

            if "token_suffix" in state_dict:
                del state_dict["token_suffix"]

            print("Loading weights to {} " 'from "{}" (epoch = {})'.format(name, model_path, epoch))
            # set strict=False
            self._models[name].load_state_dict(state_dict, strict=False)