"""OpenCLIP image/text frontend with an explicit preprojection feature."""
import json
from pathlib import Path
import torch
import torch.nn.functional as F

BIOMEDCLIP = "hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224"

class Encoder:
    def __init__(self, model_name=BIOMEDCLIP, checkpoint_dir=None, device='cuda', pretrained=None):
        import open_clip
        self.device = torch.device(device)
        if checkpoint_dir:
            folder = Path(checkpoint_dir).resolve()
            cfg = json.loads((folder/'open_clip_config.json').read_text())
            from open_clip.factory import _MODEL_CONFIGS
            model_cfg = cfg['model_cfg']
            if 'hf_model_name' in model_cfg.get('text_cfg', {}):
                model_cfg['text_cfg'].update(hf_model_name=str(folder), hf_tokenizer_name=str(folder))
            name = 'medical_tta_local'
            _MODEL_CONFIGS[name] = model_cfg
            self.model, _, self.preprocess = open_clip.create_model_and_transforms(
                name, pretrained=str(folder/'open_clip_pytorch_model.bin'), pretrained_hf=False,
                **{'image_'+k:v for k,v in cfg.get('preprocess_cfg',{}).items()})
            self.tokenizer = open_clip.get_tokenizer(name)
        else:
            if not model_name.startswith('hf-hub:') and not pretrained:
                raise ValueError("Specify --pretrained for a standard OpenCLIP model.")
            self.model, _, self.preprocess = open_clip.create_model_and_transforms(
                model_name, pretrained=pretrained)
            self.tokenizer = open_clip.get_tokenizer(model_name)
        self.model.float().to(self.device).eval().requires_grad_(False)
        self.scale = float(self.model.logit_scale.exp())
        normalization = self.preprocess.transforms[-1]
        self.mean = torch.tensor(normalization.mean, device=self.device).reshape(1,3,1,1)
        self.std = torch.tensor(normalization.std, device=self.device).reshape(1,3,1,1)

    @torch.no_grad()
    def encode(self, batch):
        x = batch.to(self.device)
        visual = self.model.visual
        if hasattr(visual, 'trunk') and hasattr(visual, 'head'):
            h = visual.trunk.forward_head(visual.trunk.forward_features(x))
            v = visual.head(h)
        elif hasattr(visual, '_embeds') and hasattr(visual, '_pool'):
            h, _ = visual._pool(visual.transformer(visual._embeds(x)))
            v = h @ visual.proj if visual.proj is not None else h
        else:
            raise TypeError("This encoder needs a backbone-specific preprojection frontend; use the cache interface.")
        if h.ndim != 2 or v.ndim != 2:
            raise ValueError("The visual frontend must return one feature vector per image.")
        return h, v

    @torch.no_grad()
    def text(self, names, templates):
        rows = []
        for name in names:
            tokens = self.tokenizer([t.format(name) for t in templates]).to(self.device)
            z = self.model.encode_text(tokens).float()
            rows.append(F.normalize(F.normalize(z, dim=-1).mean(0), dim=-1))
        return torch.stack(rows)
