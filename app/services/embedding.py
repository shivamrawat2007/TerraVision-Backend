from pathlib import Path
import hashlib
import re
import numpy as np
from PIL import Image

class EmbeddingService:
    """Unified embedding adapter.

    demo: dependency-light deterministic QA backend.
    sentence_transformers: local SentenceTransformer model.
    transformers_clip: local Hugging Face CLIP-compatible model.
    remoteclip: official RemoteCLIP OpenCLIP checkpoint staged locally.
    """
    def __init__(self, model_name, device="cpu", backend="demo"):
        self.device=device
        self.model_name=model_name or "demo-shared-features"
        self.backend=backend
        self.model=None
        self.tokenizer=None
        if backend == "sentence_transformers":
            from sentence_transformers import SentenceTransformer
            self.model=SentenceTransformer(model_name, device=device)
        elif backend == "transformers_clip":
            import torch
            from transformers import CLIPModel, CLIPProcessor
            self.processor=CLIPProcessor.from_pretrained(model_name, local_files_only=True)
            self.model=CLIPModel.from_pretrained(model_name, local_files_only=True).to(device)
            self.model.eval()
        elif backend == "remoteclip":
            import open_clip
            ckpt = Path(model_name)
            if not ckpt.exists():
                raise FileNotFoundError(f"RemoteCLIP checkpoint not found: {ckpt}")
            arch = __import__('os').getenv("REMOTECLIP_ARCH", "ViT-B-32")
            self.model, _, self.preprocess = open_clip.create_model_and_transforms(
            arch, pretrained=str(ckpt), device=device
            )

            # Memory optimization:
            # RemoteCLIP FP32 is ~577 MB, which is too large for Render's 512 MB plan.
            # FP16 reduces model parameter memory by roughly 50%.
            if device == "cpu":
                self.model = self.model.half()

            self.tokenizer = open_clip.get_tokenizer(arch)
            self.model.eval()

    @staticmethod
    def _norm(v):
        v=np.asarray(v,dtype="float32")
        n=float(np.linalg.norm(v))
        return v/n if n else v

    def _demo_image(self,path):
        im=np.asarray(Image.open(path).convert("RGB").resize((128,128)),dtype=np.float32)/255.0
        r,g,b=im[:,:,0],im[:,:,1],im[:,:,2]
        gray=im.mean(axis=2)
        blue=((b>r*1.15)&(b>g*1.05)).mean()
        green=((g>r*1.12)&(g>b*0.95)).mean()
        soil=((r>g*1.05)&(g>b*1.15)).mean()
        bright=(gray>.75).mean(); dark=(gray<.20).mean()
        edge=np.abs(np.diff(gray,axis=0)).mean()+np.abs(np.diff(gray,axis=1)).mean()
        v=np.zeros(64,dtype=np.float32); v[:6]=[blue,green,soil,bright,dark,edge]
        p=6
        for c in range(3):
            hist,_=np.histogram(im[:,:,c],bins=16,range=(0,1),density=True)
            v[p:p+16]=hist/16.0; p+=16
        return self._norm(v)

    def _demo_text(self,text):
        t=text.lower(); v=np.zeros(64,dtype=np.float32)
        keywords={
            "river":(0,2.5),"water":(0,2.5),"lake":(0,2.5),"flood":(0,2.0),
            "vegetation":(1,2.5),"forest":(1,2.5),"crop":(1,2.0),"green":(1,2),
            "soil":(2,2),"sand":(2,2),"desert":(2,2),"ground":(2,1),
            "building":(3,1.5),"buildings":(3,1.8),"structure":(3,1.8),"structures":(3,2.0),
            "road":(5,1.5),"roads":(5,1.5),"vehicle":(5,1.2),"vehicles":(5,1.5),
            "cloud":(4,1.5),"shadow":(4,1),
        }
        for word,(idx,w) in keywords.items():
            if re.search(r"\b"+re.escape(word)+r"\b",t): v[idx]+=w
        for tok in re.findall(r"[a-z0-9]+",t):
            h=int(hashlib.sha256(tok.encode()).hexdigest()[:8],16)
            v[6+(h%58)]+=0.08
        return self._norm(v)

    def image(self,path:Path):
        if self.backend=="demo": return self._demo_image(path)
        if self.backend=="transformers_clip":
            import torch
            image=Image.open(path).convert("RGB")
            with torch.inference_mode():
                inp=self.processor(images=image,return_tensors="pt").to(self.device)
                v=self.model.get_image_features(**inp)
                return self._norm(v[0].cpu().numpy())
        if self.backend=="remoteclip":
            import torch
            image=self.preprocess(Image.open(path).convert("RGB")).unsqueeze(0).to(self.device)
            with torch.inference_mode():
                v=self.model.encode_image(image)
                return self._norm(v[0].cpu().numpy())
        return self._norm(self.model.encode(Image.open(path).convert("RGB"),normalize_embeddings=True)[0])

    def text(self,text:str):
        if self.backend=="demo": return self._demo_text(text)
        if self.backend=="transformers_clip":
            import torch
            with torch.inference_mode():
                inp=self.processor(text=[text],return_tensors="pt",padding=True).to(self.device)
                v=self.model.get_text_features(**inp)
                return self._norm(v[0].cpu().numpy())
        if self.backend=="remoteclip":
            import torch
            tok=self.tokenizer([text]).to(self.device)
            with torch.inference_mode():
                v=self.model.encode_text(tok)
                return self._norm(v[0].cpu().numpy())
        return self._norm(self.model.encode([text],normalize_embeddings=True)[0])
