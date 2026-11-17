"""Four-channel variants; fixed frontend detached, gate and mix differentiable."""
import torch
from torch import nn

class Variant(nn.Module):
    def __init__(self,backbone,method):
        super().__init__()
        if method not in ('A','B','C','G','D'):raise ValueError(method)
        self.backbone=backbone;self.method=method
        if method=='G':self.b=nn.Parameter(torch.zeros(1))
        if method=='D':
            self.gate=nn.Linear(3,1)
            nn.init.zeros_(self.gate.weight);nn.init.zeros_(self.gate.bias)

    def mix(self,x,d,features):
        if x.ndim!=3 or x.shape[1:]!=(4,256) or d.shape!=x[:,0,:].shape or features.shape!=(len(x),3):raise ValueError('Input shape mismatch')
        if not all(torch.isfinite(v).all() for v in (x,d,features)):raise ValueError('Non-finite common input')
        d=d.detach();features=features.detach()
        if self.method=='A':alpha=torch.zeros_like(x[:,0,:1]);z=x[:,0,:]
        elif self.method=='B':alpha=torch.ones_like(x[:,0,:1]);z=d
        else:
            if self.method=='C':alpha=torch.full_like(x[:,0,:1],0.5)
            elif self.method=='G':alpha=torch.sigmoid(self.b).expand(len(x),1)
            else:alpha=torch.sigmoid(self.gate(features))
            z=x[:,0,:]+alpha*(d-x[:,0,:])
        return torch.cat([z[:,None,:],x[:,1:,:]],dim=1),alpha

    def forward(self,x,d,features):
        mixed,alpha=self.mix(x,d,features)
        return self.backbone(mixed),alpha
