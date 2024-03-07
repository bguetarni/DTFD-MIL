import torch
import torch.nn as nn
import torch.nn.functional as F
import einops
from Model.network import Classifier_1fc

class StainFusion(nn.Module):
    def __init__(self, mDim, stains, *args, **kwargs):
        super().__init__()

        self.stains = stains
        self.norm = nn.LayerNorm(mDim)
        self.drop = nn.Dropout(0.1)
        self.mlp = nn.Linear(mDim, len(stains)-1)

    def forward(self, x, *args, **kwargs):
        N, _ = x[self.stains[0]].shape
        stacked_seq = torch.cat([x[s] for s in self.stains], dim=0)
        stacked_seq = self.norm(stacked_seq)
        
        # global normalisation
        unstacked_seq = einops.rearrange(stacked_seq, '(e n) c -> e n c', e=len(self.stains), n=N).unbind(dim=0)
        x_HES = unstacked_seq[0]   # WARNING: HES/HE must be first stain
        x_IHCs = unstacked_seq[1:]

        cls_token = x_HES
        q = self.mlp(self.drop(cls_token))
        v = torch.stack(x_IHCs, dim=1)
        cls_token = (cls_token + q @ v) / 2
        return cls_token.squeeze(dim=1)


class Attention_Gated(nn.Module):
    def __init__(self, L=512, D=128, K=1):
        super(Attention_Gated, self).__init__()

        self.L = L
        self.D = D
        self.K = K

        self.attention_V = nn.Sequential(
            nn.Linear(self.L, self.D),
            nn.Tanh()
        )

        self.attention_U = nn.Sequential(
            nn.Linear(self.L, self.D),
            nn.Sigmoid()
        )

        self.attention_weights = nn.Linear(self.D, self.K)

    def forward(self, x, isNorm=True):
        ## x: N x L
        A_V = self.attention_V(x)  # NxD
        A_U = self.attention_U(x)  # NxD
        A = self.attention_weights(A_V * A_U) # NxK
        A = torch.transpose(A, 1, 0)  # KxN

        if isNorm:
            A = F.softmax(A, dim=1)  # softmax over N

        return A  ### K x N


class Attention_with_Classifier(nn.Module):
    def __init__(self, stains, L=512, D=128, K=1, num_cls=2, droprate=0):
        super(Attention_with_Classifier, self).__init__()
        self.attention = Attention_Gated(L, D, K)
        self.classifier = Classifier_1fc(L, num_cls, droprate)

        self.stainfusion = StainFusion(L, stains) if not stains is None else None
    def forward(self, x): ## x: N x L
        if not self.stainfusion is None:
            AA = {k: self.attention(v) for k, v in x.items()}  ## K x N
            afeat = {k: torch.mm(AA[k], x[k]) for k in x.keys()} ## K x L
            afeat = self.stainfusion(afeat)
        else:
            AA = self.attention(x)  ## K x N
            afeat = torch.mm(AA, x) ## K x L

        pred = self.classifier(afeat) ## K x num_cls
        return pred