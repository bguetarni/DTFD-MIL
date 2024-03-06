import random
import numpy as np

import torch, torch.nn as nn
import torch.nn.functional as F
import einops

"""
The batch size is set to be 1, 
    meaning that in each iteration,  one slide is processed
"""

class residual_block(nn.Module):
    def __init__(self, nChn=512):
        super(residual_block, self).__init__()
        self.block = nn.Sequential(
                nn.Linear(nChn, nChn, bias=False),
                nn.ReLU(inplace=True),
                nn.Linear(nChn, nChn, bias=False),
                nn.ReLU(inplace=True),
            )
    def forward(self, x):
        tt = self.block(x)
        x = x + tt
        return x


class DimReduction(nn.Module):
    def __init__(self, n_channels, m_dim, numLayer_Res=0):
        super(DimReduction, self).__init__()
        self.fc1 = nn.Linear(n_channels, m_dim, bias=False)
        self.relu1 = nn.ReLU(inplace=True)
        self.numRes = numLayer_Res

        self.resBlocks = []
        for ii in range(numLayer_Res):
            self.resBlocks.append(residual_block(m_dim))
        self.resBlocks = nn.Sequential(*self.resBlocks)

    def forward(self, x):

        x = self.fc1(x)
        x = self.relu1(x)

        if self.numRes > 0:
            x = self.resBlocks(x)

        return x


class Classifier_1fc(nn.Module):
    def __init__(self, n_channels, n_classes, droprate=0.0):
        super(Classifier_1fc, self).__init__()
        self.fc = nn.Linear(n_channels, n_classes)
        self.droprate = droprate
        if self.droprate != 0.0:
            self.dropout = torch.nn.Dropout(p=self.droprate)

    def forward(self, x):

        if self.droprate != 0.0:
            x = self.dropout(x)
        x = self.fc(x)
        return x


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

    def forward(self, x):
        A_V = self.attention_V(x)
        A_U = self.attention_U(x)
        A = self.attention_weights(A_V * A_U)
        A = torch.transpose(A, 1, 0)
        A = F.softmax(A, dim=1)
        afeat = torch.mm(A, x)
        return afeat

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

class Tier(nn.Module):
    def __init__(self, mDim, num_cls, stains=None, *args, **kwargs):
        """
            stains : list of stains if multi-stain model
        """
        super().__init__()
        
        self.attention = Attention_Gated(mDim)
        self.classifier = Classifier_1fc(mDim, num_cls)
        self.stainfusion = StainFusion(mDim, stains) if not stains is None else None

    def forward(self, x, *args, **kwargs):
        if not self.stainfusion is None:
            tattFeat_tensor = {k: self.attention(v) for k, v in x.items()}
            x = self.stainfusion(tattFeat_tensor)
        else:
            tattFeat_tensor = self.attention(x)
            x = tattFeat_tensor
        
        return tattFeat_tensor, self.classifier(x)

class DTFDMIL(nn.Module):
    def __init__(self, stains, in_channels, mDim, num_cls, *args, **kwargs):
        super().__init__()
        
        self.stains = stains
        self.dimReduction = DimReduction(in_channels, mDim)
        self.tier1 = Tier(mDim, num_cls, stains)
        self.tier2 = Tier(mDim, num_cls, stains)
    
    def forward(self, x, *args, **kwargs):
        """
            x is a group of bags
        """

        sub_bags_fts = []
        sub_bags_pred = []
        for sub_bag in x:

            if not self.stains is None:
                sub_bag = {k: self.dimReduction(v) for k, v in sub_bag.items()}
            else:
                sub_bag = self.dimReduction(sub_bag)
            
            tattFeat_tensor, preds = self.tier1(sub_bag)
            sub_bags_pred.append(preds)
            sub_bags_fts.append(tattFeat_tensor)
        
        if not self.stains is None:
            sub_bags_fts = {k: torch.cat([i[k] for i in sub_bags_fts], dim=0) for k in self.stains}
        else:
            sub_bags_fts = torch.cat(sub_bags_fts, dim=0)
        
        sub_bags_pred = torch.cat(sub_bags_pred, dim=0)

        _, slide_pred = self.tier2(sub_bags_fts)
        return slide_pred, sub_bags_pred


class Teacher(DTFDMIL):
    def __init__(self, stains, **kwargs):
        super().__init__(stains=stains, **kwargs)


class Student(DTFDMIL):
    def __init__(self, **kwargs):
        super().__init__(stains=None, **kwargs)
