import torch
import torch.nn.functional as F
from torch import nn

from utils._utils_RSH import gen_triplets


class RSHLoss(nn.Module):
    def __init__(self, gamma=0.0, tau=5.0, margin=0.25, square=True, normalized=True):
        super().__init__()
        self.gamma = gamma
        self.tau = tau
        self.margin = margin
        self.square = square
        self.normalized = normalized

    def forward(self, S, U, T, ref_S=None, ref_U=None):
        sim_mat = self.calc_SSM(S, U, ref_S, ref_U)

        anc_idxes, pos_idxes, neg_idxes = gen_triplets(T) if ref_S is None else gen_triplets(T, T)

        S_ap = sim_mat[anc_idxes, pos_idxes]
        S_an = sim_mat[anc_idxes, neg_idxes]

        losses = F.relu(S_an - S_ap + self.margin)

        mask = losses > 0
        N = mask.sum()

        if N == 0:
            loss = 0
        else:
            loss = losses[mask].mean()
        return loss, N

    def calc_SSM(self, S, U, ref_S, ref_U):

        if not self.normalized:
            S = F.normalize(S)

        if ref_S is None:
            ref_S = S
        else:
            if not self.normalized:
                ref_S = F.normalize(ref_S)

        if ref_U is None:
            ref_U = U

        cos_sim = S @ ref_S.T

        # norm(s1 - s2)
        alpha = F.pairwise_distance(S.unsqueeze(1), ref_S.unsqueeze(0))

        # norm(u1 + u2)
        beta = F.pairwise_distance(U.unsqueeze(1), -ref_U.unsqueeze(0))

        if self.square:
            alpha = alpha**2
            beta = beta**2

        beta_r = (beta + self.gamma) / alpha
        S_in = 1 - (1 - cos_sim) * torch.exp(-beta_r / self.tau)
        return S_in

