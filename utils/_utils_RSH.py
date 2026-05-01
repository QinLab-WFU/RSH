import os
import random
import ssl
import subprocess
import threading
from copy import deepcopy
from functools import reduce
from urllib import request

import math
import numpy as np
import torch
import torch.nn.functional as F
from loguru import logger
from torch import optim
from torch.optim.lr_scheduler import MultiStepLR, CosineAnnealingLR, StepLR, ReduceLROnPlateau
from tqdm import tqdm


def gen_triplets(labels, ref_labels=None):
    if ref_labels is None:
        sames = (labels @ labels.T) > 0
    else:
        sames = (labels @ ref_labels.T) > 0

    diffs = ~sames

    if ref_labels is None:
        sames.fill_diagonal_(False)

    anc_idxes, pos_idxes, neg_idxes = torch.where(sames.unsqueeze(2) * diffs.unsqueeze(1))
    return anc_idxes, pos_idxes, neg_idxes
