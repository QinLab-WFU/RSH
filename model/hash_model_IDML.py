import os
import torch
import logging
import torch.nn as nn
from model.model_m import build_model
from utils.logger import get_logger, get_summary_writer
import torch.nn.functional as F


class LinearHash(nn.Module):

    def __init__(self, inputDim=2048, outputDim=64):
        super(LinearHash, self).__init__()
        self.fc = nn.Linear(inputDim, outputDim)
        self.drop_out = nn.Dropout(p=0.2)
    
    def forward(self, data):
        result = self.fc(data)
        return torch.tanh(self.drop_out(result))

def l2_norm(input):
    input_size = input.size()
    buffer = torch.pow(input, 2)
    normp = torch.sum(buffer, 1).add_(1e-12)
    norm = torch.sqrt(normp)
    _output = torch.div(input, norm.view(-1, 1).expand_as(input))
    output = _output.view(input_size)

    return output

class IDML(nn.Module):

    def __init__(self, 
                outputDim=64, 
                clipPath="./ViT-B-32.pt", 
                writer=None, 
                saveDir="./result/log", 
                logger: logging.Logger=None, 
                is_train=True,
                ):
        super(IDML, self).__init__()
        os.makedirs(saveDir, exist_ok=True)
        self.logger = logger if logger is not None else get_logger(os.path.join(saveDir, "train.log" if is_train else "test.log"))
        self.writer = writer if writer is not None and is_train else get_summary_writer(os.path.join(saveDir, "tensorboard"))
        embedDim, self.clip = self.load_clip(clipPath)
        self.image_hash = LinearHash(inputDim=embedDim, outputDim=outputDim)
        self.text_hash = LinearHash(inputDim=embedDim, outputDim=outputDim)
        self.image_uncer = nn.Linear(embedDim, outputDim)
        self.text_uncer = nn.Linear(embedDim, outputDim)

    def load_clip(self, clipPath: str) -> tuple:
        try:
            model = torch.jit.load(clipPath, map_location="cpu").eval()
            state_dict = model.state_dict()
        except RuntimeError:
            state_dict = torch.load(clipPath, map_location="cpu")
        return state_dict["text_projection"].shape[1], build_model(state_dict)

    def encode_image(self, image):

        image_embed = self.clip.encode_image(image) #512
        image_embed = self.image_hash(image_embed)
        image_embed = F.normalize(image_embed)
        return image_embed

    def eval(self):
        self.image_hash.eval()
        self.text_hash.eval()

    def train(self):
        self.image_hash.train()
        self.text_hash.train()
    
    def encode_text(self, text):

        text_embed = self.clip.encode_text(text)
        text_embed = self.text_hash(text_embed)
        text_embed = F.normalize(text_embed)
        return text_embed

    def forward(self, image, text):

        image_embed = self.clip.encode_image(image)  # 512
        img_hash = self.image_hash(image_embed)
        img_uncer = self.image_uncer(image_embed)
        text_embed = self.clip.encode_text(text)
        txt_hash = self.text_hash(text_embed)
        txt_uncer = self.text_uncer(text_embed)

        return img_hash, img_uncer, txt_hash, txt_uncer
