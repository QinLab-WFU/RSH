
import numpy as np
from matplotlib import pyplot as plt
from tqdm import tqdm

import os
import torch

from torch.utils.data import DataLoader
import scipy.io as scio
import torch.nn

from model.hash_model_RSH import RSH

from train.base import TrainBase
from model.optimization import BertAdam

import time
import torch.nn.functional as F

from utils.RSHloss import RSHLoss

from utils.get_args import get_args

from utils.calc_utils import calc_map_k_matrix as calc_map_k
from dataset.dataloader import dataloader, dataloader2

from pynvml import *

class Trainer(TrainBase):

    def __init__(self, args, rank=0):
        # args = get_args()
        args.rank = rank
        super(Trainer, self).__init__(args)
        self.logger.info("dataset len: {}".format(len(self.train_loader.dataset)))
        self.cache = None
        self.run()

    def _init_model(self):
        self.logger.info("init model.")
        HashModel = RSH

        self.model = HashModel(outputDim=self.args.output_dim,
                           clipPath=self.args.clip_path, writer=self.writer, logger=self.logger,
                           is_train=self.args.is_train).to(self.rank).float()
        self.criterion = RSHLoss(gamma=self.args.gamma1, tau=self.args.tau1, square=False, normalized=False)

        to_opt = [
            {'params': self.model.clip.parameters(), 'lr': self.args.clip_lr},
            {'params': self.model.image_hash.parameters(), 'lr': self.args.lr},
            {'params': self.model.text_hash.parameters(), 'lr': self.args.lr},
            {'params': self.model.image_uncer.parameters(), 'lr': self.args.lr},
            {'params': self.model.text_uncer.parameters(), 'lr': self.args.lr},
            {'params': self.criterion.parameters(), 'lr': self.args.lr * 100}
        ]

        if self.args.pretrained != "" and os.path.exists(self.args.pretrained):
            self.logger.info("load pretrained model.")
            self.model.load_state_dict(torch.load(self.args.pretrained, map_location=f"cuda:{self.rank}"))

        self.optimizer = BertAdam(to_opt, lr=self.args.lr, warmup=self.args.warmup_proportion, schedule='warmup_cosine',
                                  b1=0.9, b2=0.98, e=1e-6, t_total=5000 * self.args.epochs,
                                  weight_decay=1e-4, max_grad_norm=1.0)

        self.total_time = 0
        print(self.model)

    # def _init_dataset(self):
    #     self.logger.info("init dataset.")
    #     self.logger.info(f"Using {self.args.dataset} dataset.")
    #     self.args.index_file = os.path.join("./dataset", self.args.dataset, self.args.index_file)
    #     self.args.caption_file = os.path.join("./dataset", self.args.dataset, self.args.caption_file)
    #     self.args.label_file = os.path.join("./dataset", self.args.dataset, self.args.label_file)
    #     train_data, query_data, retrieval_data = dataloader(captionFile=self.args.caption_file,
    #                                                         indexFile=self.args.index_file,
    #                                                         labelFile=self.args.label_file,
    #                                                         maxWords=self.args.max_words,
    #                                                         imageResolution=self.args.resolution,
    #                                                         query_num=self.args.query_num,
    #                                                         train_num=self.args.train_num,
    #                                                         seed=self.args.seed)
    #     self.train_labels = train_data.get_all_label().to(1)
    #
    #     self.noise_labels = self.add_noise_to_labels(self.train_labels)
    #
    #     train_data.labels = self.noise_labels
    #
    #     train_data.labels = train_data.labels.cpu().numpy()
    #
    #     self.query_labels = query_data.get_all_label()
    #     self.retrieval_labels = retrieval_data.get_all_label()
    #     self.args.retrieval_num = len(self.retrieval_labels)
    #     self.logger.info(f"query shape: {self.query_labels.shape}")
    #     self.logger.info(f"retrieval shape: {self.retrieval_labels.shape}")
    #     self.train_loader = DataLoader(
    #         dataset=train_data,
    #         batch_size=self.args.batch_size,
    #         num_workers=self.args.num_workers,
    #         pin_memory=True,
    #         shuffle=True
    #     )
    #     self.query_loader = DataLoader(
    #         dataset=query_data,
    #         batch_size=self.args.batch_size,
    #         num_workers=self.args.num_workers,
    #         pin_memory=True,
    #         shuffle=True
    #     )
    #     self.retrieval_loader = DataLoader(
    #         dataset=retrieval_data,
    #         batch_size=self.args.batch_size,
    #         num_workers=self.args.num_workers,
    #         pin_memory=True,
    #         shuffle=True
    #     )

    def _init_dataset(self):
        self.logger.info("init dataset.")
        self.logger.info(f"Using {self.args.dataset} dataset.")
        self.args.index_file = os.path.join("./dataset", self.args.dataset, self.args.index_file)
        self.args.caption_file = os.path.join("./dataset", self.args.dataset, self.args.caption_file)
        self.args.label_file = os.path.join("./dataset", self.args.dataset, self.args.label_file)
        train_data, query_data, retrieval_data = dataloader(captionFile=self.args.caption_file,
                                                            indexFile=self.args.index_file,
                                                            labelFile=self.args.label_file,
                                                            maxWords=self.args.max_words,
                                                            imageResolution=self.args.resolution,
                                                            query_num=self.args.query_num,
                                                            train_num=self.args.train_num,
                                                            seed=self.args.seed)
        self.train_labels = train_data.get_all_label().to(1)
        self.query_labels = query_data.get_all_label()
        self.retrieval_labels = retrieval_data.get_all_label()
        self.args.retrieval_num = len(self.retrieval_labels)
        self.logger.info(f"query shape: {self.query_labels.shape}")
        self.logger.info(f"retrieval shape: {self.retrieval_labels.shape}")
        self.train_loader = DataLoader(
            dataset=train_data,
            batch_size=self.args.batch_size,
            num_workers=self.args.num_workers,
            pin_memory=True,
            shuffle=True
        )
        self.query_loader = DataLoader(
            dataset=query_data,
            batch_size=self.args.batch_size,
            num_workers=self.args.num_workers,
            pin_memory=True,
            shuffle=True
        )
        self.retrieval_loader = DataLoader(
            dataset=retrieval_data,
            batch_size=self.args.batch_size,
            num_workers=self.args.num_workers,
            pin_memory=True,
            shuffle=True
        )

    def add_noise_to_labels(self, labels):
        labels = labels.cpu().numpy()  # 转换为 numpy
        num_samples, num_labels = labels.shape
        num_noise = int(num_samples * self.args.noise_rate)

        noise_indices = np.random.choice(num_samples, num_noise, replace=False)

        for i in noise_indices:
            ones_indices = np.where(labels[i, :] == 1)[0]
            zeros_indices = np.where(labels[i, :] == 0)[0]

            if len(ones_indices) > 0:
                j = np.random.choice(ones_indices)
                labels[i, j] = 0

            if len(zeros_indices) > 0:
                j = np.random.choice(zeros_indices)
                labels[i, j] = 1

        return torch.tensor(labels, dtype=torch.float32).to(self.rank)  # 转换回 Tensor 并放回 GPU

    def quant_loss(self, u, v, label=None, label_confidence=None):
        loss_u = torch.norm(u - u.sign(), p=1) / u.numel()
        loss_v = torch.norm(v - v.sign(), p=1) / v.numel()
        return loss_u + loss_v

    def train_epoch(self, epoch):
        self.change_state(mode="train")
        self.logger.info(">>>>>> epochs: %d/%d" % (epoch, self.args.epochs))

        all_loss = 0
        for image, text, label, index in tqdm(self.train_loader):

            if image.shape[0] < self.args.batch_size:
                continue

            start_time = time.time()
            self.global_step += 1
            image = image.to(self.rank, non_blocking=True).float()
            text = text.to(self.rank, non_blocking=True)
            label = label.to(self.rank, non_blocking=True).float()
            self.model = self.model.to(self.rank)

            noise_label = self.add_noise_to_labels(label)

            # 1
            img_hash, img_uncer, txt_hash, txt_uncer = self.model(image, text)
            loss = self.criterion(img_hash, img_uncer, noise_label)[0]
            loss += self.criterion(img_hash, img_uncer, noise_label, txt_hash, txt_uncer)[0]
            loss += self.criterion(txt_hash, txt_uncer, noise_label, img_hash, img_uncer)[0]
            loss += self.criterion(txt_hash, txt_uncer, noise_label)[0]
            loss /= 4

            loss5 = self.quant_loss(img_hash, txt_hash)
            loss = loss + 0.03 * loss5

            all_loss += loss.data
            self.optimizer.zero_grad()
            if loss != 0:
                loss.backward()
            self.optimizer.step()
            self.total_time += time.time() - start_time
        self.logger.info(
            f">>>>>> [{epoch}/{self.args.epochs}] loss: {all_loss.data / (len(self.train_loader))}, lr: {'-'.join([str('%.9f' % itm) for itm in sorted(list(set(self.optimizer.get_lr())))])}, time: {self.total_time}")

    def mixup(self, img, txt, y, alpha):
        batch_size = img.shape[0]

        _lambda = np.random.beta(alpha, alpha)
        indices = torch.randperm(batch_size, device=img.device)

        mixed_i = _lambda * img + (1 - _lambda) * img[indices, :]
        mixed_t = _lambda * txt + (1 - _lambda) * txt[indices, :]
        y_a, y_b = y, y[indices]

        return mixed_i, mixed_t, y_a, y_b, _lambda

    def train(self):
        self.logger.info("Start train.")
        for epoch in range(self.args.epochs):
            self.train_epoch(epoch)
            self.valid(epoch)
            self.save_model(epoch)

        self.logger.info(f">>>>>>> FINISHED >>>>>> Best epoch, I-T: {self.best_epoch_i}, mAP: {self.max_mapi2t}, T-I: {self.best_epoch_t}, mAP: {self.max_mapt2i}")

    def get_code(self, data_loader, length: int):

        img_buffer = torch.empty(length, self.args.output_dim, dtype=torch.float).to(self.rank)
        text_buffer = torch.empty(length, self.args.output_dim, dtype=torch.float).to(self.rank)
        encoder_time = 0
        for image, text, label, index in tqdm(data_loader):
            start_encoder_time = time.time()
            image = image.to(self.rank, non_blocking=True)
            text = text.to(self.rank, non_blocking=True)
            index = index.numpy()
            image_hash = self.model.encode_image(image)
            image_hash = torch.sign(image_hash)
            text_hash = self.model.encode_text(text)
            text_hash = torch.sign(text_hash)
            encoder_time = time.time() - start_encoder_time

            img_buffer[index, :] = image_hash.data
            text_buffer[index, :] = text_hash.data

        return img_buffer, text_buffer, encoder_time

    def valid(self, epoch):
        self.logger.info("Valid.")
        self.change_state(mode="valid")

        query_img, query_txt, q_encoder_time = self.get_code(self.query_loader, self.args.query_num)
        retrieval_img, retrieval_txt, r_encoder_time = self.get_code(self.retrieval_loader, self.args.retrieval_num)

        mAPi2t = calc_map_k(query_img, retrieval_txt, self.query_labels, self.retrieval_labels, None, self.rank)
        mAPt2i = calc_map_k(query_txt, retrieval_img, self.query_labels, self.retrieval_labels, None, self.rank)
        mAPi2i = calc_map_k(query_img, retrieval_img, self.query_labels, self.retrieval_labels, None, self.rank)
        mAPt2t = calc_map_k(query_txt, retrieval_txt, self.query_labels, self.retrieval_labels, None, self.rank)
        if self.max_mapi2t < mAPi2t:
            self.best_epoch_i = epoch
            self.save_mat(query_img, query_txt, retrieval_img, retrieval_txt, mode_name="i2t")
        self.max_mapi2t = max(self.max_mapi2t, mAPi2t)
        if self.max_mapt2i < mAPt2i:
            self.best_epoch_t = epoch
            self.save_mat(query_img, query_txt, retrieval_img, retrieval_txt, mode_name="t2i")
        self.max_mapt2i = max(self.max_mapt2i, mAPt2i)
        self.logger.info(f">>>>>> [{epoch}/{self.args.epochs}], MAP(i->t): {mAPi2t}, MAP(t->i): {mAPt2i}, MAP(t->t): {mAPt2t}, MAP(i->i): {mAPi2i}, \
                    MAX MAP(i->t): {self.max_mapi2t}, MAX MAP(t->i): {self.max_mapt2i}, query_encoder_time: {q_encoder_time}, retrieval_encoder_time: {r_encoder_time}")


    def save_mat(self, query_img, query_txt, retrieval_img, retrieval_txt, mode_name="i2t"):

        save_dir = os.path.join(self.args.save_dir, "PR_cruve")
        os.makedirs(save_dir, exist_ok=True)

        query_img = query_img.cpu().detach().numpy()
        query_txt = query_txt.cpu().detach().numpy()
        retrieval_img = retrieval_img.cpu().detach().numpy()
        retrieval_txt = retrieval_txt.cpu().detach().numpy()
        query_labels = self.query_labels.numpy()
        retrieval_labels = self.retrieval_labels.numpy()

        result_dict = {
            'q_img': query_img,
            'q_txt': query_txt,
            'r_img': retrieval_img,
            'r_txt': retrieval_txt,
            'q_l': query_labels,
            'r_l': retrieval_labels
        }
        scio.savemat(os.path.join(save_dir, str(self.args.output_dim) + "-ours-" + self.args.dataset + "-" + mode_name + ".mat"), result_dict)
        self.logger.info(f">>>>>> save best {mode_name} data!")