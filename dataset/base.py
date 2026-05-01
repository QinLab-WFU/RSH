from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from __future__ import print_function

import os

import numpy as np
from torch.utils.data import Dataset, DataLoader
import torch
import random
from PIL import Image
from torchvision.transforms import Compose, Resize, CenterCrop, ToTensor, Normalize
from model.simple_tokenizer import SimpleTokenizer as Tokenizer


class BaseDataset(Dataset):

    def __init__(self,

                 captions: dict,
                 indexs: dict,
                 labels: dict,
                 is_train=True,
                 tokenizer=Tokenizer(),
                 maxWords=32,
                 imageResolution=224,
                 npy=False):

        self.captions = captions
        self.indexs = indexs
        self.labels = labels
        self.npy = npy

        self.maxWords = maxWords
        self.tokenizer = tokenizer

        self.transform = Compose([
            Resize(imageResolution, interpolation=Image.BICUBIC),
            CenterCrop(imageResolution),
            ToTensor(),
            Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)),
        ]) if is_train else Compose([
            Resize((imageResolution, imageResolution), interpolation=Image.BICUBIC),
            ToTensor(),
            Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)),
        ])
        self.SPECIAL_TOKEN = {"CLS_TOKEN": "<|startoftext|>", "SEP_TOKEN": "<|endoftext|>",
                              "MASK_TOKEN": "[MASK]", "UNK_TOKEN": "[UNK]", "PAD_TOKEN": "[PAD]"}

        self.__length = len(self.indexs)

    def __len__(self):
        return self.__length

    def _load_image(self, index: int) -> torch.Tensor:
        if not self.npy:
            image_path = self.indexs[index].strip()
            # print(image_path)
            image = Image.open(image_path).convert("RGB")
        else:
            image = Image.fromarray(self.indexs[index]).convert("RGB")
        image = self.transform(image)

        return image

    def _load_text(self, index: int):
        captions = self.captions[index]
        use_cap = captions[random.randint(0, len(captions) - 1)]

        words = self.tokenizer.tokenize(use_cap)
        words = [self.SPECIAL_TOKEN["CLS_TOKEN"]] + words
        total_length_with_CLS = self.maxWords - 1
        if len(words) > total_length_with_CLS:
            words = words[:total_length_with_CLS]

        words = words + [self.SPECIAL_TOKEN["SEP_TOKEN"]]
        caption = self.tokenizer.convert_tokens_to_ids(words)

        while len(caption) < self.maxWords:
            caption.append(0)
        caption = torch.tensor(caption)

        return caption

    def _load_label(self, index: int) -> torch.Tensor:
        label = self.labels[index]
        label = torch.from_numpy(label)

        return label

    def get_all_label(self):
        labels = torch.zeros([self.__length, len(self.labels[0])], dtype=torch.float32)
        for i, item in enumerate(self.labels):
            labels[i] = torch.from_numpy(item)
        return labels

    def __getitem__(self, index):
        image = self._load_image(index)
        caption = self._load_text(index)
        label = self._load_label(index)

        return image, caption, label, index

class BaseDataset1(Dataset):
    def __init__(self,
                 captions: dict,
                 indexs: dict,
                 labels: dict,
                 is_train=True,
                 tokenizer=Tokenizer(),
                 maxWords=32,
                 imageResolution=224,
                 noise_ratio=0.0,  # 添加噪声比例参数
                 npy=False):

        self.captions = captions
        self.indexs = indexs
        self.labels = labels
        self.npy = npy
        self.is_train = is_train
        self.noise_ratio = noise_ratio

        self.maxWords = maxWords
        self.tokenizer = tokenizer

        self.transform = Compose([
            Resize(imageResolution, interpolation=Image.BICUBIC),
            CenterCrop(imageResolution),
            ToTensor(),
            Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)),
        ]) if is_train else Compose([
            Resize((imageResolution, imageResolution), interpolation=Image.BICUBIC),
            ToTensor(),
            Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)),
        ])
        self.SPECIAL_TOKEN = {"CLS_TOKEN": "<|startoftext|>", "SEP_TOKEN": "<|endoftext|>",
                              "MASK_TOKEN": "[MASK]", "UNK_TOKEN": "[UNK]", "PAD_TOKEN": "[PAD]"}

        self.__length = len(self.indexs)

        # 初始化噪声索引映射
        self.noisy_mapping = np.arange(self.__length)
        self.noisy_labels = None  # 用于存储噪声标签

        # 只在训练集且噪声比例大于0时生成噪声
        if is_train and noise_ratio > 0.0:
            self._generate_asymmetric_noise()

    def _generate_asymmetric_noise(self):
        np.random.seed(1814)  # 固定随机种子以保证可复现性

        # 1. 选择噪声样本
        num_samples = self.__length
        num_noisy = int(self.noise_ratio * num_samples)
        noisy_indices = np.random.choice(num_samples, num_noisy, replace=False)

        # 2. 创建置换映射（非对称噪声）
        # 对噪声样本的索引进行重排
        shuffled_indices = np.array(noisy_indices)
        np.random.shuffle(shuffled_indices)

        # 3. 创建映射关系：原始索引 -> 错误匹配的索引
        self.noisy_mapping[noisy_indices] = shuffled_indices

        # 4. 保存噪声标签信息（用于可能的噪声感知训练）
        self.noisy_labels = np.zeros(num_samples, dtype=np.int64)
        self.noisy_labels[noisy_indices] = 1  # 标记噪声样本

        print(f"Generated asymmetric noise with ratio: {self.noise_ratio}")
        print(f"Noisy samples: {num_noisy}/{num_samples}")

    def __len__(self):
        return self.__length

    def _load_image(self, index: int) -> torch.Tensor:
        # 如果有噪声映射，使用映射后的索引
        if self.is_train and self.noise_ratio > 0.0:
            actual_index = self.noisy_mapping[index]
        else:
            actual_index = index

        if not self.npy:
            image_path = self.indexs[actual_index].strip()
            image = Image.open(image_path).convert("RGB")
        else:
            image = Image.fromarray(self.indexs[actual_index]).convert("RGB")
        image = self.transform(image)

        return image

    def _load_text(self, index: int):
        # 文本保持不变，只有图像被错误匹配
        captions = self.captions[index]
        use_cap = captions[random.randint(0, len(captions) - 1)]

        words = self.tokenizer.tokenize(use_cap)
        words = [self.SPECIAL_TOKEN["CLS_TOKEN"]] + words
        total_length_with_CLS = self.maxWords - 1
        if len(words) > total_length_with_CLS:
            words = words[:total_length_with_CLS]

        words = words + [self.SPECIAL_TOKEN["SEP_TOKEN"]]
        caption = self.tokenizer.convert_tokens_to_ids(words)

        while len(caption) < self.maxWords:
            caption.append(0)
        caption = torch.tensor(caption)

        return caption

    def _load_label(self, index: int) -> torch.Tensor:
        # 标签保持不变，对应原始文本
        label = self.labels[index]
        label = torch.from_numpy(label)

        return label

    def __getitem__(self, index):
        image = self._load_image(index)
        caption = self._load_text(index)
        label = self._load_label(index)

        # 添加噪声标签信息
        if self.is_train and self.noise_ratio > 0.0:
            is_noisy = torch.tensor(self.noisy_labels[index], dtype=torch.long)
            return image, caption, label, index
        else:
            return image, caption, label, index

    def get_all_label(self):
        labels = torch.zeros([self.__length, len(self.labels[0])], dtype=torch.float32)
        for i, item in enumerate(self.labels):
            labels[i] = torch.from_numpy(item)
        return labels

    def get_noisy_info(self):
        """获取噪声信息"""
        if self.is_train and self.noise_ratio > 0.0:
            return self.noisy_mapping, self.noisy_labels
        return None, None


class BaseDataset2(Dataset):
    def __init__(self,
                 captions: dict,
                 indexs: dict,
                 labels: dict,
                 is_train=True,
                 tokenizer=Tokenizer(),
                 maxWords=32,
                 imageResolution=224,
                 noise_ratio=0.0,  # 添加噪声比例参数
                 npy=False):

        self.captions = captions
        self.indexs = indexs
        self.labels = labels
        self.npy = npy
        self.is_train = is_train
        self.noise_ratio = noise_ratio

        self.maxWords = maxWords
        self.tokenizer = tokenizer

        self.transform = Compose([
            Resize(imageResolution, interpolation=Image.BICUBIC),
            CenterCrop(imageResolution),
            ToTensor(),
            Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)),
        ]) if is_train else Compose([
            Resize((imageResolution, imageResolution), interpolation=Image.BICUBIC),
            ToTensor(),
            Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)),
        ])
        self.SPECIAL_TOKEN = {"CLS_TOKEN": "<|startoftext|>", "SEP_TOKEN": "<|endoftext|>",
                              "MASK_TOKEN": "[MASK]", "UNK_TOKEN": "[UNK]", "PAD_TOKEN": "[PAD]"}

        self.__length = len(self.indexs)

        # 初始化文本噪声映射
        self.text_noisy_mapping = np.arange(self.__length)
        self.noisy_labels = None  # 用于存储噪声标签

        # 只在训练集且噪声比例大于0时生成噪声
        if is_train and noise_ratio > 0.0:
            self._generate_asymmetric_text_noise()

    def _generate_asymmetric_text_noise(self):
        """生成非对称文本噪声（图像不变，文本错误匹配）"""
        np.random.seed(42)  # 固定随机种子以保证可复现性

        # 1. 选择噪声样本
        num_samples = self.__length
        num_noisy = int(self.noise_ratio * num_samples)
        noisy_indices = np.random.choice(num_samples, num_noisy, replace=False)

        # 2. 创建文本置换映射（非对称噪声）
        # 对噪声样本的文本索引进行重排
        shuffled_indices = np.array(noisy_indices)
        np.random.shuffle(shuffled_indices)

        # 3. 创建映射关系：原始索引 -> 错误匹配的文本索引
        self.text_noisy_mapping[noisy_indices] = shuffled_indices

        # 4. 保存噪声标签信息（用于可能的噪声感知训练）
        self.noisy_labels = np.zeros(num_samples, dtype=np.int64)
        self.noisy_labels[noisy_indices] = 1  # 标记噪声样本

        print(f"Generated asymmetric text noise with ratio: {self.noise_ratio}")
        print(f"Noisy samples: {num_noisy}/{num_samples}")
        print(f"Noisy indices example: {noisy_indices[:10]}")
        print(f"Shuffled indices example: {shuffled_indices[:10]}")

    def __len__(self):
        return self.__length

    def _load_image(self, index: int) -> torch.Tensor:
        # 图像保持不变
        if not self.npy:
            image_path = self.indexs[index].strip()
            image = Image.open(image_path).convert("RGB")
        else:
            image = Image.fromarray(self.indexs[index]).convert("RGB")
        image = self.transform(image)

        return image

    def _load_text(self, index: int):
        # 如果有文本噪声映射，使用映射后的索引获取文本
        if self.is_train and self.noise_ratio > 0.0:
            text_index = self.text_noisy_mapping[index]
        else:
            text_index = index

        captions = self.captions[text_index]
        use_cap = captions[random.randint(0, len(captions) - 1)]

        words = self.tokenizer.tokenize(use_cap)
        words = [self.SPECIAL_TOKEN["CLS_TOKEN"]] + words
        total_length_with_CLS = self.maxWords - 1
        if len(words) > total_length_with_CLS:
            words = words[:total_length_with_CLS]

        words = words + [self.SPECIAL_TOKEN["SEP_TOKEN"]]
        caption = self.tokenizer.convert_tokens_to_ids(words)

        while len(caption) < self.maxWords:
            caption.append(0)
        caption = torch.tensor(caption)

        return caption

    def _load_label(self, index: int) -> torch.Tensor:
        # 标签保持不变，对应原始图像
        label = self.labels[index]
        label = torch.from_numpy(label)

        return label

    def __getitem__(self, index):
        image = self._load_image(index)
        caption = self._load_text(index)
        label = self._load_label(index)

        # 添加噪声标签信息
        if self.is_train and self.noise_ratio > 0.0:
            is_noisy = torch.tensor(self.noisy_labels[index], dtype=torch.long)
            return image, caption, label, index, is_noisy
        else:
            return image, caption, label, index

    def get_all_label(self):
        labels = torch.zeros([self.__length, len(self.labels[0])], dtype=torch.float32)
        for i, item in enumerate(self.labels):
            labels[i] = torch.from_numpy(item)
        return labels

    def get_noisy_info(self):
        """获取噪声信息"""
        if self.is_train and self.noise_ratio > 0.0:
            return self.text_noisy_mapping, self.noisy_labels
        return None, None

    def get_correct_pairs(self):
        """获取正确的图文对（用于分析）"""
        correct_pairs = []
        for i in range(self.__length):
            if self.noisy_labels is None or self.noisy_labels[i] == 0:
                correct_pairs.append(i)
        return correct_pairs


class BaseDataset3(Dataset):
    def __init__(self,
                 captions: dict,
                 indexs: dict,
                 labels: dict,
                 is_train=True,
                 tokenizer=Tokenizer(),
                 maxWords=32,
                 imageResolution=224,
                 npy=False,
                 noise_ratio=0.0,  # 噪声比例
                 noise_file_path=None):  # 噪声文件路径

        self.captions = captions
        self.indexs = indexs
        self.labels = labels
        self.npy = npy
        self.is_train = is_train
        self.noise_ratio = noise_ratio
        self.noise_file_path = noise_file_path

        self.maxWords = maxWords
        self.tokenizer = tokenizer

        self.transform = Compose([
            Resize(imageResolution, interpolation=Image.BICUBIC),
            CenterCrop(imageResolution),
            ToTensor(),
            Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)),
        ]) if is_train else Compose([
            Resize((imageResolution, imageResolution), interpolation=Image.BICUBIC),
            ToTensor(),
            Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)),
        ])
        self.SPECIAL_TOKEN = {"CLS_TOKEN": "<|startoftext|>", "SEP_TOKEN": "<|endoftext|>",
                              "MASK_TOKEN": "[MASK]", "UNK_TOKEN": "[UNK]", "PAD_TOKEN": "[PAD]"}

        self.__length = len(self.indexs)

        # 初始化噪声索引映射
        self.noisy_text_index = np.arange(self.__length)
        self.noisy_labels = None  # 用于存储噪声标签

        # 只在训练集且噪声比例大于0时生成噪声
        if is_train and noise_ratio > 0.0:
            self._setup_asymmetric_noise()

    def _setup_asymmetric_noise(self):
        if self.noise_file_path and os.path.exists(self.noise_file_path):
            self.noisy_text_index = np.loadtxt(self.noise_file_path, dtype=np.int64)
            print(f'=> Loaded noisy index from {self.noise_file_path}')
        else:
            n_samples = self.__length
            original_index = np.arange(n_samples)
            noise_len = int(self.noise_ratio * n_samples)

            if noise_len > 0:
                all_indices = np.arange(n_samples)
                np.random.shuffle(all_indices)
                noise_part = all_indices[:noise_len]

                self.noisy_text_index = original_index.copy()
                shuffled = original_index[noise_part].copy()
                np.random.shuffle(shuffled)
                self.noisy_text_index[noise_part] = shuffled

                if self.noise_file_path:
                    os.makedirs(os.path.dirname(self.noise_file_path), exist_ok=True)
                    np.savetxt(self.noise_file_path, self.noisy_text_index, fmt='%d')
                    print(f'=> Saved noisy index to {self.noise_file_path}')

            print(f'=> Generated asymmetric noise with ratio: {self.noise_ratio}')
            print(f'=> Noisy samples: {noise_len}/{n_samples}')

        # 创建噪声标签（1表示噪声，0表示干净）
        self.noisy_labels = np.zeros(self.__length, dtype=np.int64)
        for i in range(self.__length):
            if self.noisy_text_index[i] != i:  # 如果文本索引发生变化，则是噪声样本
                self.noisy_labels[i] = 1

    def __len__(self):
        return self.__length

    def _load_image(self, index: int) -> torch.Tensor:
        # 图像保持不变
        if not self.npy:
            image_path = self.indexs[index].strip()
            image = Image.open(image_path).convert("RGB")
        else:
            image = Image.fromarray(self.indexs[index]).convert("RGB")
        image = self.transform(image)

        return image

    def _load_text(self, index: int):
        # 如果有噪声映射，使用映射后的索引获取文本
        if self.is_train and self.noise_ratio > 0.0:
            text_index = self.noisy_text_index[index]
        else:
            text_index = index

        # 获取文本（支持多caption选择）
        if isinstance(self.captions[text_index], np.ndarray):
            captions_list = self.captions[text_index].tolist()
        else:
            captions_list = [self.captions[text_index]]

        # 随机选择一个caption（如果有多个）
        if len(captions_list) > 1:
            use_cap = captions_list[random.randint(0, len(captions_list) - 1)]
        else:
            use_cap = captions_list[0]

        # 处理文本
        words = self.tokenizer.tokenize(str(use_cap))
        words = [self.SPECIAL_TOKEN["CLS_TOKEN"]] + words
        total_length_with_CLS = self.maxWords - 1
        if len(words) > total_length_with_CLS:
            words = words[:total_length_with_CLS]

        words = words + [self.SPECIAL_TOKEN["SEP_TOKEN"]]
        caption = self.tokenizer.convert_tokens_to_ids(words)

        while len(caption) < self.maxWords:
            caption.append(0)
        caption = torch.tensor(caption)

        return caption

    def _load_label(self, index: int) -> torch.Tensor:
        # 标签保持不变，对应原始图像
        label = self.labels[index]
        label = torch.from_numpy(label)

        return label

    def __getitem__(self, index):
        image = self._load_image(index)
        caption = self._load_text(index)
        label = self._load_label(index)

        # 如果训练且有噪声，返回噪声标签
        if self.is_train and self.noise_ratio > 0.0:
            is_noisy = torch.tensor(self.noisy_labels[index], dtype=torch.long)
            return image, caption, label, index, is_noisy
        else:
            return image, caption, label, index

    def get_all_label(self):
        labels = torch.zeros([self.__length, len(self.labels[0])], dtype=torch.float32)
        for i, item in enumerate(self.labels):
            labels[i] = torch.from_numpy(item)
        return labels

    def get_noise_info(self):
        """获取噪声信息"""
        if self.is_train and self.noise_ratio > 0.0:
            noise_stats = {
                'noise_ratio': self.noise_ratio,
                'noisy_samples': np.sum(self.noisy_labels),
                'total_samples': self.__length,
                'noisy_text_index': self.noisy_text_index,
                'noisy_labels': self.noisy_labels
            }
            return noise_stats
        return None
