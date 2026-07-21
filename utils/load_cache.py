import functools
import json
import os

import torch

from configs import Config
from loguru import logger

import torchvision
from PIL import Image, ImageFile
from torch.utils.data import DataLoader, Dataset, TensorDataset

ImageFile.LOAD_TRUNCATED_IMAGES = True


class AddGaussianNoise:
    """Ruído leve aplicado só no tensor de treino, depois da normalização."""

    def __init__(self, std: float = 0.04):
        self.std = std

    def __call__(self, tensor):
        return tensor + torch.randn_like(tensor) * self.std


class LoadCache(Dataset):
    def __init__(self, cache_path: str, path: str, word: bool, image_channel: int, resize: list, charset: list):
        self.cache_path = cache_path
        self.path = path
        self.word = word
        self.ImageChannel = image_channel
        self.resize = resize
        self.charset = charset
        self.caches = []
        logger.info("\nReading Cache File... ----> {}".format(self.cache_path))

        with open(self.cache_path, 'r', encoding='utf-8') as f:
            self.caches = f.readlines()
        self.caches_num = len(self.caches)
        logger.info("\nRead Cache File End! Caches Num is {}.".format(self.caches_num))

    def __len__(self):
        return self.caches_num

    def __getitem__(self, idx):
        try:
            data = self.caches[idx]
            data = data.replace("\r", "").replace("\n", "").split("\t")
            image_name = data[0]
            image_label = data[1]
            image_path = os.path.join(self.path, image_name)
            if not self.word:
                image_label = list(image_label)
            else:
                image_label = [image_label]
            if self.ImageChannel == 1:
                mode = "L"
            else:
                mode = "RGB"
            image = Image.open(image_path).convert(mode)  # shape c, h, w
            image_shape = image.size
            image_height = image_shape[1]
            image_width = image_shape[0]
            width = self.resize[0]
            height = self.resize[1]
            if self.resize[0] == -1:
                if self.word:
                    image = image.resize((height, height))
                else:
                    image = image.resize((int(image_width * (height / image_height)), height))
            else:
                image = image.resize((width, height))
            label = [int(self.charset.index(item)) for item in list(image_label)]
            return image, label

        except Exception as e:
            logger.error("\nError: {}, File: {}".format(str(e), self.caches[idx].split("\t")[0]))
            return None, None


class GetLoader:
    def __init__(self, project_name: str):
        self.project_name = project_name
        self.project_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "projects",
                                         project_name)
        if os.path.exists(self.project_path):
            self.cache_path = os.path.join(self.project_path, "cache")
            if os.path.exists(self.cache_path):
                self.cache_train_path = os.path.join(self.cache_path, "cache.train.tmp")
                self.cache_val_path = os.path.join(self.cache_path, "cache.val.tmp")

                if not os.path.exists(self.cache_train_path):
                    logger.error("\nCache Train File {} is not exists!".format(self.cache_train_path))
                    exit()
                if not os.path.exists(self.cache_val_path):
                    logger.error("\nCache Val File {} is not exists!".format(self.cache_val_path))
                    exit()

            else:
                logger.error("\nCache dir {} is not exists!".format(self.cache_path))
                exit()
        else:
            logger.error("\nProject {} is not exists!".format(project_name))
            exit()

        self.config = Config(project_name)

        self.conf = self.config.load_config()

        self.charset = self.conf['Model']['CharSet']

        logger.info("\nCharsets is {}".format(json.dumps(self.charset, ensure_ascii=False)))

        self.resize = [int(self.conf['Model']['ImageWidth']), int(self.conf['Model']['ImageHeight'])]

        logger.info("\nImage Resize is {}".format(json.dumps(self.resize)))

        self.ImageChannel = self.conf['Model']['ImageChannel']

        self.word = self.conf['Model']['Word']

        self.path = self.conf['System']['Path']

        self.batch_size = self.conf['Train']['BATCH_SIZE']

        self.val_batch_size = self.conf['Train']['TEST_BATCH_SIZE']

        logger.info("\nImage Path is {}".format(self.path))

        norm_list = []
        if self.ImageChannel == 1:
            norm_list.append(torchvision.transforms.Normalize(mean=[0.456], std=[0.224]))
        else:
            if self.ImageChannel != 3:
                logger.error("ImageChannel must be 1 or 3!")
                exit()
            norm_list.append(torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                               std=[0.229, 0.224, 0.225]))

        # Transform de validação: só ToTensor + Normalize, sem augmentação,
        # pra medir generalização real (sem "facilitar" a vida do modelo).
        self.val_transform = torchvision.transforms.Compose(
            [torchvision.transforms.ToTensor()] + norm_list
        )

        # Augmentação aplicada na imagem PIL, antes do padding (treino apenas).
        # Rotação + shear/translação leve + jitter de brilho/contraste + blur
        # ocasional simulam a variação real de captchas sem distorcer demais
        # o caractere a ponto de virar outra letra.
        self.pil_augment = torchvision.transforms.Compose([
            torchvision.transforms.RandomRotation(degrees=6, fill=255),
            torchvision.transforms.RandomAffine(degrees=0, translate=(0.02, 0.05),
                                                shear=6, fill=255),
            torchvision.transforms.ColorJitter(brightness=0.35, contrast=0.35),
            torchvision.transforms.RandomApply(
                [torchvision.transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 1.2))], p=0.35
            ),
        ])

        # Transform de treino: ToTensor + Normalize + ruído leve + apagamento
        # aleatório de um pedaço pequeno da imagem (RandomErasing), pra evitar
        # que a rede dependa demais de detalhes pixel-a-pixel específicos.
        self.train_transform = torchvision.transforms.Compose(
            [torchvision.transforms.ToTensor()] + norm_list + [
                AddGaussianNoise(std=0.04),
                torchvision.transforms.RandomErasing(p=0.25, scale=(0.02, 0.06), ratio=(0.3, 3.3), value=0),
            ]
        )

        train_loader = LoadCache(self.cache_train_path, self.path, self.word, self.ImageChannel, self.resize,
                                 self.charset)
        if len(train_loader) < self.batch_size:
            self.batch_size = len(train_loader)
        val_loader = LoadCache(self.cache_val_path, self.path, self.word, self.ImageChannel, self.resize, self.charset)
        if len(val_loader) < self.batch_size:
            self.val_batch_size = len(val_loader)
        self.loaders = {
            'train': DataLoader(dataset=train_loader, batch_size=self.batch_size, shuffle=True, drop_last=True,
                                num_workers=0,
                                collate_fn=functools.partial(self.collate_to_sparse, augment=True)),
            'val': DataLoader(dataset=val_loader, batch_size=self.val_batch_size, shuffle=True, drop_last=True,
                              num_workers=0,
                              collate_fn=functools.partial(self.collate_to_sparse, augment=False)),
        }
        del val_loader
        del train_loader

    def collate_to_sparse(self, batch, augment: bool = False):
        values = []
        images = []
        shapes = []
        max_width = 0
        for n, (img, seq) in enumerate(batch):
            if img is None or seq is None:
                continue
            if len(seq) == 0: continue
            if augment:
                img = self.pil_augment(img)
            if max_width < img.size[0]:
                max_width = img.size[0]
            values.extend(seq)
            images.append(img)
            shapes.append(len(seq))
        transform = self.train_transform if augment else self.val_transform
        images_pad = []
        for img in images:
            img = torchvision.transforms.Pad((0, 0, int(max_width - img.size[0]), 0))(img)
            if transform is not None:
                img = transform(img)
            images_pad.append(img)
        images_pad = torch.stack(images_pad, dim=0)
        return [images_pad, torch.FloatTensor(values), torch.IntTensor(shapes)]