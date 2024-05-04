import os

from math import ceil

import pandas as pd
import h5py
import re
from io import BytesIO

import requests
import torch
from PIL import Image
from torch import zeros
from torchvision import transforms

data_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data')


def preprocess_text(text):
    """
        Очищает текст от ссылок, эмодзи, пунктуации, цифр и лишних пробелов.

        Args:
            text (str): Сырой текст.

        Returns:
            str: Очищенный текст.

    """
    if text:
        text_no_url = re.sub(
            r'https?:\/\/(www\.)?[-a-zA-Z0-9@:%._\+~#=]{1,256}\.[a-zA-Z0-9()]{1,6}\b([-a-zA-Z0-9()@:%_\+.~#?&//=]*)',
            '',
            text)
        text_no_emojis = re.sub('[✔★’✅]', ' ', text_no_url)
        text_no_emojis = re.sub(r'\\x\S\S', ' ', text_no_emojis)
        text_no_punct = re.sub(r'[^\w\s]', '', text_no_emojis)
        text_no_digits = ''.join(i for i in text_no_punct if not i.isdigit())
        text_no_spaces = re.sub(r' +', ' ', text_no_digits)
        return text_no_spaces.lower()
    return ''


def preprocess_url(url):
    """
        Очищает URL от пробелов, заменяя их '%20'.

        Args:
            url (str): URL

        Returns:
            str: Обработанный URL.

    """
    return re.sub(' ', '%20', url)


def preprocess_img(img_src) -> torch.Tensor:
    """
        Обрабатывает изображение: меняет размер, обрезает, нормализует и конвертирует в torch tensor.

        Args:
            img_src (str): URL изображения.

        Returns:
            Tensor: Обработанный тензор изображения.

    """
    if img_src:
        response = requests.get(preprocess_url(img_src))
        img = Image.open(BytesIO(response.content)).convert('RGB')

        transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])
        return transform(img).unsqueeze(0)
    return zeros(1, 3, 224, 224)


def write_embeddings(recommender, asset_data):
    asset_id = asset_data['id']
    file_path = os.path.join(data_dir, f"embeddings_all.h5")
    mode = 'a' if os.path.exists(file_path) else 'w'
    with h5py.File(file_path, mode) as f:
        group = f.create_group(f'asset_{asset_id}')
        desc_embedding = group.create_dataset('desc_embedding', data=recommender.get_text_emb(asset_data['description']))
        tech_details_embedding = group.create_dataset('tech_details_embedding',
                                                      data=recommender.get_text_emb(asset_data['tech_details']))
        tags_embedding = group.create_dataset('tags_embedding', data=recommender.get_tags_emb(asset_data['tags']))

        if isinstance(asset_data['images_link'], list):
            img_embedding = group.create_dataset('img_embedding', data=recommender.get_img_emb(asset_data['images_link'][0]))
        elif isinstance(asset_data['images_link'], str):
            img_embedding = group.create_dataset('img_embedding', data=recommender.get_img_emb(asset_data['images_link']))
        else:
            print(f"Invalid type for asset.images_link: {type(asset_data['images_link'])}")



def read_embeddings(id_: int) -> dict:
    """
    Читает эмбеддинги из файла.
    Args:
        file: Путь до HDF5 файла.

    Returns:
        Словарь с эмбеддингами описания, технических деталей, тегов и изображения.
    """
    with h5py.File(os.path.join(data_dir, f"embeddings_all.h5"), 'r') as f:
        # access the group for the asset
        asset_group_name = f"asset_{id_}"
        group = f[asset_group_name]
        # get the description embedding dataset
        return {
            'desc_embedding': group['desc_embedding'][()],
            'tech_details_embedding': group['tech_details_embedding'][()],
            'tags_embedding': group['tags_embedding'][()],
            'img_embedding': group['img_embedding'][()]
        }
