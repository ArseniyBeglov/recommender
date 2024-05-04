import torch
import torchvision.models as models
from utils import preprocess_text, preprocess_img
import numpy as np
from transformers import BertTokenizer, BertModel
from numpy.linalg import norm
import os


def __calc_distance(a, b):
    """
    Считает косинусное расстояние между векторами.
    Args:
        a: Вектор 1.
        b: Вектор 2.
    Returns:
    Косинусное расстояние от 0 до 1.
    """
    return np.dot(a, b) / (norm(a) * norm(b))


def calc_metric(a, b):
    """
    Считает похожесть ассетов используя их эмбеддинги.
    Args:
        a: Словарь эмбеддингов ассета 1.
        b: Словарь эмбеддингов ассета 2.
    Returns:
    Метрика похожести.
    """
    tags_sim = np.dot(a['tags_embedding'], b['tags_embedding']) / (
            norm(a['tags_embedding']) * norm(b['tags_embedding']))

    desc_sim = np.dot(a['desc_embedding'].reshape(-1), b['desc_embedding'].reshape(-1)) / (
            norm(a['desc_embedding']) * norm(b['desc_embedding']))

    tech_de_sim = np.dot(a['tech_details_embedding'].reshape(-1), b['tech_details_embedding'].reshape(-1)) / (
            norm(a['tech_details_embedding']) * norm(b['tech_details_embedding']))

    img_sim = (np.dot(a['img_embedding'], b['img_embedding'].reshape(-1, 1)) / (
            norm(a['img_embedding']) * norm(b['img_embedding'])))[0][0]

    metric = (tags_sim + desc_sim + tech_de_sim + img_sim) / 4
    return float(metric)


class Recommender:
    """
        Класс для рекомендательной системы.
    """

    def __init__(self):
        """
        Инициализирует EfficientNet  and BERT модели.
        """
        self.efficientnet = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.IMAGENET1K_V1)

        self.efficientnet.eval()

        self.tokenizer = BertTokenizer.from_pretrained('bert-base-multilingual-uncased')
        self.bert = BertModel.from_pretrained('bert-base-multilingual-uncased')

    def get_img_emb(self, url) -> np.array:
        """
        Обрабатывает изображение и считает эмбеддинг изображения по ссылке используя EfficientNet.
        Args:
            url: URL изображения.
        Returns:
            NumPy array эмбеддинг изображения.
        """
        i0 = preprocess_img(url)
        with torch.no_grad():
            o0 = self.efficientnet(i0)
        o0 = o0.detach().numpy()
        return o0

    def get_text_emb(self, text) -> np.array:
        """
        Обрабатывает текст и считает эмбеддинг для данного текста используя BERT.
        Args:
            text: Текст.
        Returns:
            NumPy array усредненный эмбеддинг слов в тексте.
        """
        text_prep = preprocess_text(text)
        input_ids = torch.tensor([self.tokenizer.encode(text_prep, add_special_tokens=True)])
        max_len = 512
        num_parts = (len(input_ids[0]) + max_len - 1) // max_len
        input_ids = input_ids.repeat(num_parts, 1)
        input_ids = input_ids.view(num_parts, -1)
        input_ids = input_ids[:, :max_len]
        with torch.no_grad():
            last_hidden_states = self.bert(input_ids)[0]
        emb = last_hidden_states.mean(dim=1).detach().numpy()
        return emb.mean(axis=0)

    def get_tags_emb(self, tags) -> np.array:
        """
        Считает эмбеддиго для данного набора тегов используя BERT.
        Args:
            tags: Список тегов.
        Returns:
            NumPy array усредненный эмбеддинг  тегов.
        """
        text_prep = preprocess_text(tags)
        input_ids = torch.tensor([self.tokenizer.encode(text_prep, add_special_tokens=True)])
        with torch.no_grad():
            last_hidden_states = self.bert(input_ids)[0]
        emb = last_hidden_states.mean(dim=1).detach().numpy()
        return emb.mean(axis=0)


recommender = Recommender()
