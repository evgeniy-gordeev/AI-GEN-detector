# os
import warnings
import time
import tqdm
import random
import string
import re
import csv
from pathlib import Path
from tqdm import tqdm
import argparse
import glob
import os
import json
import logging
from itertools import chain
from string import punctuation
import sentencepiece
from random import shuffle

# модели
import torch
from parrot import Parrot
from transformers import BartForConditionalGeneration, BartTokenizer
from transformers import T5ForConditionalGeneration, T5Tokenizer
from transformers import GPT2LMHeadModel, GPT2Tokenizer
from transformers import MarianMTModel, MarianTokenizer
from concurrent.futures import ThreadPoolExecutor, as_completed
from transformers import AutoModelForSequenceClassification, BertTokenizer
from transformers import PegasusForConditionalGeneration, PegasusTokenizer

# метрики
import nltk
from nltk.translate.bleu_score import sentence_bleu
#from datasets import load_metric
from rouge_score import rouge_scorer
from nltk.corpus import wordnet
#from nltk.translate.ter_score import ter_score

warnings.filterwarnings("ignore")
import ssl
try:
    _create_unverified_https_context = ssl._create_unverified_context
except AttributeError:
    pass
else:
    ssl._create_default_https_context = _create_unverified_https_context


nltk.download('punkt', )
nltk.download('wordnet', )
nltk.download('all', )



# Знаки перпинания
def modify_syntax(text):
    modified_text = text

    # Генерация случайных порогов вероятности
    add_spaces_around_punctuation = random.uniform(0.74, 0.9)
    remove_extra_punctuation = random.uniform(0.6, 0.8)
    replace_question_exclamation = random.uniform(0.5, 0.74)
    normalize_ellipsis = random.uniform(0.7, 0.9)
    add_spaces_around_non_alphanumeric = random.uniform(0.75, 1.0)

    # Добавление пробелов вокруг знаков препинания
    if random.random() < add_spaces_around_punctuation:
        modified_text = re.sub(r'([,.\?!])', r' \1 ', modified_text)
        if random.random() < 0.5:
            modified_text = re.sub(r' +', ' ', modified_text)

    # Удаление лишних знаков препинания
    if random.random() < remove_extra_punctuation:
        modified_text = re.sub(r'[,\.]+(?=[\)\]"])', '', modified_text)
        if random.random() < 0.3:
            modified_text = re.sub(r'[,\.]+', random.choice([',', '.', '']), modified_text)

    # Замена вопросительного знака с восклицательным
    if random.random() < replace_question_exclamation:
        modified_text = re.sub(r'\?!', random.choice(['?', '!']), modified_text)

    # Нормализация использования многоточий
    if random.random() < normalize_ellipsis:
        modified_text = re.sub(r'\.{2,}', '...', modified_text)
        if random.random() < 0.4:
            modified_text = re.sub(r'\.{3}', random.choice(['...', '...']), modified_text)

    # Добавление пробелов вокруг непечатных символов
    if random.random() < add_spaces_around_non_alphanumeric:
        modified_text = re.sub(r'([^a-zA-Z0-9])', r' \1 ', modified_text)
        if random.random() < 0.6:
            modified_text = re.sub(r' +', ' ', modified_text)

    # Удаление лишних пробелов
    modified_text = re.sub(r'\s+', ' ', modified_text)

    return modified_text

# Изменениие длины предложений
import re
def modify_sentence_length(text):
    sentences = re.split(r'[.!?]+', text)

    modified_sentences = []
    for sentence in sentences:
        if len(sentence.split()) > 20:
            short_sentences = re.split(r'[,;]+', sentence)
            modified_sentences.extend(short_sentences)
        else:
            if len(modified_sentences) > 0 and len(modified_sentences[-1].split()) < 10:
                modified_sentences[-1] += ' ' + sentence.strip()
            else:
                modified_sentences.append(sentence.strip())

    modified_text = '. '.join(modified_sentences) + '.'

    return modified_text

# Перестановка слов
def modify_word_order(text):
    sentences = re.split(r'[.!?]+', text)

    modified_sentences = []
    for sentence in sentences:
        words = sentence.split()

        if len(words) > 2:
            middle_words = words[1:-1]
            shuffle(middle_words)

            modified_sentence = ' '.join([words[0]] + middle_words + [words[-1]])
        else:
            modified_sentence = ' '.join(words)

        modified_sentences.append(modified_sentence)

    modified_text = '. '.join(modified_sentences) + '.'

    return modified_text

# Замена пассивных конструкций на активные
def activate_passive_sentences(text):
    active_text = re.sub(r'(\w+) was (\w+)', r'\2 \1', text)
    return active_text

# Добавление эмоциональных/оценочных элементов
def add_emotional_elements(text):
    emotional_words = [
        'весело', 'радостно', 'волнушкой', 'смеяться',
        'возбужденно', 'вдохновляюще', 'интригующе', 'взрывно',
        'трепетом', 'волнением', 'страстно', 'горячо', 'влюбленно'
    ]

    for word in emotional_words:
        text = re.sub(r'\bкроме того\b', f'{word} ', text)

    return text

# Изменение стиля изложения на более разговорный
def change_style_to_conversational(text):
    conversational_phrases = {
        'формальный': 'разговорный',
        'предложения': 'бабочки',
        'легче': 'просто',
        'воспринимаются': 'понимаются',
        'включение': 'добавление',
        'живым': 'активным',
        'выразительным': 'выражительным',
        'кроме того': 'еще',
        'также': 'а еще',
        'дополнительно': 'и так далее',
        'конечно': 'ну конечно',
        'действительно': 'правильно',
        'по сути': 'в общем',
        'в основном': 'в основном'
    }

    for key, value in conversational_phrases.items():
        text = text.replace(key, value)

    return text

# Случайные перестановки букв в словах
def scramble_words_with_control(text, probability=0.1):
    words = re.findall(r'\b\w+\b', text)
    for i, word in enumerate(words):
        if random.random() < probability:
            scrambled_word = ''.join(random.sample(word, len(word)))
            text = text.replace(word, scrambled_word)
    return text



# Замена на синонимы
def find_synonyms(word):
    synonyms = []
    for syn in wordnet.synsets(word):
        for lemma in syn.lemmas():
            synonyms.append(lemma.name())
    return synonyms

def replace_synonyms(text):
    words = text.split()
    new_words = []
    for word in words:
        if word.lower() in wordnet.words():
            synonyms = find_synonyms(word)
            if len(synonyms) > 0:
                new_word = random.choice(synonyms)
                new_words.append(new_word)
            else:
                new_words.append(word)
        else:
            new_words.append(word)
    return ' '.join(new_words)

