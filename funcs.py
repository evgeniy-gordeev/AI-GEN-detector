#os
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


# модели
import torch
from parrot import Parrot
from transformers import BartForConditionalGeneration, BartTokenizer
from transformers import T5ForConditionalGeneration, T5Tokenizer
from transformers import GPT2LMHeadModel, GPT2Tokenizer
from transformers import MarianMTModel, MarianTokenizer
from concurrent.futures import ThreadPoolExecutor, as_completed

# метрики
from nltk.translate.bleu_score import sentence_bleu
from datasets import load_metric
import nltk
#from nltk.translate.ter_score import ter_score
import sentencepiece

import torch
from transformers import pipeline

warnings.filterwarnings("ignore")
nltk.download('punkt')
nltk.download('wordnet')


def set_seed(seed):
  torch.manual_seed(seed)
  if torch.cuda.is_available():
    torch.cuda.manual_seed_all(seed)

def create_paraphrase_T5(full_text, max_length, num_return_sequences, early_stopping):

    model_1 = "t5-base"
    model_2 = "t5-large"

    model = T5ForConditionalGeneration.from_pretrained(model_2)
    tokenizer = T5Tokenizer.from_pretrained(model_2)

    input_text = full_text
    input_ids = tokenizer.encode(input_text, return_tensors='pt')

    output_ids = model.generate(input_ids,
                                max_length=max_length,
                                num_return_sequences=num_return_sequences,
                                early_stopping=early_stopping)
    t5_output_text = tokenizer.decode(output_ids[0],
                                      skip_special_tokens=True)

    # Вычисление BLEU
    bleu_score = sentence_bleu([full_text.split()], t5_output_text.split())
    print(f"BLEU Score: {bleu_score:.2f}")

    return t5_output_text

def create_paraphrase_bart(full_text, max_length, num_return_sequences, early_stopping):

    model_1 = "facebook/bart-base"
    model_2 = "facebook/bart-large"

    model = BartForConditionalGeneration.from_pretrained(model_2)
    tokenizer = BartTokenizer.from_pretrained(model_2)

    input_text = full_text
    input_ids = tokenizer.encode(input_text, return_tensors='pt')

    output_ids = model.generate(input_ids,
                                max_length=max_length,
                                num_return_sequences=num_return_sequences,
                                early_stopping=early_stopping)
    output_text_bart = tokenizer.decode(output_ids[0], skip_special_tokens=True)

    # Оценка качества перефразирования
    bleu_score = sentence_bleu([full_text.split()], output_text_bart.split())
    print(f"BLEU Score: {bleu_score:.2f}")

    print(f"Перефразированный текст: {output_text_bart}")

    return output_text_bart

def create_paraphrase(full_text):
    # Использование модели T5 для перефразирования
    model = T5ForConditionalGeneration.from_pretrained("t5-base")
    tokenizer = T5Tokenizer.from_pretrained("t5-base")

    input_text = full_text
    input_ids = tokenizer.encode(input_text, return_tensors='pt')

    output_ids = model.generate(input_ids, max_length=100, num_return_sequences=1, early_stopping=True)
    t5_output_text = tokenizer.decode(output_ids[0], skip_special_tokens=True)

    # Использование библиотеки Parrot для дополнительного перефразирования
    parrot = Parrot(model_tag="prithivida/parrot_paraphraser_on_T5", use_gpu=True)

    phrases = [x.strip() for x in full_text.split('.')]
    output_phrases = []

    for phrase in phrases:
        if len(phrase) > 1:
            para_phrases = parrot.augment(input_phrase=phrase,
                                          use_gpu=False,
                                          diversity_ranker="levenshtein",
                                          do_diverse=False,
                                          max_return_phrases=10,
                                          max_length=32,
                                          adequacy_threshold=0.99,
                                          fluency_threshold=0.90)

            try:
                for para_phrase in para_phrases:
                    (x, y) = para_phrase
                    x = x[0].upper() + x[1:] # capitalize
                    output_phrases.append(x)
                    break # just get the first phrase
            except:
                print("Exception occurred with this one.")

    parrot_output_text = ".".join(output_phrases)

    # Оценка качества перефразирования
    bleu_score = sentence_bleu([full_text.split()], parrot_output_text.split())
    print(f"BLEU Score: {bleu_score:.2f}")

    return t5_output_text, parrot_output_text


def is_generated_by_ai(text):
    if text == '':
        return ""
    else:
        text_classifier = pipeline("text-classification", model="Juner/AI-generated-text-detection")
        result = text_classifier(text)[0]
        if result['label'] == "LABEL_1":
            result['label'] = "Human-generated"
            result['score'] = 1 - result['score']
        else:
            result['label'] =  "Human-generated"
            result['score'] = result['score']
        return result
    
# Функция для перефразирования текста
def paraphrase(text):

    # Загрузка моделей машинного перевода
    src_model = MarianMTModel.from_pretrained("Helsinki-NLP/opus-mt-en-ru")
    src_tokenizer = MarianTokenizer.from_pretrained("Helsinki-NLP/opus-mt-en-ru")

    tgt_model = MarianMTModel.from_pretrained("Helsinki-NLP/opus-mt-ru-en")
    tgt_tokenizer = MarianTokenizer.from_pretrained("Helsinki-NLP/opus-mt-ru-en")

    # Перевод текста с английского на русский
    input_ids = src_tokenizer.encode(text, return_tensors="pt")
    output_ids = src_model.generate(input_ids, max_length=100, num_beams=4, early_stopping=True)
    paraphrased_text_ru = src_tokenizer.decode(output_ids[0], skip_special_tokens=True)

    # Перевод обратно с русского на английский
    input_ids = tgt_tokenizer.encode(paraphrased_text_ru, return_tensors="pt")
    output_ids = tgt_model.generate(input_ids, max_length=100, num_beams=4, early_stopping=True)
    paraphrased_text = tgt_tokenizer.decode(output_ids[0], skip_special_tokens=True)

    return paraphrased_text

def russian_paraphrase(text):
    MODEL_NAME = 'cointegrated/rut5-base-paraphraser'
    model = T5ForConditionalGeneration.from_pretrained(MODEL_NAME)
    tokenizer = T5Tokenizer.from_pretrained(MODEL_NAME)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()

    x = tokenizer(text, return_tensors='pt', padding=True).to(device)
    max_size = int(x.input_ids.shape[1] * 1.5 + 10)
    out = model.generate(**x, encoder_no_repeat_ngram_size=4, num_beams=5, max_length=max_size, do_sample=False)
    
    paraphrased_text = tokenizer.decode(out[0], skip_special_tokens=True)
    return paraphrased_text