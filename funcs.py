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
#from parrot import Parrot
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