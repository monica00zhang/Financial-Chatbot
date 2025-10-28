"""
Model Utilities for T5 Fine-tuning
Comprehensive utilities for model loading, weight mapping, and common operations
"""

import torch
import torch.nn as nn
import os
import json
import numpy as np
from typing import Dict, List, Tuple, Optional, Union
from pathlib import Path
import shutil
from datetime import datetime

from transformers import T5ForConditionalGeneration, T5Tokenizer, T5Config as HFT5Config

import random
import logging

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def set_seed(seed: int = 42):
    # set random seeds
    
    torch.manual_seed(seed)
    np.random.seed(seed)
        
    logger.info(f"Random seed set to {seed}")

""" ================                 Model          ================ """                
def load_tokenizer(config):
    # tokenzier for encoding the text
    return T5Tokenizer.from_pretrained(config.device)

def load_model(config):
    model = T5ForConditionalGeneration.from_pretrained(config.model)
    model = model.to(config.device)
    return model



""" ================                DataSet         ================ """      
def load_data(data_path):

    df = pd.read_csv(data_path)
    dataset = []

    for i in range(df.shape[0]):
        ques, ans = df.iloc[i]["question"], df.iloc[i]["answer"]
        dataset.append(ques, ans))

    return dataset



class DataSetClass(Dataset):
  """
  Creating a custom dataset for reading the dataset and 
  loading it into the dataloader to pass it to the neural network for finetuning the model

  """

  def __init__(self, dataframe, tokenizer, source_len, target_len, source_text, target_text):
    self.tokenizer = tokenizer
    self.data = dataframe
    self.source_len = source_len
    self.summ_len = target_len
    self.target_text = self.data[target_text]
    self.source_text = self.data[source_text]

  def __len__(self):
    return len(self.target_text)

  def __getitem__(self, index):
    source_text = str(self.source_text[index])
    target_text = str(self.target_text[index])

    #cleaning data so as to ensure data is in string type
    source_text = ' '.join(source_text.split())
    target_text = ' '.join(target_text.split())

    source = self.tokenizer.batch_encode_plus([source_text], max_length= self.source_len, pad_to_max_length=True, truncation=True, padding="max_length", return_tensors='pt')
    target = self.tokenizer.batch_encode_plus([target_text], max_length= self.summ_len, pad_to_max_length=True, truncation=True, padding="max_length", return_tensors='pt')

    source_ids = source['input_ids'].squeeze()
    source_mask = source['attention_mask'].squeeze()
    target_ids = target['input_ids'].squeeze()
    target_mask = target['attention_mask'].squeeze()

    return {
        'source_ids': source_ids.to(dtype=torch.long), 
        'source_mask': source_mask.to(dtype=torch.long), 
        'target_ids': target_ids.to(dtype=torch.long),
        'target_ids_y': target_ids.to(dtype=torch.long)
    }
     


       

