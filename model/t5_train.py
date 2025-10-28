import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

import pandas as pd
import os
import time
from datetime import datetime

from rich.console import Console
# define a rich console logger
console=Console(record=True)

from config import T5Config
from model_utils import (load_model,DataSetClass, set_seed)


def train_epoch(model, epoch, tokenizer, device, loader, optimizer):
    """
    Train for one epoch 
    """
    model.train()
    for _,data in enumerate(loader, 0):
        y = data['target_ids'].to(device, dtype = torch.long)
        y_ids = y[:, :-1].contiguous()
        lm_labels = y[:, 1:].clone().detach()
        lm_labels[y[:, 1:] == tokenizer.pad_token_id] = -100
        ids = data['source_ids'].to(device, dtype = torch.long)
        mask = data['source_mask'].to(device, dtype = torch.long)

        outputs = model(input_ids = ids, attention_mask = mask, decoder_input_ids=y_ids, labels=lm_labels)
        loss = outputs[0]

        if _%10==0:
        training_logger.add_row(str(epoch), str(_), str(loss))
        console.print(training_logger)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

def validate(model, epoch, tokenizer, device, loader):

    """
    Function to evaluate model for predictions
    """

    model.eval()
    predictions = []
    actuals = []
    with torch.no_grad():
        for _, data in enumerate(loader, 0):
            y = data['target_ids'].to(device, dtype = torch.long)
            ids = data['source_ids'].to(device, dtype = torch.long)
            mask = data['source_mask'].to(device, dtype = torch.long)

            generated_ids = model.generate(
                input_ids = ids,
                attention_mask = mask, 
                max_length=150, 
                num_beams=2,
                repetition_penalty=2.5, 
                length_penalty=1.0, 
                early_stopping=True
                )
            preds = [tokenizer.decode(g, skip_special_tokens=True, clean_up_tokenization_spaces=True) for g in generated_ids]
            target = [tokenizer.decode(t, skip_special_tokens=True, clean_up_tokenization_spaces=True)for t in y]
            if _%10==0:
                console.print(f'Completed {_}')

            predictions.extend(preds)
            actuals.extend(target)
    return predictions, actuals
     


def T5_trainer(dataset, config):
    """ T5 trainer"""

    # Set random seed for reproducibility
    set_seed(config)
 
    # logging
    console.log(f"""[Model]: Loading {config.MODEL}...\n""")

    # tokenzier for encoding the text
    tokenizer = load_tokenizer(config)

    # Defining the model. We are using t5-base model and added a Language model layer on top for generation. 
    # Further this model is sent to device (GPU/TPU) for using the hardware.
    model = load_model(config)

    
    # split dataset
    train_size = int(config.TRAIN_SIZE * len(dataset))
    train_dataset = dataset[:train_size]
    val_dataset = dataset[train_size:]

    # Creating the Training and Validation dataset for further creation of Dataloader
    training_set = YourDataSetClass(train_dataset, tokenizer, config.MAX_SOURCE_TEXT_LENGTH, config.MAX_TARGET_TEXT_LENGTH, source_text, target_text)
    val_set = YourDataSetClass(val_dataset, tokenizer, config.MAX_SOURCE_TEXT_LENGTH, config.MAX_TARGET_TEXT_LENGTH, source_text, target_text)


    # Defining the parameters for creation of dataloaders
    train_params = {
        'batch_size': config.BATCH_SIZE,
        'shuffle': True,
        'num_workers': 0
        }


    val_params = {
        'batch_size': config.BATCH_SIZE,
        'shuffle': False,
        'num_workers': 0
        }


    # Creation of Dataloaders for testing and validation. This will be used down for training and validation stage for the model.
    training_loader = DataLoader(training_set, **train_params)
    val_loader = DataLoader(val_set, **val_params)


    # Defining the optimizer that will be used to tune the weights of the network in the training session. 
    optimizer = torch.optim.Adam(params =  model.parameters(), lr=config.LEARNING_RATE)


    # Training loop
    console.log(f'[Initiating Fine Tuning]...\n')

    for epoch in range(config.EPOCHS):
        train(epoch, tokenizer, model, device, training_loader, optimizer)
        
    console.log(f"[Saving Model]...\n")

    #Saving the model after training
    path = os.path.join(config.output_dir, "model_files")
    model.save_pretrained(path)
    tokenizer.save_pretrained(path)


    # evaluating test dataset
    console.log(f"[Initiating Validation]...\n")
    for epoch in range(config.EPOCHS):
        predictions, actuals = validate(epoch, tokenizer, model, device, val_loader)
        final_df = pd.DataFrame({'Generated Text':predictions,'Actual Text':actuals})
        final_df.to_csv(os.path.join(config.output_dir,'predictions.csv'))
    
    console.save_text(os.path.join(config.output_dir,'logs.txt'))
    
    console.log(f"[Validation Completed.]\n")
    console.print(f"""[Model] Model saved @ {os.path.join(config.output_dir, "model_files")}\n""")
    console.print(f"""[Validation] Generation on Validation data saved @ {os.path.join(config.output_dir,'predictions.csv')}\n""")
    console.print(f"""[Logs] Logs saved @ {os.path.join(config.output_dir,'logs.txt')}\n""")
        
        

if __name__ == "__main__":

    # load data
        # logging
    console.log(f"[Data]: Reading data...\n")

    # Importing the raw dataset
    dataframe = dataframe[[source_text,target_text]]
    
    # Creation of Dataset and Dataloader
    # Defining the train size. So 80% of the data will be used for training and the rest for validation. 
    train_size = 0.8
    train_dataset=dataframe.sample(frac=train_size).reset_index(drop=True)
    val_dataset=dataframe.drop(train_dataset.index).reset_index(drop=True)
    
    console.print(f"FULL Dataset: {dataframe.shape}")
    console.print(f"TRAIN Dataset: {train_dataset.shape}")
    console.print(f"TEST Dataset: {val_dataset.shape}\n")

    # train

    T5_trainer()

     
