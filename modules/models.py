# This file contains 4 classes:
# MyModule, MyDataset, Data_Augmenter, and RE_Classifier

import pandas as pd
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from transformers import GPT2LMHeadModel, GPT2Tokenizer, AutoTokenizer, AutoModelForSequenceClassification
import torch
from torch.optim import Adam, AdamW
import json
import os
from tqdm.auto import tqdm
from modules.util import load_json
from torch import nn

class MyModule():
    '''
    This class takes in a dataframe, and produces DataLoaders of train/test data. It divides the data either bby bootstrapping  or with a standard shuffle and split. To retrieve the DataLoaders use get_trainloader/get_testloader/get_valloader.
    '''
    def __init__(self, dataframe, bootstrap=False):
        self.dataframe = dataframe
        if bootstrap:
            self.bootstrap_and_split()
        else:
            self.shuffle_and_split()

    # DARE model does not end up using the bootstrap method - just uses the shuffle and split method.
    def bootstrap_and_split(self):
        counts = self.dataframe.group_label_id.value_counts()
        self.dataframe["weight"] = self.dataframe.group_label_id.map({c : 1/counts[c] for c in range(5)})
        train_boot = self.dataframe.sample(frac=1, weights=self.dataframe["weight"], replace=True, ignore_index=False)
        OOB = set(list(self.dataframe.index)) - set(train_boot.index)
        leftover = self.dataframe.loc[list(OOB)].reset_index(drop=True)
        test, val = train_test_split(leftover, shuffle=True, train_size=0.75, stratify=leftover.group_label_id)

        train_boot.reset_index(drop=True, inplace=True)
        test.reset_index(drop=True, inplace=True)
        val.reset_index(drop=True, inplace=True)
        self.train_data = MyDataset(train_boot)
        self.test_data = MyDataset(test)
        self.val_data = MyDataset(val)
        return


    def shuffle_and_split(self):
        train, test = train_test_split(
            self.dataframe, shuffle=True, 
            train_size=0.75, stratify=self.dataframe["group_label_id"])
        
        train.reset_index(drop=True, inplace=True)
        test.reset_index(drop=True, inplace=True)

        self.train_data = MyDataset(train)
        self.test_data = MyDataset(test)
        return
    
    def get_trainloader(self, label=None):
        if label != None:
            self.train_data.filterByClass(label)
        else:
            self.train_data.unFilterDataFrame()
        return DataLoader(self.train_data, batch_size=4)

    def get_testloader(self):
        return DataLoader(self.test_data, batch_size=4)

    def get_valloader(self):
        return DataLoader(self.val_data, batch_size=4)

class MyDataset(Dataset):
    '''
    Dataset class required in order to create a DataLoader object.
    '''
    def __init__(self, dataframe):
        self.dataframe = dataframe
        self.full_dataset = dataframe.copy()

    def filterByClass(self, label):
        self.dataframe = self.full_dataset[self.full_dataset.group_label_id==label].copy()
        return

    def unFilterDataFrame(self):
        self.dataframe = self.full_dataset.copy()
        return

    def bootstrap(self):
        # Evenly weight boot
        counts = self.dataframe.group_label_id.value_counts()
        self.dataframe["weight"] = self.dataframe.group_label_id.map({c : 1/counts[c] for c in range(5)})
        self.dataframe = self.dataframe.sample(frac=1, weights=self.dataframe["weight"], replace=True, ignore_index=True)
        return

    def strip_fakes(self):
        self.dataframe = self.dataframe[self.dataframe.fake_example==0]
        self.dataframe.reset_index(inplace=True, drop=True)
        return

    def get_weights(self):
        weights = []
        freq_min = min(self.dataframe.group_label_id.value_counts())
        for c in range(5):
            w = freq_min/self.dataframe.group_label_id.value_counts()[c]
            weights.append(float(w))
        return weights 

    def get_size(self, label):
        return self.dataframe[self.dataframe.group_label_id==label].shape[0]

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, index):
        item = self.dataframe.iloc[index]
        return {
            "text" : item["text_modified"],
            "labels" : item["group_label_id"]
        }



class Data_Augmenter():
    '''
    Data Augmenter class loads in the GPT2 language model either locally or from huggingface. It also creates a MyModule object which is used for fine-tuning. It fine-tunes given a label, as well as generates given a label. Finally it saves the generated examples using save_examples() and it can save the mixed synthetic + real training data as well as the all real test data using save_train_test(). 
    '''
    def __init__(self, config_files="config_files", gpu_device=0):
        self.config = load_json(f"{config_files}/dare_config.json")
        self.paths = load_json(f"{config_files}/paths.json")
        tokenizer_path = self.paths["dare_tokenizer_local"] if os.path.exists(self.paths["dare_tokenizer_local"]) else self.paths["dare_API"]
        model_path = self.paths["dare_model_local"] if os.path.exists(self.paths["dare_model_local"]) else self.paths["dare_API"]
        
        self.model = GPT2LMHeadModel.from_pretrained(model_path)
        self.tokenizer = GPT2Tokenizer.from_pretrained(tokenizer_path)
        
        if tokenizer_path == self.paths["dare_API"] and model_path == self.paths["dare_API"]:
            self.update_tokenizer()

        self.data_module = MyModule(dataframe=pd.read_csv(self.paths["dataset"]))

        self.device = torch.device(f"cuda:{gpu_device}") if torch.cuda.is_available() else torch.device("cpu")

        self.examples = {
            0 : [],
            1 : [],
            2 : [],
            3 : [],
            4 : []
        }

    def update_tokenizer(self):
        self.tokenizer.add_tokens(self.config["EntityA"])
        self.tokenizer.add_tokens(self.config["EntityB"])
        self.tokenizer.add_special_tokens({"pad_token": "[PAD]"})
        self.model.resize_token_embeddings(len(self.tokenizer))
        self.tokenizer.save_pretrained(self.paths["dare_tokenizer_local"])
        self.model.save_pretrained(self.paths["dare_model_local"])
        return

    def reset_model(self):
        self.model = GPT2LMHeadModel.from_pretrained(self.paths["dare_model_local"])
        return

    def train(self, label, location=None):
        self.reset_model()
        self.model.to(self.device)

        data = self.data_module.get_trainloader(label)
        self.optimizer = Adam(self.model.parameters(), lr=self.config["learning_rate"])
        n_epochs = self.config["epochs"]
        progress_bar = tqdm(range(n_epochs*len(data)), desc=f"Training DARE-{label}")
        self.model.train()
        for _ in range(n_epochs):
            for i, batch in enumerate(data):
                tokens = self.tokenizer(batch["text"],
                                        max_length=128,
                                        truncation=True,
                                        padding="max_length",
                                        return_attention_mask=True,
                                        return_tensors="pt")


                inputs = tokens.input_ids
                attention_mask = tokens.attention_mask
                outputs = self.model(input_ids=inputs.to(self.device), attention_mask=attention_mask.to(self.device), labels=inputs.to(self.device))
                # print(batch["text"])
                # print(inputs)
                # print(attention_mask)
                # print(tokens)
                print(outputs)
                break
                loss = outputs.loss
                loss.backward()
                
                if i % 2 != 0:
                    self.optimizer.step()
                    self.optimizer.zero_grad()
                
                progress_bar.update(1)
        if location == None:
            folder = self.paths["dare_model_trained"]
            location = f"{folder}model_{label}"
        self.model.save_pretrained(location)
        return
    
    def generate(self, label, num_sentences=None):
        model_path = self.paths["dare_model_trained"] + f"/model_{label}"
        
        self.train(label, location=model_path)
        self.model = GPT2LMHeadModel.from_pretrained(model_path)
        if num_sentences == None:
            num_sentences = int(self.data_module.train_data.get_size(label)*0.5) # Doubling the dataset is computationally very expensive, only increasing the data by 50%.
        progress_bar = tqdm(range(num_sentences), desc="Generating Sentences")
        while num_sentences > 0:
            outputs = self.model.generate(
                    do_sample=True,
                    max_length=100,
                    top_k=5,
                    temperature=1.0,
                    min_length=8,
                    pad_token_id=self.tokenizer.pad_token_id, 
                    num_return_sequences=num_sentences)

            print(outputs)
            for generated in outputs:
                text = self.tokenizer.decode(generated, skip_special_tokens=True)
                if self.config["EntityA"] in text and self.config["EntityB"] in text:
                    self.examples[label].append(text)
                    num_sentences -= 1
                    progress_bar.update(1)
        return
    
    def save_examples(self):
        df = pd.DataFrame(columns=["text_modified", "group_label_id"])
        for key in self.examples.keys():
            for text in self.examples[key]:
                df.loc[len(df)] = [text, key]
        df["fake_example"] = 1
        df.to_csv(self.paths["dare_dataset"], index=False)
        return

    def save_train_test(self):
        real_data = self.data_module.train_data.full_dataset.copy() 
        real_data["fake_example"] = 0
        real_data = real_data[["text_modified", "group_label_id", "fake_example"]]
        fake_data = pd.read_csv(self.paths["dare_dataset"])
        self.dare_train_data = pd.concat([real_data, fake_data])
        self.dare_train_data.to_csv(self.paths["dare_train_dataset"], index=False)
        self.data_module.test_data.dataframe.to_csv(self.paths["dare_test_dataset"], index=False)
        return 


class RE_Classifier():
    '''
    Relation extraction classifier. This class reads in the biomed roberta model either locally or from huggingface. If DARE is being used it can be instructed to load in the DARE train/test split data, otherwise it will read in the dataset specified in config_files/paths.json (Still ChemProt or ChemProt_Reduced but will make a new random split rather than use the split saved by the last Data_Augmenter). RE_Classifier fine-tunes using the train() method and tests using the test()
    '''
    def __init__(self, config_path="config_files", gpu_device=0, loaded_data=False, strip_fakes=False):
        self.config = load_json(f"{config_path}/re_config.json")
        self.paths = load_json(f"{config_path}/paths.json")
        
        if loaded_data:
            train = MyDataset(pd.read_csv(self.paths["dare_train_dataset"]))
            if strip_fakes:
                train.strip_fakes()
            self.class_weights = torch.tensor(train.get_weights())    
            
            test = MyDataset(pd.read_csv(self.paths["dare_test_dataset"]))

            self.train_loader = DataLoader(train, batch_size=4, shuffle=True)
            self.test_loader = DataLoader(test, batch_size=4, shuffle=True)
        else:
            self.DataModule = MyModule(dataframe=pd.read_csv(self.paths["dataset"]))
            self.build_datasets()
        

        tokenizer_path = self.paths["re_tokenizer_local"] if os.path.exists(self.paths["re_tokenizer_local"]) else self.paths["re_API"]
        model_path = self.paths["re_model_local"] if os.path.exists(self.paths["re_model_local"]) else self.paths["re_API"]
        
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_path, num_labels=self.config["n_classes"])
        self.add_masks_to_tokenizer()

        self.device = torch.device(f"cuda:{gpu_device}") if torch.cuda.is_available() else torch.device("cpu")
        self.model.to(self.device)

        self.results = {
            "labels" : [],
            "predictions" : [],
            "probs" : []
        }

    def add_masks_to_tokenizer(self):
        self.tokenizer.add_tokens(self.config["EntityA"])
        self.tokenizer.add_tokens(self.config["EntityB"])
        self.model.resize_token_embeddings(len(self.tokenizer))
        self.model.save_pretrained(self.paths["re_model_local"])
        self.tokenizer.save_pretrained(self.paths["re_tokenizer_local"])
        return

    def build_datasets(self):
        self.class_weights = torch.tensor(self.DataModule.train_data.get_weights())
        self.train_loader = self.DataModule.get_trainloader()
        self.test_loader = self.DataModule.get_testloader()
        return

    def save_model(self, path=None):
        if path == None:
            path = self.paths["re_model_trained"]
        self.model.save_pretrained(path)
        return

    def get_results(self):
        return self.results

    def save_results(self, filename):
        with open(filename, "w") as file:
            json.dump(self.results, file)
        return 

    def reset_results(self):
        self.results = {
            "labels" : [],
            "predictions" : [],
            "probs" : []
        }
        return

    def train(self):
        self.model.train()
        
        n_epochs = self.config["epochs"]
        total_steps = n_epochs * len(self.train_loader)
        
        self.optimizer = AdamW(self.model.parameters(), lr=self.config["learning_rate"])

        progress_bar = tqdm(range(total_steps), desc="Training RE Model")
        for epoch in range(n_epochs):
            for batch in self.train_loader:
                # Tokenize Batch
                tokens = self.tokenizer(batch["text"],
                                        return_tensors="pt",
                                        truncation=True,
                                        padding="max_length",
                                        max_length=self.config["RE_Tokenizer_MaxLength"],
                                        return_attention_mask=True)

                input_ids = tokens.input_ids.reshape(-1, self.config["RE_Tokenizer_MaxLength"])
                attention_mask = tokens.attention_mask.reshape(-1, self.config["RE_Tokenizer_MaxLength"])
                labels = batch["labels"].reshape(-1)

                print(tokens)
                print(input_ids)
                print(attention_mask)
                
                # Feed Model
                outputs = self.model(input_ids=input_ids.to(self.device), attention_mask=attention_mask.to(self.device), labels=labels.to(self.device))

                loss_fn = nn.CrossEntropyLoss(weight=self.class_weights).to(self.device)
                loss = loss_fn(outputs.logits, labels.to(self.device))
                loss.backward()

                # Update Model
                self.optimizer.step()
                self.optimizer.zero_grad()
                progress_bar.update(1)
        return

    def test(self):
        self.model.eval()
        self.reset_results()
        
        with torch.no_grad():
            for batch in self.test_loader:
                tokens = self.tokenizer(batch["text"],
                                        return_tensors="pt",
                                        truncation=True,
                                        padding="max_length",
                                        max_length=self.config["RE_Tokenizer_MaxLength"],
                                        return_attention_mask=True)
                input_ids = tokens.input_ids.reshape(-1, self.config["RE_Tokenizer_MaxLength"])
                attention_mask = tokens.attention_mask.reshape(-1, self.config["RE_Tokenizer_MaxLength"])
                labels = batch["labels"].reshape(-1, 1)

                # Feed Model
                outputs = self.model(input_ids=input_ids.to(self.device), attention_mask=attention_mask.to(self.device))
                logits = outputs.logits
                probs = logits.softmax(-1)
                predictions = torch.argmax(probs, dim=-1)

                # Accumulate Outputs
                self.results["labels"].extend(labels.cpu().flatten().tolist())
                self.results["predictions"].extend(predictions.cpu().tolist())
                self.results["probs"].extend(probs.cpu().tolist())
        return
