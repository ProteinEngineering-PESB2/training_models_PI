import torch
from tqdm import tqdm
import numpy as np
import pandas as pd

class EmbeddingBased(object):

    def __init__(
            self,
            name_device="cuda",
            name_model=None,
            name_tokenizer=None,
            dataset=None,
            column_seq=None,
            columns_ignore=[]):
        
        self.name_device = name_device
        self.name_model = name_model
        self.name_tokenizer = name_tokenizer
        self.dataset = dataset
        self.column_seq = column_seq
        self.columns_ignore = columns_ignore

        self.tokenizer = None
        self.model = None

        self.embeddings = []

        self.status = True
        self.message = ""

        self.__select_device()

    def __select_device(self):
        self.device = torch.device(self.name_device)
        print("Using device: ", self.device)
    
    def loadModelTokenizer(self):
        pass

    def cleaning_memory(self):
        
        print("Empty cache memory")
        torch.cuda.empty_cache()
    
    def embeddingBatch(self, batch, max_length=1024):
        
        inputs = self.tokenizer(
            batch, 
            return_tensors="pt", 
            truncation=True, 
            padding=True,
            add_special_tokens=False, 
            max_length=max_length).to(self.device)
        
        inputs = {key: value.to(self.device) for key, value in inputs.items()}

        with torch.no_grad():
            all_hiden_layers = self.model(**inputs, output_hidden_states=False)
            
        del inputs

        return all_hiden_layers.last_hidden_state
        
    def embeddingProcess(self, batch_size=100):

        sequences = self.dataset[self.column_seq].tolist()

        layer_embeddings = []

        for i in tqdm(range(0, len(sequences), batch_size), desc="[+] Embedding", unit="batch"):
            batch = sequences[i:i + batch_size]

            last_hidden_layer = self.embeddingBatch(batch=batch)

            batch_embedding = last_hidden_layer.mean(dim=1).cpu().numpy()
            layer_embeddings.append(batch_embedding)

        layer_embeddings = np.concatenate(layer_embeddings, axis=0)
        
        header = [f"p_{i+1}" for i in range(layer_embeddings.shape[1])]

        df_embedding = pd.DataFrame(data=layer_embeddings, columns=header)
        
        for column in self.columns_ignore:
            df_embedding[column] = self.dataset[column].values
        
        return df_embedding