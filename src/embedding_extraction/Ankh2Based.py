from embedding_extraction.EmbeddingBased import EmbeddingBased
from transformers import AutoTokenizer, AutoModel
import torch
from tqdm import tqdm
import numpy as np
import sys

class Ankh2BasedEmbedding(EmbeddingBased):

    def __init__(
            self, 
            name_device="cuda", 
            name_model="ElnaggarLab/ankh2-ext1", 
            name_tokenizer="ElnaggarLab/ankh2-ext1",
            dataset=None,
            column_seq=None,
            columns_ignore=[]):
        
        super().__init__(name_device, name_model, name_tokenizer, dataset, column_seq, columns_ignore)

    def loadModelTokenizer(self):
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.name_model, 
            trust_remote_code=True)
        
        self.model = AutoModel.from_pretrained(
            self.name_model,
            trust_remote_code=True).to(self.device)
        
        self.model.eval()
    
    def embeddingBatch(self, batch, max_length=1024):
        
        inputs = self.tokenizer(
            batch, 
            return_tensors="pt", 
            truncation=True, 
            padding=True, 
            max_length=max_length,
            add_special_tokens=False
        ).to(self.device)

        # Forward pass using encoder-only mode
        with torch.no_grad():
            encoder_outputs = self.model.encoder(
                input_ids=inputs["input_ids"],
                output_hidden_states=False
            )
        
        del inputs

        return encoder_outputs.last_hidden_state
