import warnings
from typing import List, Dict, Optional, Tuple

import numpy as np
import torch
from rich import print
from tqdm import tqdm

from esm.models.esmc import ESMC
from esm.sdk.api import ESMProtein, LogitsConfig

from embedding_extraction.EmbeddingBased import EmbeddingBased

warnings.simplefilter("ignore", category=FutureWarning)

class ESMBasedEmbedding(EmbeddingBased):
    def __init__(
            self,
            name_device="cuda", 
            name_model="esmc_600m", 
            name_tokenizer=None,
            dataset=None,
            column_seq=None,
            columns_ignore=[]):
        
        super().__init__(name_device, name_model, name_tokenizer, dataset, column_seq, columns_ignore)

    def pad_sequence(self, sequence: str, target_length: int, pad_char: str = "X") -> str:
        """Pads or truncates a protein sequence to a fixed length."""
        return sequence[:target_length].ljust(target_length, pad_char)


    def embed_batch(self, name_model, batch: List[str], device: str) -> List[Dict] :
        """Embeds a batch of protein sequences on the GPU efficiently."""
        
        results: List[Dict] = []
        
        with torch.no_grad():
            for sequence in batch:
                try:
                    protein = ESMProtein(sequence=sequence)
                    protein_tensor = name_model.encode(protein).to(device)

                    EMBEDDING_CONFIG = LogitsConfig(return_embeddings=True, return_hidden_states=True)
                    logits_output = name_model.logits(protein_tensor, EMBEDDING_CONFIG)
                    
                    last_layer_embedings_mean = torch.mean(logits_output.embeddings, dim=1).squeeze()
                    
                    results.append({
                        "sequence": sequence,
                        "embedding": last_layer_embedings_mean.cpu().numpy(),
                        "hidden_states": logits_output.hidden_states
                    })
                    
                except Exception as e:
                    results.append({"sequence": sequence, "error": str(e)})

        return results
    

    def embeddingProcess(self,
                         seq_len: Optional[int] = None,
                         batch_size: int = 100) -> Tuple[List, List]:

        """Efficiently embeds large-scale protein sequences using a high-memory GPU."""
        sequences = self.dataset[self.column_seq].tolist()

        device = "cuda" if torch.cuda.is_available() else "cpu"
        model = ESMC.from_pretrained(self.name_model).to(device)
        
        
        if seq_len is None:
            target_length = max(len(seq) for seq in sequences)
        else:
            target_length = seq_len
            
        # print(f"[*] Target Length: {target_length}")
        padded_sequences = [self.pad_sequence(seq, target_length) for seq in sequences]
        
        # print(f"[*] Padded Sequences: {len(padded_sequences)}")

        # Detect actual embedding dimension dynamically
        sample_output = self.embed_batch(model, [padded_sequences[0]], device)[0]
        # print(f"[*] Output keys: {sample_output.keys()}")
        # print(f"[*] embedding: {sample_output}")
        # print(f"[*] hidden_states: {sample_output['hidden_states'].shape}")
        # print(f"[*] embedding: {sample_output['embedding'].shape}")
        

        # Process sequences in mini-batches with real-time progress tracking
        layers_embeddings = []
        states_means = []
        last_embeddings = []
        
        for i in tqdm(range(0, len(padded_sequences), batch_size), desc="[+] Embedding: ", unit="batch"):
            batch = padded_sequences[i : i + batch_size]
            batch_results = self.embed_batch(model, batch, device)
            last_embeddings.extend([result["embedding"] for result in batch_results])

            states_means.extend([torch.mean(result["hidden_states"], dim=-2).squeeze() for result in batch_results])
            torch.cuda.empty_cache()
                    

        n_layers, _ = states_means[0].shape
        
        # Convert each layer-mean to numpy array and store in a list
        for layer_idx in range(n_layers):
            layers_embeddings.append(torch.stack([layer[layer_idx, : ] for layer in states_means]).float().cpu().numpy())
            
        # Append the last layer embeddings
        layers_embeddings.extend([last_embeddings])
        
        
        # print(f"[*] States Length: {len(states_means)}")
        # print(f"[*] states-mean-shape: {states_means[0].shape}")
        # print(f"[*] Layers shape: {np.array(layers_embeddings).shape}")
        # print(f"[*] len(layers[-1]): [{len(last_embeddings)}] - Shape: {np.array(last_embeddings).shape}")

        return layers_embeddings, last_embeddings


    def export_layers_processed(self, layers_embeddings: List[np.ndarray], path, prefix: str) -> None:
        """Saves the embeddings to an AnnData file."""
        # print(f"[*] Exporting layers to: {path}")
        import os
        os.makedirs(path, exist_ok=True)
        
        for layer_idx, layer in enumerate(layers_embeddings):
            name_export = f"{path}{prefix}_layer_{layer_idx}.npy"
            with open(name_export, 'wb') as f:
                np.save(f, layer)