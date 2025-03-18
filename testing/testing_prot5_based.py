import sys
sys.path.insert(0, "../src/")

from embedding_extraction.Prot5Based import Prot5Based
import pandas as pd

df_data = pd.read_csv("/home/dmedina/Desktop/tutorials/training_models_PI/raw_data/Antimicrobial/train_data.csv")
df_data = df_data[:100]

df_data["sequence"] = df_data["sequence"].apply(lambda x: ' '.join(x))

name_model = "Rostlab/ProstT5"

prot5_based = Prot5Based(
    name_device="cuda", 
    dataset=df_data, 
    name_model=name_model,
    name_tokenizer=name_model,
    column_seq="sequence",
    columns_ignore=["label"]
)

print("Loading model/tokenizer")
prot5_based.loadModelTokenizer()

print("Generating embedding")
df_embedding = prot5_based.embeddingProcess(batch_size=50)

print(df_embedding)

prot5_based.cleaning_memory()
print("Process finished")
