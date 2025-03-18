import sys
sys.path.insert(0, "../src/")

from embedding_extraction.ESMBased import ESMBasedEmbedding
import pandas as pd

df_data = pd.read_csv("/home/dmedina/Desktop/tutorials/training_models_PI/raw_data/Antimicrobial/train_data.csv")

name_model = "facebook/esm2_t36_3B_UR50D"

esm_based = ESMBasedEmbedding(
    name_device="cuda", 
    dataset=df_data, 
    name_model=name_model,
    name_tokenizer=name_model,
    column_seq="sequence",
    columns_ignore=["label"]
)

print("Loading model/tokenizer")
esm_based.loadModelTokenizer()

print("Generating embedding")
df_embedding = esm_based.embeddingProcess(batch_size=50)

print(df_embedding)

esm_based.cleaning_memory()
print("Process finished")
