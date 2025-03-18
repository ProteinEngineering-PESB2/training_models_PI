import sys
sys.path.insert(0, "../src/")

from embedding_extraction.BertBased import BertBasedMebedding
import pandas as pd

df_data = pd.read_csv("/home/dmedina/Desktop/tutorials/training_models_PI/raw_data/Antimicrobial/train_data.csv")
df_data = df_data[:100]

df_data["sequence"] = df_data["sequence"].apply(lambda x: ' '.join(x))

name_model = "Rostlab/prot_bert_bfd_ss3"

bert_based = BertBasedMebedding(
    name_device="cuda", 
    dataset=df_data, 
    name_model=name_model,
    name_tokenizer=name_model,
    column_seq="sequence",
    columns_ignore=["label"]
)

print("Loading model/tokenizer")
bert_based.loadModelTokenizer()

print("Generating embedding")
df_embedding = bert_based.embeddingProcess(batch_size=50)

print(df_embedding)

bert_based.cleaning_memory()
print("Process finished")
