from embedding_extraction.EmbeddingBased import EmbeddingBased
from transformers import BertModel, BertTokenizer

class BertBasedMebedding(EmbeddingBased):

    def __init__(
            self, 
            name_device="cuda", 
            name_model="Rostlab/prot_bert", 
            name_tokenizer="Rostlab/prot_bert",
            dataset=None,
            column_seq=None,
            columns_ignore=[]):
        
        super().__init__(name_device, name_model, name_tokenizer, dataset, column_seq, columns_ignore)
    
    def loadModelTokenizer(self):
        self.tokenizer = BertTokenizer.from_pretrained(
            self.name_model, 
            do_lower_case=False)
        
        self.model = BertModel.from_pretrained(self.name_model).to(self.device)

        self.model.eval()
