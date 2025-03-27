from transformers import T5EncoderModel, T5Tokenizer

from embedding_extraction.embedding_based import EmbeddingBased


class Prot5Based(EmbeddingBased):
    def __init__(
        self,
        name_device="cuda",
        name_model="Rostlab/prot_t5_xl_uniref50",
        name_tokenizer="Rostlab/prot_t5_xl_uniref50",
        dataset=None,
        column_seq=None,
        columns_ignore=[],
    ):
        super().__init__(
            name_device, name_model, name_tokenizer, dataset, column_seq, columns_ignore
        )

    def load_model_tokenizer(self):
        self.tokenizer = T5Tokenizer.from_pretrained(
            self.name_model, do_lower_case=False, use_fast=False
        )

        self.model = T5EncoderModel.from_pretrained(self.name_model).to(self.device)

        self.model.eval()
