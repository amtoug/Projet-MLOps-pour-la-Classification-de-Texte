import pandas as pd
import re
from transformers import AutoTokenizer

class PreTraitement:
    _instance = None

    def __new__(cls, dataset: pd.DataFrame):
        if not cls._instance:
            cls._instance = super(PreTraitement, cls).__new__(cls)
            cls._instance.dataset = dataset
            cls._instance.tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
            # cls._instance.nlp = spacy.load("en_core_web_sm", disable=["ner", "parser"]) 
        return cls._instance

    def SupprimerValMonquent(self) -> None:
        self.dataset.dropna(inplace=True)

    def remove_specials(self, text: str) -> str:
        text = text.strip()
        text = re.sub(r'\s+', ' ', text)
        return re.sub(r'[^a-zA-Z0-9À-ÿ\s]', '', text)

    def spacy_tokenizer(self, text: str)-> list[str]:
        """Tokenise le texte. Utilise spaCy ou le tokenizer BERT."""
        #                       Pour spaCy :
        # Assuming the spacy tokenizer is needed, but it should be uncommented if used
        # doc = self.nlp(text)
        # return [token.lemma_ for token in doc if not token.is_stop and not token.is_punct]
        #                   Pour BERT tokenizer :
        return self.tokenizer.tokenize(text) 





# -------------------------------------------------------------------------------------------
# import pandas as pd
# import re
# # import spacy
# from transformers import AutoTokenizer
# class PreTraitement():
#     def __init__(self, dataset: pd.DataFrame):
#         self.dataset = dataset
#         # self.nlp = spacy.load("en_core_web_sm", disable=["ner", "parser"])
#         self.tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')

#     def SupprimerValMonquent(self) -> None:
#         self.dataset.dropna(inplace=True)

#     def remove_specials(self, text: str) -> str:
#         text = text.strip()
#         text = re.sub(r'\s+', ' ', text)
#         return re.sub(r'[^a-zA-Z0-9À-ÿ\s]', '', text)

#     def spacy_tokenizer(self, text: str):
#         doc = self.nlp(text)
#         return [token.lemma_ for token in doc if not token.is_stop and not token.is_punct]