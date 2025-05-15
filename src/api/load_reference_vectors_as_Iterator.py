import pandas as pd
import sys
sys.path.append(r"C:\Users\Abdessamad\Desktop\MLOpsClassificationTexteV2\src\data")
from classPreTraitement import PreTraitement
import pandas as pd
from typing import List, Tuple,Optional,Iterator

class CSVIterator:
    def __init__(self, file_path: str, chunksize: int = 100) -> None:
        self.file_path: str = file_path
        self.chunksize: int = chunksize
        self.reader: pd.io.parsers.TextFileReader = pd.read_csv(file_path, chunksize=chunksize)
        self.chunk: Optional[pd.DataFrame] = None
        self.chunk_iter: Iterator = iter([])

    def __iter__(self)->"CSVIterator":
        return self  

    def __next__(self)->Tuple:
        try:
            row = next(self.chunk_iter) 
            return row
        except StopIteration:
            self.chunk = next(self.reader, None)
            if self.chunk is None:
                raise StopIteration  
            self.chunk_iter = iter(self.chunk.itertuples(index=False, name=None))
            return next(self.chunk_iter) 

file_path = "../../data/processed/resultat_train.csv"
csv_iter = CSVIterator(file_path)

def reference_chunks() -> Tuple[List[str], List[str]]:
    reference_chunks_text = []
    reference_chunks_label = []
    for row in csv_iter:
        traitement = PreTraitement(pd.DataFrame([{'text': row[0]}]))
        traitement.SupprimerValMonquent()
        lot_cleaned = traitement.dataset["text"].apply(traitement.remove_specials)
        traitement.dataset["text"] = lot_cleaned
        
        reference_chunks_text.append(traitement.dataset["text"].iloc[0])
        reference_chunks_label.append(row[1])
        break
    return reference_chunks_text, reference_chunks_label


reference_chunks()
