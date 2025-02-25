import spacy
from nltk.tokenize import sent_tokenize
import pandas as pd
from ast import literal_eval
import os 
import sys
import pathlib
folder_path = pathlib.Path(__file__).parent.parent.resolve()
sys.path.append(str(folder_path))
from Utils import load_subs_datasets


class NamedEntityRecognizer:
    def __init__(self):
        self.nlp_model = self.load_model()
        pass

    def load_model(self):
        return spacy.load("en_core_web_trf")

    def get_ners_inference(self,script):
        script_sentences = sent_tokenize(script)

        ner_output = []

        for sentence in script_sentences:
            doc = self.nlp_model(sentence)
            ners = set()
            for entity in doc.ents:
                if entity.label_ =="PERSON":
                    full_name = entity.text
                    first_name = full_name.split(" ")[0]
                    first_name = first_name.strip()
                    ners.add(first_name)
            ner_output.append(ners)

        return ner_output

    def get_ners(self,dataset_path,save_path=None):
        if save_path is not None and os.path.exists(save_path):
            df = pd.read_csv(save_path)
            df['ners'] = df['ners'].apply(lambda x: literal_eval(x) if isinstance(x,str) else x)
            return df

        df = load_subs_datasets(dataset_path)
        df['ners'] = df['script'].apply(self.get_ners_inference)

        if save_path is not None:
            df.to_csv(save_path,index=False)
        
        return df