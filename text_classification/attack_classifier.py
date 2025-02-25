import pandas as pd 
import huggingface_hub
import torch 
from transformers import (AutoTokenizer,
                          AutoModelForSequenceClassification,
                          DataCollatorWithPadding,
                          TrainingArguments,
                          pipeline)

from sklearn.preprocessing import LabelEncoder 
from sklearn.model_selection import train_test_split
from datasets import Dataset
import gc

from .Cleaner import Cleaner
from .training_utils import get_class_weights, compute_metrics
from .trainer import CustomTrainer

class AttackClassifier():
    def __init__(self,
                 model_path,
                 data_path=None,
                 text_column_name='text',
                 label_column_name='attack',
                 model_name = "distilbert/distilbert-base-uncased",
                 test_size=0.2,
                 num_labels=3,
                 huggingface_token = None
                 ):
        if torch.cuda.is_available():
            self.device = "cuda"
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            self.device = "mps"
        else:
            self.device = "cpu"
            
        self.model_path = model_path
        self.data_path = data_path
        self.text_column_name = text_column_name
        self.label_column_name = label_column_name
        self.model_name = model_name
        self.test_size = test_size
        self.num_labels = num_labels

        self.huggingface_token = huggingface_token
        if self.huggingface_token is not None:
            huggingface_hub.login(self.huggingface_token)
        
        self.tokenizer = self.load_tokenizer()

        if not huggingface_hub.repo_exists(self.model_path):

            if data_path is None:
                raise ValueError("Data path is required to train the model,since the model path does not exist in huggingface hub")

            train_data, test_data = self.load_data(self.data_path)
            train_data_df = train_data.to_pandas()
            test_data_df = test_data.to_pandas()

            all_data = pd.concat([train_data_df, test_data_df]).reset_index(drop=True)
            class_weights = get_class_weights(all_data)

            self.train_model(train_data, test_data, class_weights)

        self.model = self.load_model(self.model_path)

    def load_model(self,model_path):
        model = pipeline('text-classification', model=model_path, return_all_scores=True)
        return model

    def train_model(self, train_data,test_data,class_weights):
        model = AutoModelForSequenceClassification.from_pretrained(self.model_name, 
                                                                   num_labels=self.num_labels,
                                                                   id2label=self.label_dict,
                                                                   )
        model.to(self.device)
        data_collator = DataCollatorWithPadding(tokenizer=self.tokenizer)

        training_args = TrainingArguments(
            output_dir = self.model_path,
            learning_rate=2e-4,
            per_device_train_batch_size=4,
            per_device_eval_batch_size=4,
            num_train_epochs=2,
            weight_decay=0.01,
            evaluation_strategy="epoch",
            logging_strategy="epoch",
            push_to_hub=True,
        )

        trainer = CustomTrainer(
            model=model,
            args=training_args,
            train_dataset = train_data,
            eval_dataset = test_data,
            tokenizer = self.tokenizer,
            data_collator=data_collator,
            compute_metrics= compute_metrics
        )

        trainer.set_device(self.device)
        trainer.set_class_weights(class_weights)

        trainer.train()

        # Flush Memory
        del trainer,model
        gc.collect()

        if self.device == 'cuda':
            torch.cuda.empty_cache()

    def simplify_attack(self, attack):
        if "Genjutsu" in attack:
            return "Genjutsu"
        if "Ninjutsu" in attack:
            return "Ninjutsu"
        if "Taijutsu" in attack:
            return "Taijutsu"
    
    def preprocess_function(self,tokenizer,examples):
        return tokenizer(examples['text_cleaned'],truncation=True)

    def load_data(self,data_path):
        df = pd.read_json(data_path,lines=True)
        df['attack_type_simplified'] = df['attack_type'].apply(self.simplify_attack)
        df['text'] = df['attack_name'] + ". " + df['Attack_description']
        df[self.label_column_name] = df['attack_type_simplified']
        df = df[['text', self.label_column_name]]
        df = df.dropna()

        # Clean Text
        cleaner = Cleaner()
        df['text_cleaned'] = df[self.text_column_name].apply(cleaner.clean)

        # Encode Labels 
        le = LabelEncoder()
        le.fit(df[self.label_column_name].tolist())

        label_dict = {index:label_name for index, label_name in enumerate(le.__dict__['classes_'].tolist())}
        self.label_dict = label_dict
        df['label'] = le.transform(df[self.label_column_name].tolist())

        # Train / Test Split
        test_size = 0.2
        df_train, df_test = train_test_split(df, 
                                            test_size=test_size, 
                                            stratify=df['label'],)
        
        # Conver Pandas to a hugging face dataset
        train_dataset = Dataset.from_pandas(df_train)
        test_dataset = Dataset.from_pandas(df_test)

        # tokenize the dataset
        tokenized_train = train_dataset.map(lambda examples: self.preprocess_function(self.tokenizer, examples),
                                            batched=True)
        tokenized_test = test_dataset.map(lambda examples: self.preprocess_function(self.tokenizer, examples),
                                            batched=True)
        
        return tokenized_train, tokenized_test

    def load_tokenizer(self):
        if huggingface_hub.repo_exists(self.model_path):
            tokenizer = AutoTokenizer.from_pretrained(self.model_path)
        else:
            tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        return tokenizer

    def postprocess(self,model_output):
        output=[]
        for pred in model_output:
            label = max(pred, key=lambda x: x['score'])['label']
            output.append(label)
        return output

    def classify_attack(self,text):
        model_output = self.model(text)
        predictions =self.postprocess(model_output)
        return predictions