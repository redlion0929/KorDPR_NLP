import transformers
from datasets import Dataset
from transformers import BertModel, BertTokenizerFast, Trainer, TrainingArguments
import torch
from datasets import load_from_disk, load_dataset
from tqdm import tqdm
from rank_bm25 import BM25Okapi
import random
import json
import pandas as pd

tokenizer = BertTokenizerFast.from_pretrained('klue/bert-base')

class preprocess_dataset():
    def __init__(self, file_name, tokenizer):
        self.contexts = []
        self.questions = []
        self.dataset = self.get_json(file_name)
        self.tokenizer = tokenizer

        for data in self.dataset:
            for q_p in data['paragraphs']:
                passage = q_p['context']
                self.contexts.append(passage)
                self.questions.append([])
                for q in q_p['qas']:
                    if q['is_impossible'] == False:
                        self.questions[-1].append(q['question'])

        #bm25에 contexts 전달
        tokenized_corpus = [self.tokenizer(doc)['input_ids'] for doc in self.contexts]
        self.bm25 = BM25Okapi(tokenized_corpus)

    def get_json(self,file_name):
        with open(file_name) as f:
            return json.load(f)['data']

    #using bm25, return top_doc_index
    def get_negative_passage_bm25(self, question_row, question_column):
        tokenized_question = self.tokenizer(self.questions[question_row][question_column])['input_ids']
        top_doc = self.bm25.get_top_n(tokenized_question, self.contexts, n=100)
        top_doc_idx = [self.contexts.index(doc) for doc in top_doc]
        if question_row in top_doc_idx:
            return top_doc_idx[0]
        else:
            return -1

    def make_dataset(self, num_data):
        question_data = []
        pos_passage_data = []
        negative_passage_data = []
        negative_question_data = []
        for row, questions in tqdm(enumerate(self.questions)):
            for col, question in tqdm(enumerate(questions)):
                top_idx = self.get_negative_passage_bm25(row, col)
                if top_idx == -1:
                    continue
                else:
                    if len(self.questions[top_idx]) == 0:
                        continue
                    if len(self.questions[top_idx]) == 1:
                        negative_question_data.append(self.questions[top_idx][0])
                    else:
                        negative_question_data.append(
                            self.questions[top_idx][random.randint(0, len(self.questions[top_idx]) - 1)])
                question_data.append(question)
                pos_passage_data.append(self.contexts[row])
                negative_passage_data.append(self.contexts[top_idx])
            if len(question_data)>=num_data:
                break

        #suffle
        index_list = list(range(len(question_data)))
        random.shuffle(index_list)

        questions = [question_data[idx] for idx in index_list]
        pos_passages = [pos_passage_data[idx] for idx in index_list]
        negative_passages = [negative_passage_data[idx] for idx in index_list]

        return Dataset.from_pandas(pd.DataFrame({'pos_question': questions[:num_data],
                                                     'pos_passage': pos_passages[:num_data],
                                                     'neg_passage': negative_passages[:num_data]}))

#해당 데이터 셋은 AI-HUB의 도서자료 기계독해 데이터셋을 다운받아 넣으면 됩니다.
train_dataset_process = preprocess_dataset('book_mrc_train.json', tokenizer)
eval_dataset_process = preprocess_dataset('book_mrc_eval.json', tokenizer)

train_dataset = train_dataset_process.make_dataset(65000)
eval_dataset = eval_dataset_process.make_dataset(8500)

train_dataset.save_to_disk('book_mrc_train')
eval_dataset.save_to_disk('book_mrc_eval')

