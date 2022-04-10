from torch.utils.data import Dataset
import torch
import random
from datasets import load_from_disk
from datasets import Dataset as data_set

class merged_kdpr_dataset:
    def __init__(self, dataset_array):
        self.dataset_array = dataset_array
        self.pos_questions = []
        self.pos_passages = []
        self.neg_questions = []
        self.neg_passages = []
        for dataset in self.dataset_array:
            for data in dataset:
                self.pos_questions.append(data['pos_question'])
                self.pos_passages.append(data['pos_passage'])
                self.neg_questions.append(data['neg_question'])
                self.neg_passages.append(data['neg_passage'])

    def make_dataset(self):
        # suffle
        index_list = list(range(len(self.pos_questions)))
        random.shuffle(index_list)

        questions = [self.pos_questions[idx] for idx in index_list]
        pos_passages = [self.pos_passages[idx] for idx in index_list]
        negative_questions = [self.neg_questions[idx] for idx in index_list]
        negative_passages = [self.neg_passages[idx] for idx in index_list]

        return data_set.from_dict({'pos_question': questions,
                                  'pos_passage': pos_passages,
                                  'neg_question': negative_questions,
                                  'neg_passage': negative_passages})

train_data = merged_kdpr_dataset([
                                    load_from_disk('squad_hub_nia_clue'),
                                    load_from_disk('book_mrc_train'),
                                ])

eval_data = merged_kdpr_dataset([
                                    load_from_disk('squad_hub_nia_normal'),
                                    load_from_disk('book_mrc_eval'),
                                ])

train_dataset = train_data.make_dataset()
eval_dataset = eval_data.make_dataset()

train_dataset.save_to_disk('kdpr_train')
eval_dataset.save_to_disk('kdpr_eval')
