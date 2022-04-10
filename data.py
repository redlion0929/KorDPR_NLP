from torch.utils.data import Dataset
import torch
import random
from datasets import load_from_disk
from datasets import Dataset as data_set

class kdpr_dataset(Dataset):
    def __init__(self, data_dir, tokenizer, num_use=None):
        self.tokenizer = tokenizer
        self.pad_token = self.tokenizer.convert_tokens_to_ids(['[PAD]'])[0]
        self.num_use = num_use
        if num_use != None:
            self.dataset = data_set.from_dict(load_from_disk(data_dir)[:num_use])
        else:
            self.dataset = load_from_disk(data_dir)


    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        feature = dict()
        data = self.dataset[idx]
        #total_question = [data['pos_question'], data['neg_question']]
        total_question = [data['pos_question']]

        total_passage = [data['pos_passage'], data['neg_passage']]

        tokenized_question = self.tokenizer(total_question, truncation=True, padding=True)
        tokenized_passage = self.tokenizer(total_passage, truncation=True, padding=True)

        feature['q_input_ids'] = torch.LongTensor(tokenized_question['input_ids'])
        feature['q_attention_mask'] = torch.FloatTensor(tokenized_question['attention_mask'])
        feature['q_token_type_ids'] = torch.LongTensor(tokenized_question['token_type_ids'])
        feature['p_input_ids'] = torch.LongTensor(tokenized_passage['input_ids'])
        feature['p_attention_mask'] = torch.FloatTensor(tokenized_passage['attention_mask'])
        feature['p_token_type_ids'] = torch.LongTensor(tokenized_passage['token_type_ids'])

        return feature

    def collate_fn(self, batch):
        merged_batch = {key: [d[key] for d in batch] for key in batch[0]}
        max_q_len = max(len(x) for y in merged_batch['q_input_ids'] for x in y)
        max_p_len = max(len(x) for y in merged_batch['p_input_ids'] for x in y)

        merged_batch['q_input_ids'] = self.func_in_collate_q('q_input_ids', merged_batch, max_q_len).long()
        merged_batch['q_attention_mask'] = self.func_in_collate_q('q_attention_mask', merged_batch, max_q_len).float()
        merged_batch['q_token_type_ids'] = self.func_in_collate_q('q_token_type_ids', merged_batch, max_q_len).long()

        merged_batch['p_input_ids'] = self.func_in_collate('p_input_ids', merged_batch, max_p_len).long()
        merged_batch['p_attention_mask'] = self.func_in_collate('p_attention_mask', merged_batch, max_p_len).float()
        merged_batch['p_token_type_ids'] = self.func_in_collate('p_token_type_ids', merged_batch, max_p_len).long()

        question = {'input_ids': merged_batch['q_input_ids'],
                  'token_type_ids': merged_batch['q_token_type_ids'],
                  'attention_mask': merged_batch['q_attention_mask']}

        passage = {'input_ids': merged_batch['p_input_ids'],
                    'token_type_ids': merged_batch['p_token_type_ids'],
                    'attention_mask': merged_batch['p_attention_mask']}
        return [question, passage]

    def func_in_collate(self, feature, merged_batch, max_len):
        collated_data = []
        for data in merged_batch[feature]:
            pad_token = self.pad_token
            collated_data.append(data[0].tolist() + [pad_token]*(max_len - len(data[0])))
            collated_data.append(data[1].tolist() + [pad_token]*(max_len - len(data[1])))

        return torch.tensor(collated_data)

    def func_in_collate_q(self, feature, merged_batch, max_len):
        collated_data = []
        for data in merged_batch[feature]:
            pad_token = self.pad_token
            collated_data.append(data[0].tolist() + [pad_token]*(max_len - len(data[0])))

        return torch.tensor(collated_data)
