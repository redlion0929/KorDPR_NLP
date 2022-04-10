import transformers
from transformers import BertModel, BertTokenizerFast, Trainer, TrainingArguments
import torch
from torch import nn
import numpy as np
from tqdm import tqdm
from datasets import load_from_disk, load_dataset
from torch.utils.data import DataLoader
from data import kdpr_dataset
from model import kdpr
import argparse
from attrdict import AttrDict
import os
import json
from inference import inference
from transformers import get_linear_schedule_with_warmup

class train_dpr(object):
    def __init__(self, args):
        self.args = args
        self.m = nn.LogSoftmax(dim=1)

        np.random.seed(args.random_seed)
        torch.manual_seed(args.random_seed)
        torch.cuda.manual_seed_all(args.random_seed)

        # self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.device = torch.device("cuda:" + str(self.args.gpu_ids[0]) if torch.cuda.is_available() else "cpu")
        self.best_model = None
        self.best_val_loss = float('inf')

        self.build_model()
        self.build_dataloader()
        self.setup_training()


    def build_dataloader(self):
        self.train_dataset = kdpr_dataset(self.args.train_data_dir, self.tokenizer)
        self.eval_dataset = kdpr_dataset(self.args.validation_data_dir, self.tokenizer)
        self.raw_eval_dataset = load_from_disk(self.args.validation_data_dir)

        self.train_dataloader = DataLoader(
            self.train_dataset,
            batch_size=self.args.batch_size,
            num_workers=self.args.cpu_workers,
            shuffle=True,
            drop_last=True,
            collate_fn=self.train_dataset.collate_fn
        )

        self.eval_dataloader = DataLoader(
            self.eval_dataset,
            batch_size=self.args.batch_size,
            num_workers=self.args.cpu_workers,
            shuffle=True,
            drop_last=True,
            collate_fn=self.eval_dataset.collate_fn
        )

        print(
            """
            # -------------------------------------------------------------------------
            #   BUILD DATALOADER DONE
            # -------------------------------------------------------------------------
            """
        )

    def build_model(self):
        self.tokenizer = BertTokenizerFast.from_pretrained('klue/bert-base')
        self.model = kdpr().to(self.device)

        # Use Multi-GPUs
        if -1 not in self.args.gpu_ids and len(self.args.gpu_ids) > 1:
            print("Let's use", torch.cuda.device_count(), "GPUs!")
            self.model = nn.DataParallel(self.model)


        print(
            """
            # -------------------------------------------------------------------------
            #   BUILD MODEL DONE
            # -------------------------------------------------------------------------
            """
        )

    def setup_training(self):

        # =============================================================================
        #   optimizer / scheduler
        # =============================================================================

        self.iterations = len(self.train_dataset) // self.args.batch_size

        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.args.learning_rate)
        self.scheduler = get_linear_schedule_with_warmup(
            self.optimizer, num_warmup_steps=100,
            num_training_steps=self.iterations * self.args.num_epochs
        )

        self.loss_fn = nn.NLLLoss()

        print(
            """
            # -------------------------------------------------------------------------
            #   SETUP TRAINING DONE
            # -------------------------------------------------------------------------
            """
        )

    def train_one_epoch(self):
        self.model.train()

        for i, batch in tqdm(enumerate(self.train_dataloader)):
            question, passage=batch

            q_input_ids = question['input_ids']
            q_input_ids = q_input_ids.to(self.device)

            q_attention_mask = question['attention_mask']
            q_attention_mask = q_attention_mask.to(self.device)

            q_token_type_ids = question['token_type_ids']
            q_token_type_ids = q_token_type_ids.to(self.device)

            p_input_ids = passage['input_ids']
            p_input_ids = p_input_ids.to(self.device)

            p_attention_mask = passage['attention_mask']
            p_attention_mask = p_attention_mask.to(self.device)

            p_token_type_ids = passage['token_type_ids']
            p_token_type_ids = p_token_type_ids.to(self.device)

            question_output, passage_output = self.model(q_input_ids, q_attention_mask, q_token_type_ids, p_input_ids, p_attention_mask, p_token_type_ids)

            output = self.m(torch.matmul(question_output, passage_output.T))

            #target = torch.tensor([i for i in range(question['input_ids'].size(0))])
            target = torch.tensor([2*i for i in range(self.args.batch_size)])
            target = target.to(self.device)

            loss = self.loss_fn(output, target)

            loss.backward()


            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 0.5)
            self.optimizer.step()
            self.scheduler.step()
            self.optimizer.zero_grad()

    def evaluate(self):
        self.model.eval()
        total_loss = 0.
        correct = 0
        correct_1 = 0
        cnt = 0
        with torch.no_grad():
            for batch in self.eval_dataloader:
                question, passage = batch

                q_input_ids = question['input_ids']
                q_input_ids = q_input_ids.to(self.device)

                q_attention_mask = question['attention_mask']
                q_attention_mask = q_attention_mask.to(self.device)

                q_token_type_ids = question['token_type_ids']
                q_token_type_ids = q_token_type_ids.to(self.device)

                p_input_ids = passage['input_ids']
                p_input_ids = p_input_ids.to(self.device)

                p_attention_mask = passage['attention_mask']
                p_attention_mask = p_attention_mask.to(self.device)

                p_token_type_ids = passage['token_type_ids']
                p_token_type_ids = p_token_type_ids.to(self.device)

                question_output, passage_output = self.model(q_input_ids, q_attention_mask, q_token_type_ids,
                                                             p_input_ids, p_attention_mask, p_token_type_ids)

                output = self.m(torch.matmul(question_output, passage_output.T))

                # target = torch.tensor([i for i in range(question['input_ids'].size(0))])
                target = torch.tensor([2 * i for i in range(self.args.batch_size)])
                target = target.to(self.device)

                loss = self.loss_fn(output, target)
                total_loss += loss

                #정확도 계산
                for i in range(question['input_ids'].size(0)):
                    values, indices = torch.topk(output[i], k=3)
                    cnt+=1
                    if (2*i) in indices:
                        correct+=1

                    values, indices = torch.topk(output[i], k=1)
                    if (2 * i) in indices:
                        correct_1 += 1

        return total_loss / len(self.eval_dataloader), correct/cnt, correct_1/cnt


    def train(self):
        for e in tqdm(range(self.args.num_epochs)):
            print('Epoch:', e)
            self.train_one_epoch()
            val_loss, val_acc, val1_acc = self.evaluate()
            print('val_loss', val_loss)
            print('val_acc', val_acc)
            print('val_acc_about_top1', val1_acc)
            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                self.best_model = self.model
        torch.save(self.best_model.state_dict(), self.args.save_dirpath + self.args.load_pthpath + 'kmrc_mrc' + '.pth')


if __name__ == '__main__':
    arg_parser = argparse.ArgumentParser(description="DPR-korean (PyTorch)")
    arg_parser.add_argument("--mode", dest="mode", type=str, default="train")
    arg_parser.add_argument("--config_dir", dest="config_dir", type=str, default="config", help="Config Directory")
    arg_parser.add_argument("--config_file", dest="config_file", type=str, default="kdpr",
                            help="Config json file")
    parsed_args = arg_parser.parse_args()

    # Read from config file and make args
    with open(os.path.join(parsed_args.config_dir, "{}.json".format(parsed_args.config_file))) as f:
        args = AttrDict(json.load(f))

    # print("Training/evaluation parameters {}".format(args))
    dpr_train = train_dpr(args)
    if parsed_args.mode == 'train':
        dpr_train.train()
    else:
        inf = inference(args, dpr_train.model)
        top_5, top_20, top_100 = inf.test_retriever()
        print("top_5", top_5)
        print("top_20", top_20)
        print("top_100", top_100)
