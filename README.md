## KorDPR_NLP 

[Dense Passage Retrieval for Open-Domain Question Answering](https://arxiv.org/abs/2004.04906) 에서 소개된 모델인 DPR을 한국어 데이터셋을 사용하여 학습시킨 모델이다.

## Quick Links

  - [DPR이란?](#what_is_dpr)
  - [데이터 셋 (Dataset)](#dataset)
  - [학습 결과](#result)
  - [Environment Setting](#environment_setting)
  - [Hyperparameter](#hyperparameter)
  - [Train](#train)
  - [Evaluation](#evaluation)

## DPR이란?

DPR이란 BERT 기반 bi-encoder 구조를 사용하여 open-domain에서 question을 풀기 위한 passage를 찾아 주는 모델이다. 
DPR의 등장은 간단한 BERT 구조만을 가지고도 좋은 성능의 retriever를 만들 수 있음을 증명하였다. 
학습 시에는 in-batch negative training 전략을 사용하여 효율적인 학습을 진행하였으며, inference시에는 [faiss](https://github.com/facebookresearch/faiss) 를 사용하여 빠른 처리가 가능하도록 하였다.

본 repository에서는 이러한 원리의 DPR을 직접 구현해보고, 한국어 데이터셋을 가지고 fine-tuning 하였다. 학습에는 [AI Hub](https://aihub.or.kr/aihub-data/natural-language/about) 에서 제공하는 기계독해, 도서자료 기계독해 데이터셋을 사용하였다.

`klue/bert-base`모델을 사용하여 학습을 진행하였다.

## 데이터 셋 (Dataset)

기계독해, 도서자료 기계독해 데이터 셋을 사용하였으며, 각 데이터 셋을 처리할 때 아래와 같은 방법을 사용하였다.

1. BM25를 사용하여 전체 passage 중에 question과 가장 관련있는 negative passage(hard negative)를 찾는다.
   1. 이때 BM25를 사용하여 가장 관련있는 100개의 passage를 찾은 후 100개 안에 positive passage가 없으면 해당 question은 버린다.
2. 하나의 데이터는 (question, positive passage, hard negative) 형태를 가진다.
3. 전체 데이터 셋 중 train dataset, validation dataset을 각각 65000, 8500개씩 랜덤으로 저장한다.
이때 train dataset, validataion dataset은 서로 다른 source에서부터 나오므로 겹치지 않는다.

처리가 끝난 두 데이터 셋(기계독해, 도서자료 기계독해)을 합쳐 최종 학습, 평가에 쓰일 데이터 셋을 만든다.

관련 내용은 `preprocess_mrc.py`, `preprocess_book_mrc.py`, `dataset_merge.py`에 있다.

## 학습 결과
평가에는 validation dataset(약 17000개의 데이터)를 사용하였다.
이때 각 데이터는 두 개의 passage (possitive, negative)를 가지고 있으므로, 총 passage는 약 34,000여 개이다.

Faiss를 사용하여 34,000여 개의 passage과 데이터 셋 내의 각 question에 대해 inference를 진행하였으며 성능은 아래 표에 정리되어있다.

| Top k   | Accuracy |
|:--------|:--------:|
| Top 100 |   93.8   |
| Top 20  |   80.9   |
| Top 5   |   56.8   |

## Evaluation Setting
아래와 같은 라이브러리를 사용하여 실험을 진행하였다.

| Module       | Version |
|:-------------|:-------:|
| faiss-cpu    |  1.7.2  |
| transformers | 4.17.0  |
| torch        |  1.7.1  |
| tqdm         | 4.63.0  |
| datasets     | 1.18.3  |
| attrdict     |  2.0.1  |
| rank_bm25    |  0.2.2  |

## Hyperparameter

| Parameter     |          Value          |
|:--------------|:-----------------------:|
| Batch size    |           32            |
| Cpu workers   |            2            |
| Epochs        |           32            |
| Learning rate |          1e-5           |
| Optimizer     |          Adam           |
| Scheduler     | Linear warmup scheduler |

scheduler의 경우 초기 100step에 한해 warmup을 진행하였다.

학습 중 validation loss가 가장 적은 모델을 최종 모델로 선택하였다.

## Train

Train을 하기 위해서는, 다음과 같은 명령어를 사용하면 된다.
 `python train.py --mode train`

## Evaluation

Evaluation을 하기 위해서는, 다음과 같은 명령어를 사용하면 된다.

 `python train.py --mode evaluation`

이 외에 mode 인자로 train 외에 다른 문자가 들어와도 evaluation이 진행된다.


* `train.py`에서 inference 클래스를 선언할 때, `save_dir` 인자를 넣어주지 않으면 faiss index를 생성한다.
만약 `save_dir`인자에 index의 경로를 넣어주면 해당 index에 대한 평가를 진행한다.



