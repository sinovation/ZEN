# ZEN

## Introduction

ZEN, a BERT-based Chinese **(Z)** text encoder **E**nhanced by **N**-gram representations, where different combinations of characters are considered during training. The potential word or phrase boundaries are explicitly pre-trained and fine-tuned with the character encoder (BERT). ZEN incorporates the comprehensive information of both the character sequence and words or phrases it contains. ZEN is tested on a series of Chinese NLP tasks, where it requires less resource than other published encoders, and achieves state-of-the-art performance on most tasks.

## Quick tour of pre-training and fine-tune using ZEN

The library comprises several example scripts for conducting Chinese NLP tasks:

- `run_pre_train.py`: an example pre-training ZEN
- `run_sequence_level_classification.py`: an example fine-tuning ZEN on DC, SA, SPM and NLI tasks (*sequence-level classification*)
- `run_token_level_classification.py`: an example fine-tuning ZEN on CWS, POS and NER tasks (*token-level classification*)

Three quick usage examples for these scripts:

### `run_pre_train.py`: Pre-train ZEN model from scratch or BERT model

```shell
python run_pre_train.py  \
    --pregenerated_data /path/to/pregenerated_data   \
    --bert_model /path/to/bert_model  \
    --do_lower_case  \
    --output_dir /path/to/output_dir   \
    --epochs 20  \
    --train_batch_size 128   \
    --reduce_memory  \
    --fp16  \
    --scratch  \
    --save_name ZEN_pretrain_base_
```

### `run_sequence_level_classification.py`: Fine-tune on tasks for sequence classification

```shell
python run_sequence_level_classification.py \
    --task_name TASKNAME \
    --do_train \
    --do_eval \
    --do_lower_case \
    --data_dir /path/to/dataset \
    --bert_model /path/to/zen_model \
    --max_seq_length 512 \
    --train_batch_size 32 \
    --learning_rate 2e-5 \
    --num_train_epochs 30.0
```
where TASKNAME can be one of DC, SA, SPM and NLI

script of fine-tuning thucnews
```shell
python run_sequence_level_classification.py \
    --task_name thucnews \
    --do_train \
    --do_eval \
    --do_lower_case \
    --data_dir /path/to/dataset/thucnews \
    --bert_model /path/to/zen_model \
    --max_seq_length 512 \
    --train_batch_size 32 \
    --learning_rate 2e-5 \
    --num_train_epochs 30.0
```

script of fine-tuning chnsenticorp
```shell
python run_sequence_level_classification.py \
    --task_name ChnSentiCorp \
    --do_train \
    --do_eval \
    --do_lower_case \
    --data_dir /path/to/dataset/ChnSentiCorp \
    --bert_model /path/to/zen_model \
    --max_seq_length 512 \
    --train_batch_size 32 \
    --learning_rate 2e-5 \
    --num_train_epochs 30.0
```

script of fine-tuning LCQMC
```shell
python run_sequence_level_classification.py \
    --task_name lcqmc \
    --do_train \
    --do_eval \
    --do_lower_case \
    --data_dir /path/to/dataset/lcqmc \
    --bert_model /path/to/zen_model \
    --max_seq_length 128 \
    --train_batch_size 128 \
    --learning_rate 5e-5 \
    --num_train_epochs 30.0
```

script of fine-tuning XNLI
```shell
python run_sequence_level_classification.py \
    --task_name xnli \
    --do_train \
    --do_eval \
    --do_lower_case \
    --data_dir /path/to/dataset/xnli \
    --bert_model /path/to/zen_model \
    --max_seq_length 128 \
    --train_batch_size 128 \
    --learning_rate 5e-5 \
    --num_train_epochs 30.0
```


### `run_token_level_classification.py`: Fine-tune on tasks for sequence classification

```shell
python run_token_level_classification.py \
    --task_name TASKNAME \
    --do_train \
    --do_eval \
    --do_lower_case \
    --data_dir /path/to/dataset \
    --bert_model /path/to/zen_model \
    --max_seq_length 128 \
    --do_train  \
    --do_eval \
    --train_batch_size 128 \
    --num_train_epochs 30 \
    --warmup_proportion 0.1
```
where TASKNAME can be one of CWS, POS and NER

script of fine-tuning msra
```shell
python run_token_level_classification.py \
    --task_name cwsmsra \
    --do_train \
    --do_eval \
    --do_lower_case \
    --data_dir /path/to/dataset \
    --bert_model /path/to/zen_model \
    --max_seq_length 256 \
    --do_train  \
    --do_eval \
    --train_batch_size 96 \
    --num_train_epochs 30 \
    --warmup_proportion 0.1
```

script of fine-tuning CTB5
```shell
python run_token_level_classification.py \
    --task_name pos \
    --do_train \
    --do_eval \
    --do_lower_case \
    --data_dir /path/to/dataset \
    --bert_model /path/to/zen_model \
    --max_seq_length 256 \
    --do_train  \
    --do_eval \
    --train_batch_size 96 \
    --num_train_epochs 30 \
    --warmup_proportion 0.1
```

script of fine-tuning msra_ner
```shell
python run_token_level_classification.py \
    --task_name msra \
    --do_train \
    --do_eval \
    --do_lower_case \
    --data_dir /path/to/dataset \
    --bert_model /path/to/zen_model \
    --max_seq_length 128 \
    --do_train  \
    --do_eval \
    --train_batch_size 128 \
    --num_train_epochs 30 \
    --warmup_proportion 0.1
```

## Datasets used in our experiments


### Chinese word segmentation (CWS):
[CWS dataset](http://sighan.cs.uchicago.edu/bakeoff2005/)
MSR dataset from SIGHAN2005 Chinese word segmentation Bakeoff.

### Part-of-speech (POS) tagging:
CTB5 (Xue et al., 2005) dataset with standard splits from [CTB5 dataset](https://catalog.ldc.upenn.edu/LDC2005T01)


### Named entity recognition (NER):
MSRA dataset from international Chinese language
processing Bakeoff 2006. [**NER**](http://sighan.cs.uchicago.edu/bakeoff2006/)


### Document classification (DC):
THUCNews (News) dataset (Sun et al., 2016) from Sina
news with 10 evenly distributed classes.[**THUCNews**](http://thuctc.thunlp.org)


### Sentiment analysis (SA):
The ChnSentiCorp (CSC) dataset with 12,000 documents from three domains, i.e., book, computer and hotel.
[**ChnSentiCorp**](https://github.com/pengming617/bert_classification)

### Sentence pair matching (SPM):
The LCQMC (a large-scale Chinese question matching corpus) proposed by Liu et al. (2018), where each
instance is a pair of two sentences with a label
indicating whether their intent is matched.
[**LCQMC**](http://icrc.hitsz.edu.cn/info/1037/1146.htm)

### Natural language inference (NLI):
The Chinese part of the XNLI (Conneau et al., 2018) [**XNLI**](https://github.com/google-research/bert/blob/master/multilingual.md)


