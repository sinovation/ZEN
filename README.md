# ZEN

ZEN is a BERT-based Chinese **(Z)** text encoder **E**nhanced by **N**-gram representations, where different combinations of characters are considered during training. The potential word or phrase boundaries are explicitly pre-trained and fine-tuned with the character encoder (BERT). ZEN incorporates the comprehensive information of both the character sequence and words or phrases it contains. ZEN is tested on a series of Chinese NLP tasks, where it requires less resource than other published encoders, and achieves state-of-the-art performance on most tasks.

![ZEN_model](http://zen.chuangxin.com/front/assets/zen.png)


## Quick tour of pre-training and fine-tune using ZEN

The library comprises several example scripts for conducting [**Chinese NLP tasks**](/datasets):

- `run_pre_train.py`: an example pre-training ZEN
- `run_sequence_level_classification.py`: an example fine-tuning ZEN on DC, SA, SPM and NLI tasks (*sequence-level classification*)
- `run_token_level_classification.py`: an example fine-tuning ZEN on CWS, POS and NER tasks (*token-level classification*)


[**Examples**](/examples) of pre-training and fine-tune using ZEN.


## Contact information

For help or issues using ZEN, please submit a GitHub issue.

For personal communication related to ZEN, please contact chenguimin(`chenguimin@chuangxin.com`).

