# BERT-like MLM Training Procedure
A repository to train a transformer by masking random tokens in the sequence

The repo expects the data in the following format: a text file with integers of the embedding tokens, separated by commas. This repo does not contain any tokenizing procedure.

Edit the config in args.yaml, then run the following.
```
python3 main.py
```
