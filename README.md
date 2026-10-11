# Text Classification Experiment

## Environment Setup

```bash
pip install numpy pandas scikit-learn gensim transformers torch accelerate
```

## Dataset

- **nyt.csv**: NYT news dataset (11,519 samples, 3 classes: sports, politics, business)
- **ag.csv**: AG News dataset (90,000 samples, for Word2Vec training)

## Running Instructions

### Task 1: Bag of Words
```bash
python homework1.py
```
Results: `task1_results.csv`

### Task 2.1: GloVe Pre-trained Embedding
Download GloVe.6B from https://nlp.stanford.edu/data/glove.6B.zip, extract to `glove.6B/` folder
```bash
python homework1_glove.py
```
Results: `task2_glove_results.csv`

### Task 2.2 & 2.3: Word2Vec Training
```bash
python homework1_word2vec.py
```
Results: `task2_results.csv`

### Task 3: BERT Fine-tuning
Download BERT model from https://huggingface.co/google-bert/bert-base-uncased to `bert-base-uncased/` folder
```bash
python homework1_bert.py
```
Results: `task3_results.csv`

## Results Summary

| Task | Method | Accuracy | Macro-F1 |
|------|--------|----------|----------|
| Task 1 | Word Frequency | 0.9870 | 0.9687 |
| Task 2 | GloVe (Pre-trained) | 0.9813 | 0.9573 |
| Task 3 | BERT (Fine-tuned) | 0.9766 | 0.9469 |

## Notes

- Random seed: 42 (fixed for reproducibility)
- Train/Test split: 80%/20%
- Word2Vec dimension: 100
- BERT max_length: 64, epochs: 3
