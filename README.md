# 文本分类实验

## 环境配置

```bash
pip install numpy pandas scikit-learn gensim transformers torch accelerate
```

## 数据集

- **nyt.csv**: NYT新闻数据集（11,519条，3个类别：体育、政治、商业）
- **ag.csv**: AG新闻数据集（90,000条，用于训练Word2Vec）

## 运行说明

### Task 1: 词袋模型
```bash
python homework1.py
```
结果文件: `task1_results.csv`

### Task 2.1: GloVe预训练词向量
从 https://nlp.stanford.edu/data/glove.6B.zip 下载GloVe.6B，解压到 `glove.6B/` 文件夹
```bash
python homework1_glove.py
```
结果文件: `task2_glove_results.csv`

### Task 2.2 & 2.3: Word2Vec训练
```bash
python homework1_word2vec.py
```
结果文件: `task2_results.csv`

### Task 3: BERT微调
从 https://huggingface.co/google-bert/bert-base-uncased 下载BERT模型到 `bert-base-uncased/` 文件夹
```bash
python homework1_bert.py
```
结果文件: `task3_results.csv`

## 实验结果

| 任务 | 方法 | Accuracy | Macro-F1 |
|------|------|----------|----------|
| Task 1 | 词频 | 0.9870 | 0.9687 |
| Task 2 | GloVe (预训练) | 0.9813 | 0.9573 |
| Task 3 | BERT (微调) | 0.9766 | 0.9469 |

## 注意事项

- 随机种子: 42（固定以保证可重复性）
- 训练/测试集划分: 80%/20%
- Word2Vec维度: 100
- BERT序列长度: 64，训练轮数: 3
