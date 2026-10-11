"""
Task 2.1: GloVe Pre-trained Embedding
"""

import numpy as np
import pandas as pd
import os
import re
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from sklearn.preprocessing import LabelEncoder
import warnings
warnings.filterwarnings('ignore')

print("=" * 60)
print("Task 2.1: Pre-trained GloVe")
print("=" * 60)

DATA_DIR = r"F:\360MoveData\Users\DELL\Desktop\HW-1"
GLOVE_PATH = os.path.join(DATA_DIR, "glove.6B", "glove.6B.100d.txt")

# 加载数据
nyt = pd.read_csv(os.path.join(DATA_DIR, "nyt.csv"))
le = LabelEncoder()
nyt['label_encoded'] = le.fit_transform(nyt['label'])

# 使用与Task 1相同的数据划分
RANDOM_STATE = 42
X_train_text, X_test_text, y_train, y_test = train_test_split(
    np.array(nyt['text'].tolist()),
    np.array(nyt['label_encoded'].tolist()),
    test_size=0.2,
    random_state=RANDOM_STATE,
    stratify=np.array(nyt['label_encoded'].tolist())
)

# 加载GloVe词向量
print("加载GloVe词向量...")
word_vectors = {}
with open(GLOVE_PATH, 'r', encoding='utf-8') as f:
    for line in f:
        values = line.strip().split()
        word = values[0]
        vector = np.array([float(v) for v in values[1:]])
        word_vectors[word] = vector
print(f"GloVe词表大小: {len(word_vectors)}")

# 分词函数
def simple_tokenize(text):
    if not isinstance(text, str):
        return []
    text = text.lower()
    tokens = re.findall(r'\b\w+\b', text)
    return tokens

# 获取文档向量
def get_doc_vector(tokens):
    vectors = []
    for token in tokens:
        if token in word_vectors:
            vectors.append(word_vectors[token])
    if len(vectors) == 0:
        return np.zeros(100)
    return np.mean(vectors, axis=0)

# 生成文档向量
print("生成文档向量...")
X_train_glove = np.array([get_doc_vector(simple_tokenize(t)) for t in X_train_text])
X_test_glove = np.array([get_doc_vector(simple_tokenize(t)) for t in X_test_text])

# 训练分类器
print("训练分类器...")
lr_glove = LogisticRegression(max_iter=1000, random_state=RANDOM_STATE)
lr_glove.fit(X_train_glove, y_train)
y_pred = lr_glove.predict(X_test_glove)

acc = accuracy_score(y_test, y_pred)
macro_f1 = f1_score(y_test, y_pred, average='macro')

print(f"\nGloVe Pre-trained 结果:")
print(f"  Accuracy: {acc:.4f}")
print(f"  Macro-F1: {macro_f1:.4f}")

# 保存结果
import pickle
results = {'GloVe (Pre-trained)': {'accuracy': acc, 'macro_f1': macro_f1}}
with open(os.path.join(DATA_DIR, 'task2_glove_results.pkl'), 'wb') as f:
    pickle.dump(results, f)

import pandas as pd
rows = [{'Task': 'Task 2: Word2Vec', 'Method': 'GloVe (Pre-trained)', 'Accuracy': acc, 'Macro-F1': macro_f1}]
df = pd.DataFrame(rows)
df.to_csv(os.path.join(DATA_DIR, 'task2_glove_results.csv'), index=False)
print("\n结果已保存到 task2_glove_results.csv")
