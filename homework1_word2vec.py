"""
作业一：Task 2 - Word2Vec 训练
"""

import numpy as np
import pandas as pd
import pickle
import os
import re
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from sklearn.preprocessing import LabelEncoder
from gensim.models import Word2Vec
import warnings
warnings.filterwarnings('ignore')

print("=" * 60)
print("Task 2: Word2Vec")
print("=" * 60)

DATA_DIR = r"F:\360MoveData\Users\DELL\Desktop\HW-1"

# 加载数据
nyt = pd.read_csv(os.path.join(DATA_DIR, "nyt.csv"))
ag = pd.read_csv(os.path.join(DATA_DIR, "ag.csv"))

# 标签编码
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

# 简单的文本分词函数
def simple_tokenize(text):
    if not isinstance(text, str):
        return []
    text = text.lower()
    tokens = re.findall(r'\b\w+\b', text)
    return tokens

def get_doc_vector(word_vectors, tokens, dim=100):
    """获取文档向量（词向量平均）"""
    vectors = []
    for token in tokens:
        if token in word_vectors:
            vectors.append(word_vectors[token])
    if len(vectors) == 0:
        return np.zeros(dim)
    return np.mean(vectors, axis=0)

def evaluate_and_report(y_true, y_pred, method_name):
    acc = accuracy_score(y_true, y_pred)
    macro_f1 = f1_score(y_true, y_pred, average='macro')
    print(f"\n{method_name}:")
    print(f"  Accuracy: {acc:.4f}")
    print(f"  Macro-F1: {macro_f1:.4f}")
    return {'accuracy': acc, 'macro_f1': macro_f1}

# 2.2 Word2Vec Trained on AG News
print("\n--- 2.2 Word2Vec Trained on AG News ---")
print("使用AG News训练Word2Vec...")

ag_tokens = [simple_tokenize(str(text)) for text in ag['text'].tolist()]
print(f"AG News文档数: {len(ag_tokens)}")

w2v_ag = Word2Vec(sentences=ag_tokens, vector_size=100, window=5, min_count=1, workers=4, epochs=10, seed=RANDOM_STATE)
print(f"AG News Word2Vec词表大小: {len(w2v_ag.wv)}")

print("生成文档向量...")
X_train_w2v_ag = np.array([get_doc_vector(w2v_ag.wv, simple_tokenize(t), 100) for t in X_train_text])
X_test_w2v_ag = np.array([get_doc_vector(w2v_ag.wv, simple_tokenize(t), 100) for t in X_test_text])

lr_w2v_ag = LogisticRegression(max_iter=1000, random_state=RANDOM_STATE)
lr_w2v_ag.fit(X_train_w2v_ag, y_train)
y_pred_w2v_ag = lr_w2v_ag.predict(X_test_w2v_ag)
results_ag = evaluate_and_report(y_test, y_pred_w2v_ag, "Word2Vec (AG News)")

# 2.3 Word2Vec Trained on NYT
print("\n--- 2.3 Word2Vec Trained on NYT ---")
print("使用NYT训练Word2Vec...")

nyt_train_tokens = [simple_tokenize(str(text)) for text in X_train_text]
print(f"NYT训练文档数: {len(nyt_train_tokens)}")

w2v_nyt = Word2Vec(sentences=nyt_train_tokens, vector_size=100, window=5, min_count=1, workers=4, epochs=10, seed=RANDOM_STATE)
print(f"NYT Word2Vec词表大小: {len(w2v_nyt.wv)}")

print("生成文档向量...")
X_train_w2v_nyt = np.array([get_doc_vector(w2v_nyt.wv, simple_tokenize(t), 100) for t in X_train_text])
X_test_w2v_nyt = np.array([get_doc_vector(w2v_nyt.wv, simple_tokenize(t), 100) for t in X_test_text])

lr_w2v_nyt = LogisticRegression(max_iter=1000, random_state=RANDOM_STATE)
lr_w2v_nyt.fit(X_train_w2v_nyt, y_train)
y_pred_w2v_nyt = lr_w2v_nyt.predict(X_test_w2v_nyt)
results_nyt = evaluate_and_report(y_test, y_pred_w2v_nyt, "Word2Vec (NYT)")

# 保存结果
task2_results = {
    'Word2Vec (AG News)': results_ag,
    'Word2Vec (NYT)': results_nyt
}

with open(os.path.join(DATA_DIR, 'task2_results.pkl'), 'wb') as f:
    pickle.dump(task2_results, f)

# 保存为CSV
import pandas as pd
rows = []
for name, res in task2_results.items():
    rows.append({
        'Task': 'Task 2: Word2Vec',
        'Method': name,
        'Accuracy': res['accuracy'],
        'Macro-F1': res['macro_f1']
    })
df = pd.DataFrame(rows)
df.to_csv(os.path.join(DATA_DIR, 'task2_results.csv'), index=False)

print("\n[Task 2 完成] 结果已保存到 task2_results.csv")
