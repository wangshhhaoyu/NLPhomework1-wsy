"""
作业一：文本分类实现
完成了三种文本表示方法的比较：Bag of Words, Word2Vec, BERT
"""

import numpy as np
import pandas as pd
import pickle
import os
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.metrics import accuracy_score, f1_score
from sklearn.preprocessing import LabelEncoder
import re
import warnings
warnings.filterwarnings('ignore')

# ==================== 数据加载与划分 ====================
print("=" * 60)
print("步骤0: 数据加载与划分")
print("=" * 60)

DATA_DIR = r"F:\360MoveData\Users\DELL\Desktop\HW-1"
nyt = pd.read_csv(os.path.join(DATA_DIR, "nyt.csv"))
ag = pd.read_csv(os.path.join(DATA_DIR, "ag.csv"))

print(f"NYT数据集: {nyt.shape[0]} 条样本")
print(f"类别分布: {nyt['label'].value_counts().to_dict()}")

# 标签编码
le = LabelEncoder()
nyt['label_encoded'] = le.fit_transform(nyt['label'])
print(f"标签映射: {dict(zip(le.classes_, le.transform(le.classes_)))}")

# 数据划分 - 统一设置random_state保证可重复性
RANDOM_STATE = 42
X_train_text, X_test_text, y_train, y_test = train_test_split(
    np.array(nyt['text'].tolist()),
    np.array(nyt['label_encoded'].tolist()),
    test_size=0.2,
    random_state=RANDOM_STATE,
    stratify=np.array(nyt['label_encoded'].tolist())
)

print(f"\n训练集大小: {len(X_train_text)}")
print(f"测试集大小: {len(X_test_text)}")

# ==================== Task 1: Bag of Words ====================
print("\n" + "=" * 60)
print("Task 1: Bag of Words")
print("=" * 60)

def evaluate_and_report(y_true, y_pred, method_name):
    """评估并报告结果"""
    acc = accuracy_score(y_true, y_pred)
    macro_f1 = f1_score(y_true, y_pred, average='macro')
    print(f"\n{method_name}:")
    print(f"  Accuracy: {acc:.4f}")
    print(f"  Macro-F1: {macro_f1:.4f}")
    return {'accuracy': acc, 'macro_f1': macro_f1}

results = {}

# 1. Binary Bag of Words
print("\n--- 1. Binary Bag of Words ---")
binary_vectorizer = CountVectorizer(binary=True, token_pattern=r'(?u)\b\w+\b')
X_train_binary = binary_vectorizer.fit_transform(X_train_text)
X_test_binary = binary_vectorizer.transform(X_test_text)

lr_binary = LogisticRegression(max_iter=1000, random_state=RANDOM_STATE)
lr_binary.fit(X_train_binary, y_train)
y_pred_binary = lr_binary.predict(X_test_binary)
results['Binary BoW'] = evaluate_and_report(y_test, y_pred_binary, "Binary Bag of Words")

# 2. Word Frequency (使用TF)
print("\n--- 2. Word Frequency ---")
tf_vectorizer = TfidfVectorizer(use_idf=False, norm=None)
X_train_tf = tf_vectorizer.fit_transform(X_train_text)
X_test_tf = tf_vectorizer.transform(X_test_text)

lr_tf = LogisticRegression(max_iter=1000, random_state=RANDOM_STATE)
lr_tf.fit(X_train_tf, y_train)
y_pred_tf = lr_tf.predict(X_test_tf)
results['Word Frequency'] = evaluate_and_report(y_test, y_pred_tf, "Word Frequency (TF)")

# 3. Count Vector
print("\n--- 3. Count Vector (词表大小) ---")
count_vectorizer = CountVectorizer(token_pattern=r'(?u)\b\w+\b')
X_train_count = count_vectorizer.fit_transform(X_train_text)
X_test_count = count_vectorizer.transform(X_test_text)

print(f"词表大小: {len(count_vectorizer.vocabulary_)}")

lr_count = LogisticRegression(max_iter=1000, random_state=RANDOM_STATE)
lr_count.fit(X_train_count, y_train)
y_pred_count = lr_count.predict(X_test_count)
results['Count Vector'] = evaluate_and_report(y_test, y_pred_count, "Count Vector")
results['Count Vector']['vocab_size'] = len(count_vectorizer.vocabulary_)

# 保存Task 1结果
with open(os.path.join(DATA_DIR, 'task1_results.pkl'), 'wb') as f:
    pickle.dump(results, f)

print("\n[Task 1 完成] 结果已保存到 task1_results.pkl")
