"""
作业一：Task 3 - BERT Fine-tuning
"""

import os
import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, f1_score
from transformers import BertTokenizer, BertForSequenceClassification, Trainer, TrainingArguments
from torch.utils.data import Dataset
import warnings
warnings.filterwarnings('ignore')

print("=" * 60)
print("Task 3: BERT Fine-tuning")
print("=" * 60)

DATA_DIR = r"F:\360MoveData\Users\DELL\Desktop\HW-1"
BERT_MODEL_PATH = r"F:\360MoveData\Users\DELL\Desktop\HW-1\bert-base-uncased"

# 检查GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"使用设备: {device}")

# 加载数据
nyt = pd.read_csv(os.path.join(DATA_DIR, "nyt.csv"))

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

print(f"训练集大小: {len(X_train_text)}")
print(f"测试集大小: {len(X_test_text)}")

# 自定义数据集
class TextDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_length=64):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = str(self.texts[idx])
        label = self.labels[idx]

        encoding = self.tokenizer(
            text,
            add_special_tokens=True,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )

        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.tensor(label, dtype=torch.long)
        }

# 加载BERT - 从本地加载
print("\n加载 BERT 模型...")
tokenizer = BertTokenizer.from_pretrained(BERT_MODEL_PATH)
model = BertForSequenceClassification.from_pretrained(BERT_MODEL_PATH, num_labels=3)
model.to(device)

# 创建数据集
print("创建数据集...")
train_dataset = TextDataset(X_train_text, y_train, tokenizer, max_length=64)
test_dataset = TextDataset(X_test_text, y_test, tokenizer, max_length=64)

# 训练参数
training_args = TrainingArguments(
    output_dir='./bert_output',
    num_train_epochs=3,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=32,
    warmup_steps=500,
    weight_decay=0.01,
    logging_steps=100,
    eval_strategy="no",
    save_strategy="no",
    report_to="none"
)

# Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
)

# 训练
print("\n开始训练 BERT...")
trainer.train()

# 预测
print("\n在测试集上预测...")
model.eval()
all_preds = []

from torch.utils.data import DataLoader

test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

for batch in test_loader:
    input_ids = batch['input_ids'].to(device)
    attention_mask = batch['attention_mask'].to(device)

    with torch.no_grad():
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        logits = outputs.logits
        preds = torch.argmax(logits, dim=1).cpu().numpy()
        all_preds.extend(preds)

# 评估
y_pred_bert = np.array(all_preds)
acc_bert = accuracy_score(y_test, y_pred_bert)
macro_f1_bert = f1_score(y_test, y_pred_bert, average='macro')

print(f"\nBERT Fine-tuning 结果:")
print(f"  Accuracy: {acc_bert:.4f}")
print(f"  Macro-F1: {macro_f1_bert:.4f}")

# 保存结果
import pickle
task3_results = {
    'BERT (Fine-tuned)': {'accuracy': acc_bert, 'macro_f1': macro_f1_bert}
}

with open(os.path.join(DATA_DIR, 'task3_results.pkl'), 'wb') as f:
    pickle.dump(task3_results, f)

import pandas as pd
rows = []
for name, res in task3_results.items():
    rows.append({
        'Task': 'Task 3: BERT',
        'Method': name,
        'Accuracy': res['accuracy'],
        'Macro-F1': res['macro_f1']
    })
df = pd.DataFrame(rows)
df.to_csv(os.path.join(DATA_DIR, 'task3_results.csv'), index=False)

print("\n[Task 3 完成] 结果已保存到 task3_results.csv")
