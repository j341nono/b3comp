from transformers.trainer_utils import set_seed
set_seed(42)

import re
from sudachipy import tokenizer
from sudachipy import dictionary
import pandas as pd
import numpy as np
from pprint import pprint
import torch
import re
import emoji
import neologdn
from bs4 import BeautifulSoup
import unicodedata
import json

train_path = './data/train4.json'
valid_path = './data/valid4.json'
test_path = './data/test4.json'

def clean_text(text):
    """
    テキストの前処理を行う関数
    """
    # HTMLタグ除去
    text = BeautifulSoup(text, "html.parser").get_text().strip()
    # 正規化
    text = neologdn.normalize(text)
    text = unicodedata.normalize('NFKC', text)
    # URL除去
    text = re.sub(r'https?://[\w/:%#\$&\?\(\)~\.=\+\-]+', '', text)
    # メンション・ハッシュタグ除去
    text = re.sub(r'@[\w_]+', '', text)
    text = re.sub(r'#[\w一-龠ぁ-んァ-ヴーａ-ｚＡ-Ｚ0-9０-９]+', '', text)
    # 絵文字を文字に変換（または除去）
    text = emoji.replace_emoji(text, '絵文字')
    # リツイート表記の削除
    text = re.sub(r'^RT[:：]', '', text)
    # 特殊文字の削除
    text = re.sub(r'[［\]「」『』【】〈〉《》〔〕]', '', text)
    # 連続した空白・句読点の正規化
    text = re.sub(r'\s+', ' ', text)
    text = re.sub(r'。+', '。', text)
    text = re.sub(r'、+', '、', text)
    text = re.sub(r'！+', '！', text)
    text = re.sub(r'？+', '？', text)
    # 数字の正規化
    text = re.sub(r'\d+', '0', text)
    # 前後の空白を削除
    return text.strip()

def validate_text(text, min_length=8, max_length=70):
    """
    テキストの有効性を判定する関数
    """
    return (
        min_length <= len(text) <= max_length and
        len(set(text)) >= 3 and  # ユニークな文字が3文字以上
        bool(re.search(r'[ぁ-んァ-ン一-龠]', text))  # 日本語文字を含む
    )

def load_json_data(file_path, is_test=False, is_valid=False):
    with open(file_path, 'r', encoding='utf-8') as f:
        texts = []
        for text in f:
            text_dict = json.loads(text.strip())
            
            text = text_dict['sentence']
            # ノイズ除去と正規化
            text = clean_text(text)

            if is_test: 
                text = {
                    'sentence': text, 
                    'predicted_genre': text_dict["predicted_genre"], 
                    "id":text_dict["id"]
                }
                texts.append(text)
            elif is_valid:        
                text = {
                    'sentence': text,
                    'writer_sentiment': text_dict['writer_sentiment'],
                    'predicted_genre': text_dict["predicted_genre"],
                    "id":text_dict["id"]
                }
                texts.append(text)
            else:
                if validate_text(text):
                    text = {
                        'sentence': text,
                        'writer_sentiment': text_dict['writer_sentiment'],
                        'predicted_genre': text_dict["predicted_genre"], 
                        "id":text_dict["id"]
                    }
                    texts.append(text)
        return texts

train_dataset = load_json_data(train_path)
valid_dataset = load_json_data(valid_path, is_valid=True)
test_dataset = load_json_data(test_path, is_test=True)


from transformers import AutoTokenizer, AutoModelForMaskedLM

model_name = "studio-ousia/luke-japanese-large-lite"
tokenizer = AutoTokenizer.from_pretrained(model_name)

def preprocess_text_classification(texts):
    encoded_texts = []
    for text in texts:
        encoded = tokenizer(text['sentence'], max_length=32)
        text['writer_sentiment'] += 2
        encoded['label'] = text['writer_sentiment']
        encoded_texts.append(encoded)
    return encoded_texts

def preprocess_text_classification_test(texts):
    encoded_texts = []
    for text in texts:
        encoded = tokenizer(text['sentence'], max_length=32)
        encoded_texts.append(encoded)
    return encoded_texts


# データセット全体をトーク内ゼーションする処理
encoded_train_dataset = preprocess_text_classification(train_dataset)
encoded_valid_dataset = preprocess_text_classification(valid_dataset)

encoded_test_dataset = preprocess_text_classification_test(test_dataset)


from transformers import DataCollatorWithPadding
data_collator = DataCollatorWithPadding(tokenizer = tokenizer)

sentences = [item['writer_sentiment'] for item in train_dataset]


from transformers import AutoModelForSequenceClassification, AutoConfig

class_label = ['0', '1', '2', '3', '4']

label2id = {label: id for id, label in enumerate(class_label)}
id2label = {id: label for id, label in enumerate(class_label)}

config = AutoConfig.from_pretrained(model_name, num_labels=5)
config.hidden_dropout_prob = 0.2  # デフォルトは0.1
config.attention_probs_dropout_prob = 0.2 ## 0.2

model = AutoModelForSequenceClassification.from_pretrained(
    model_name,
    config=config,
    ignore_mismatched_sizes=True,
)
device = torch.device("cuda:0")
model = model.to(device)

# 学習に関するクラス
from transformers import TrainingArguments

training_args = TrainingArguments(
    output_dir = './output/luke',
    per_device_train_batch_size = 64,
    per_device_eval_batch_size = 128,
    learning_rate = 1e-5,
    lr_scheduler_type = 'cosine',  
    optim="adamw_torch_fused",
    warmup_ratio = 0.1,
    weight_decay=0.01,
    max_grad_norm=1.0,
    num_train_epochs = 30,
    save_strategy = 'epoch',
    logging_strategy = 'epoch',
    evaluation_strategy = 'epoch',
    load_best_model_at_end = True,
    metric_for_best_model = 'eval_qwk',
    fp16 = False,
    bf16 = True,
    report_to = ["tensorboard"],
    save_total_limit = 1,
    eval_steps = 200,
    logging_steps = 200,
)


from sklearn.metrics import cohen_kappa_score
import numpy as np

def compute_qwk(res):
    predictions, labels = res
    logit = np.argmax(predictions, axis=1)
    if len(labels.shape) > 1:  # ワンホットエンコーディング
        labels = np.argmax(labels, axis=1)
    qwk = cohen_kappa_score(labels, logit, weights="quadratic")
    return {'eval_qwk': qwk}

from transformers import Trainer
from transformers import EarlyStoppingCallback

import torch
import torch.nn as nn
import torch.nn.functional as F

class QWKLoss(nn.Module):
    def __init__(self, num_classes, eps=1e-10, scaling_factor=1.0): # 1
        super().__init__()
        self.num_classes = num_classes
        self.eps = eps
        self.scaling_factor = scaling_factor
        
        # 重み行列の初期化
        self.weights = torch.zeros((num_classes, num_classes))
        for i in range(num_classes):
            for j in range(num_classes):
                self.weights[i, j] = (i - j) ** 2
        
        # 必要に応じてクラスの重みを追加
        self.class_weights = None
    
    def forward(self, logits, targets):
        self.weights = self.weights.to(logits.device)
        probs = F.softmax(logits, dim=1)
        targets_one_hot = F.one_hot(targets, num_classes=self.num_classes).float()
        
        # バッチ統計量の計算
        O = torch.matmul(targets_one_hot.t(), probs)
        target_dist = targets_one_hot.sum(0).view(-1, 1)
        pred_dist = probs.sum(0).view(1, -1)
        E = torch.matmul(target_dist, pred_dist)
        
        N = torch.sum(O)
        if N == 0:
            return torch.tensor(0.0, device=logits.device)
            
        O = O / N
        E = E / N
        
        num = torch.sum(self.weights * O)
        den = torch.sum(self.weights * E)
        
        # 数値安定性の向上
        kappa = 1 - (num / (den + self.eps))
        loss = (1 - kappa) * self.scaling_factor
        
        # クラスの重みづけ（必要な場合）
        if self.class_weights is not None:
            loss = loss * (targets_one_hot * self.class_weights).sum(dim=1).mean()
            
        return loss

class CustomTrainer(Trainer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.qwk_loss = QWKLoss(num_classes=5)  # Adjust number of classes as needed
        
    def compute_loss(self, model, inputs, *args, **kwargs):
        labels = inputs.get("labels")
        outputs = model(**inputs)
        logits = outputs.logits
        
        # Calculate QWK loss
        loss = self.qwk_loss(logits, labels)
        
        return (loss, outputs) if kwargs.get("return_outputs", False) else loss
           
trainer = CustomTrainer(
    model = model,
    train_dataset = encoded_train_dataset,
    eval_dataset = encoded_valid_dataset,
    data_collator = data_collator,
    args = training_args,
    compute_metrics = compute_qwk,
    callbacks=[EarlyStoppingCallback(early_stopping_patience=3)]
)
print("訓練を開始")
trainer.train()
print("学習が終了")

from pprint import pprint
eval_metrics = trainer.evaluate(encoded_valid_dataset)
pprint(eval_metrics)

test_predict = []
pred_result = trainer.predict(encoded_test_dataset, ignore_keys=['loss', 'last_hidden_state', 'hidden_states', 'attentions'])
test_predict = pred_result.predictions.argmax(axis=1).tolist()
test_predict = [x-2 for x in test_predict]
# print(type(test_predict))
with open('./save/luke.txt', 'w') as f:
    f.write("\n".join(map(str, test_predict)))


# 予測結果を取得（logits）
logits = pred_result.predictions  # shape: (num_samples, num_classes)

# **検証データの推論結果を取得**
valid_pred_result = trainer.predict(encoded_valid_dataset)  # モデルで推論
valid_logits = valid_pred_result.predictions  # shape: (num_samples, num_classes)

def save_predictions(dataset, predictions, filename, is_valid=False):
    jsonl_data = []
    for i, logit in enumerate(predictions):
        logit_dict = {str(2 - j): float(logit[j]) for j in range(len(logit))}

        if is_valid:
            entry = {
                **logit_dict,
                "pred": test_predict[i],
                "predicted_genre": dataset[i]["predicted_genre"],
                "id": dataset[i]["id"],
                "true": dataset[i]["writer_sentiment"]
            }
        else:
            entry = {
                **logit_dict,
                "pred": test_predict[i],
                "predicted_genre": dataset[i]["predicted_genre"],
                "id": dataset[i]["id"],
            }
        jsonl_data.append(entry)

    with open(filename, "w", encoding="utf-8") as f:
        for entry in jsonl_data:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")

    print(f"JSONL ファイルを保存しました: {filename}")
    
    
# テストデータの結果を保存
test_jsonl_path = "./save/luke_test.jsonl"
save_predictions(test_dataset, logits, test_jsonl_path)

# 検証データの結果を保存（変数を適宜修正）
valid_logits = valid_pred_result.predictions  # 検証データ用の logits
valid_jsonl_path = "./save/luke_valid.jsonl"
save_predictions(valid_dataset, valid_logits, valid_jsonl_path, is_valid=True)