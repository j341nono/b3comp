import json
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import cohen_kappa_score
from xgboost import XGBClassifier

# JSONLファイルのパス
jsonl_files = {
    "luke": "./emsamble_data_valid/luke_valid.jsonl",
    "mb": "./emsamble_data_valid/mb_valid.jsonl",
    "swallow": "./emsamble_data_valid/swallow_valid.jsonl",
    "qwen": "./emsamble_data_valid/qwen_valid.jsonl",
}

# 検証データのロジットを格納
valid_data = []

# 各モデルの検証データを読み込んで統合
for model_name, file_path in jsonl_files.items():
    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            data = json.loads(line)
            sample_id = data["id"]
            
            # 初回のみエントリ作成
            if not any(d["id"] == sample_id for d in valid_data):
                valid_data.append({
                    "id": sample_id,
                    "true": data["true"],  # 正解ラベル
                    "predicted_genre": data["predicted_genre"]
                })
            
            # 各ラベルのロジットを保存
            for entry in valid_data:
                if entry["id"] == sample_id:
                    for label in ["-2", "-1", "0", "1", "2"]:
                        entry[f"{model_name}_logit_{label}"] = data[label]
                    break

# pandas DataFrame に変換
valid_df = pd.DataFrame(valid_data)
valid_df = valid_df.sort_values(by="id").reset_index(drop=True)


# 変換
valid_df["predicted_genre"] = valid_df["predicted_genre"].astype("category")
X = valid_df.drop(columns=["id", "true"])
# X = valid_df.drop(columns=["id", "true", "predicted_genre"])


y = valid_df["true"]


# 訓練データと検証データに分割
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# XGBoostのモデルを訓練
# xgb = XGBClassifier(n_estimators=100, max_depth=3, learning_rate=0.05, random_state=42)
xgb = XGBClassifier(n_estimators=100, max_depth=3, learning_rate=0.05, random_state=42, enable_categorical=True)
xgb.fit(X_train, y_train)

# 検証データで予測
y_pred = xgb.predict(X_val)

# QWKスコアの計算
qwk = cohen_kappa_score(y_val, y_pred, weights="quadratic")
print(f"XGBoost後のQWK: {qwk:.4f}")


# print(xgb.feature_importances_)

# import matplotlib.pyplot as plt

# plt.figure(figsize=(10, 6))  # 図のサイズを指定（オプション）
# plt.barh(X.columns, xgb.feature_importances_)
# plt.xlabel("Feature Importance")
# plt.ylabel("Features")
# plt.title("Feature Importance of XGBoost Model")
# plt.show()  # これを追加




# JSONLファイル（テストデータ版）
jsonl_files_test = {
    "luke": "./emsamble_data_test/luke_test.jsonl",
    "mb": "./emsamble_data_test/mb_test.jsonl",
    "swallow": "./emsamble_data_test/swallow_test.jsonl",
    "qwen": "./emsamble_data_test/qwen_test.jsonl",
}

# テストデータのロジットを格納
test_data = []

for model_name, file_path in jsonl_files_test.items():
    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            data = json.loads(line)
            sample_id = data["id"]
            
            # 初回のみエントリ作成
            if not any(d["id"] == sample_id for d in test_data):
                test_data.append({
                    "id": sample_id,
                    "predicted_genre": data["predicted_genre"]
                })
            
            # 各ラベルのロジットを保存
            for entry in test_data:
                if entry["id"] == sample_id:
                    for label in ["-2", "-1", "0", "1", "2"]:
                        entry[f"{model_name}_logit_{label}"] = data[label]
                    break

# pandas DataFrame に変換
test_df = pd.DataFrame(test_data)
test_df = test_df.sort_values(by="id").reset_index(drop=True)

# 特徴量のみ抽出
test_df["predicted_genre"] = test_df["predicted_genre"].astype("category")
# X_test = test_df.drop(columns=["id", "predicted_genre"])
X_test = test_df.drop(columns=["id"])


# スタッキングモデルで予測
test_pred = xgb.predict(X_test)

test_pred_mapped = test_pred - 2

# 予測結果を保存
test_df["pred"] = test_pred
test_df.to_json("xgboost_test_predictions.jsonl", orient="records", force_ascii=False, lines=True)

print("テストデータのスタッキングアンサンブル予測を保存しました。")

# 予測結果をテキストファイルに保存
with open("xgb_predictions.txt", "w", encoding="utf-8") as f:
    for label in test_df["pred"]:
        f.write(f"{label}\n")

print("予測結果を保存しました。")