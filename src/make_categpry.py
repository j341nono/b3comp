import pandas as pd
import numpy as np
import torch
import re
import logging
import json
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModel

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

SENTENCE_MODEL_NAME = "sentence-transformers/paraphrase-multilingual-mpnet-base-v2"
BERT_MODEL_NAME = "nlp-waseda/roberta-large-japanese"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

def load_json(train):
    logger.info("Loading JSON data...")
    try:
        train_df = pd.read_json(train, lines=True)
        train_df['sentence'] = train_df['sentence'].str.replace('\n', ' ')
        if 'writer_sentiment' in train_df.columns:
            train_df['writer_sentiment'] = train_df['writer_sentiment']
        train_data = [row.to_dict() for _, row in train_df.iterrows()]
        logger.info(f"Loaded {len(train_data)} entries")
        return train_data
    except Exception as e:
        logger.error(f"Error loading JSON: {str(e)}")
        raise

def get_enhanced_embedding(text, sentence_model, bert_model, bert_tokenizer):
    try:
        st_embedding = sentence_model.encode([text], convert_to_numpy=True)
        inputs = bert_tokenizer(text, return_tensors="pt", truncation=True, max_length=512).to(DEVICE)
        with torch.no_grad():
            outputs = bert_model(**inputs)

        bert_embedding = outputs.last_hidden_state[:, 0, :].cpu().numpy()  # (1, 1024) かも

        # RoBERTa の埋め込みを 768次元に縮小
        if bert_embedding.shape[1] > 768:
            bert_embedding = bert_embedding[:, :768]  # 1024次元の最初の768次元を使用

        bert_embedding = bert_embedding / np.linalg.norm(bert_embedding, axis=1, keepdims=True)

        # 重み付き結合
        combined_embedding = 0.7 * st_embedding + 0.3 * bert_embedding
        combined_embedding = combined_embedding / np.linalg.norm(combined_embedding)

        return combined_embedding

    except Exception as e:
        logger.error(f"Error in enhanced embedding: {str(e)}")
        return None


def process_batch_efficient(texts, sentence_model, bert_model, bert_tokenizer, batch_size=16):
    logger.info(f"Processing batch of {len(texts)} texts")
    embeddings = []
    
    for i in range(0, len(texts), batch_size):
        if i % 100 == 0:
            logger.info(f"Processing batch {i}/{len(texts)}")
            
        batch = texts[i:i + batch_size]
        batch_embeddings = []
        
        for text in batch:
            embedding = get_enhanced_embedding(
                text["sentence"],
                sentence_model,
                bert_model,
                bert_tokenizer
            )
            
            if embedding is None:
                embedding = sentence_model.encode([text["sentence"]], convert_to_numpy=True)
            
            batch_embeddings.append(embedding)
        
        embeddings.extend(batch_embeddings)
    
    return np.vstack(embeddings)

def preprocess_text(text):
    p_text = re.sub(r'[^\w\s.,!?！？。、（）｛｝「」『』〜ー]', '', text)
    p_text = re.sub(r'([.!?！？。、]){3,}', r'\1\1', p_text)
    p_text = re.sub(r'(.)\1{2,}', r'\1\1', p_text)
    p_text = ' '.join(p_text.split())
    return p_text if p_text.strip() else text

def save_results_to_jsonl(data, similarity_matrix, group_names, output_file):
    logger.info(f"Saving results to {output_file}")

    results = []
    
    for idx, (item, similarities) in enumerate(zip(data, similarity_matrix)):
        best_genre_idx = np.argmax(similarities)
        best_genre = group_names[best_genre_idx]
        best_similarity = float(similarities[best_genre_idx])

        output_data = {
            'id': idx,
            'sentence': item['sentence'],
            'writer_sentiment': item.get('writer_sentiment', None),
            'predicted_genre': best_genre,
            'genre_similarity': best_similarity
        }

        results.append(output_data)

    # 各ジャンルごとにソート
    sorted_results = sorted(results, key=lambda x: (-x['genre_similarity'], x['predicted_genre']))

    with open(output_file, 'w', encoding='utf-8') as f:
        for entry in sorted_results:
            json.dump(entry, f, ensure_ascii=False)
            f.write('\n')

    logger.info("Results saved successfully")


def main(input_file, data2_file, output_file):
    logging.info("Starting main process")
    
    # データロード
    data = load_json(input_file)
    data2 = load_json(data2_file)
    
    logging.info("Initializing models")
    sentence_model = SentenceTransformer("all-MiniLM-L6-v2")
    
    # テキスト前処理
    for item in data:
        item["sentence"] = preprocess_text(item["sentence"])
    for item in data2:
        item["sentence"] = preprocess_text(item["sentence"])
    
    # 埋め込み計算
    logging.info("Encoding sentences")
    data_embeddings = sentence_model.encode([item["sentence"] for item in data], convert_to_numpy=True)
    data2_embeddings = sentence_model.encode([item["sentence"] for item in data2], convert_to_numpy=True)
    
    # 類似度計算
    logging.info("Calculating similarities")
    similarity_matrix = np.dot(data_embeddings, data2_embeddings.T)
    
    # 各 data の文章に対し、data2 から writer_sentiment のラベルごとに上位2つを選択
    for i, item in enumerate(data):
        candidates = list(zip(data2, similarity_matrix[i]))
        candidates.sort(key=lambda x: x[1], reverse=True)  # 類似度の降順
        
        # writer_sentiment ごとに2つずつ選択
        sentiment_groups = {s: [] for s in range(-2, 3)}
        
        for entry, sim in candidates:
            sentiment = entry["writer_sentiment"]
            if len(sentiment_groups[sentiment]) < 2:
                sentiment_groups[sentiment].append({
                    "sentence": entry["sentence"],
                    "writer_sentiment": sentiment
                })
            
            if sum(len(v) for v in sentiment_groups.values()) >= 10:
                break
        
        # 10個の類似する文章＆ラベルを追加し、元の順番の ID も保存
        item["id"] = i
        item["matched"] = [
            {"sentence": entry["sentence"], "writer_sentiment": entry["writer_sentiment"]}
            for entries in sentiment_groups.values() for entry in entries
        ]
    
    with open(output_file, "w", encoding="utf-8") as f:
        for entry in data:
            json.dump(entry, f, ensure_ascii=False)
            f.write("\n")

if __name__ == "__main__":
    input_file = "./data/test.json"
    data2_file = "./data/train_1000.json" # 10個の文章を抽出される方
    output_file = "./save/tset10.json"
    main(input_file, data2_file, output_file)
