import json
import jieba
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import ComplementNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

#讀取 JSON
with open("problem.json", "r", encoding="utf-8") as file:
    data = json.load(file)

# 準備數據
dataset_dict = {"question": [], "category": [], "keywords": []}
for item in data:
    dataset_dict["question"].append(item["question"])
    dataset_dict["category"].append(item["category"])
    dataset_dict["keywords"].append(item["keywords"])

df = pd.DataFrame(dataset_dict)

# 斷詞處理
def jieba_cut(text):
    return " ".join(jieba.cut(text)) if isinstance(text, str) else ""

df["processed_text"] = df["question"].apply(jieba_cut)

# 結合問題與關鍵字
df["combined_text"] = df["processed_text"] + " " + df["keywords"].apply(lambda x: " ".join(x) if isinstance(x, list) else "")

# 處理 NaN
df["combined_text"] = df["combined_text"].fillna("")

# 讀取停用詞表
stopwords = set()
with open("cn_stopwords.txt", "r", encoding="utf-8") as f:
    stopwords = [line.strip() for line in f if line.strip()]

print(f"停用詞表長度: {len(stopwords)}")

# 設定 TF-IDF
vectorizer = TfidfVectorizer(
    stop_words=stopwords,
    ngram_range=(1, 2),
    max_df=0.9, min_df=1
)

X = vectorizer.fit_transform(df["combined_text"])
y = df["category"]

# 拆分訓練集與測試集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 訓練 ComplementNB 模型
model = ComplementNB()
model.fit(X_train, y_train)

# 測試分類效果
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"分類準確率: {accuracy:.2f}")

# 測試新問題
new_question = ["該找他復合嗎？"]
new_X = vectorizer.transform([jieba_cut(new_question[0])])
predicted_category = model.predict(new_X)
print(f"問題分類: {predicted_category[0]}")


import joblib

# 儲存模型與 TF-IDF 向量化器
joblib.dump(model, "naive_bayes_model.pkl")      # 儲存 Naïve Bayes 模型
joblib.dump(vectorizer, "tfidf_vectorizer.pkl")  # 儲存 TF-IDF 向量化器

print("模型和向量化器已成功儲存！")
