import joblib
import jieba

# 載入模型與向量化器
model = joblib.load("naive_bayes_model.pkl")
vectorizer = joblib.load("tfidf_vectorizer.pkl")

# 測試新問題
new_question = "我的下一任甚麼時候會出現？"

# 進行斷詞
def jieba_cut(text):
    return " ".join(jieba.cut(text))

new_X = vectorizer.transform([jieba_cut(new_question)])  # 轉換為 TF-IDF 特徵
predicted_category = model.predict(new_X)  # 預測分類

print(f"問題分類: {predicted_category[0]}")
