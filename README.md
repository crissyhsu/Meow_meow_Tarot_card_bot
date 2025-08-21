# 針對感情問題的喵喵塔羅解牌機器人

## 相關依賴
請先安裝`requirements.txt`中的依賴：
```
pip install -r requirements.txt
```

## 簡介
本專案主要分為**問題分類系統`NLP_2.py`**和**塔羅占卜機器人`Tarot_bot.py`**、**Llama3.1微調語言模型**三個部分:
- **問題分類系統`NLP_2.py`**:用貝氏分類器模型與問題資料集`problem.json`進行分類器訓練，之後便會根據訓練結果產生`tfidf_vectorizer.pkl`與`naive_bayes_model.pkl`兩個檔案
- **塔羅占卜機器人`Tarot_bot.py`**：結合問題分類與生成式AI，提供個性化塔羅牌解讀(語言模型是經過微調的Llama3.1)
- **Llama3.1微調語言模型**：
    - 微調的資料集來源是：[喵喵解牌dataset](https://huggingface.co/datasets/xcr1005/tarot_reading)
    - 調好的模型：[喵喵解牌model](https://huggingface.co/xcr1005/tarot_try)

## 檔案結構
├── README.md
├── requirements.txt
├── NLP_2.py                # 問題分類訓練腳本
├── Tarot_bot.py            # 塔羅機器人主程式
├── problem.json            # 問題分類訓練資料
├── cn_stopwords.txt        # 中文停用詞表
├── tarot_card_gpt.json     # 塔羅牌資料庫
├── naive_bayes_model.pkl   # 訓練好的分類模型
└── tfidf_vectorizer.pkl    # TF-IDF向量化器

## 執行方式
在終端機輸入：
```
python Tarot_bot.py
```
之後在`http://127.0.0.1:7860`啟動 Gradio 網頁介面，便可執行

## 大概效果展示
詢問頁面：
![alt text](image.png)

解牌結果顯示(因為是本地運作 通常要等一下)：
![alt text](image-1.png)