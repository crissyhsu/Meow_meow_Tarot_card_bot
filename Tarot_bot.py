import gradio as gr
import json
import random
import joblib
import jieba
from transformers import pipeline, AutoModelForCausalLM, AutoTokenizer

#載入在HuggingFace上已經自己微調好的模型
model_name = "xcr1005/tarot_try"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)
generator = pipeline("text-generation", model=model, tokenizer=tokenizer)


model = joblib.load("naive_bayes_model.pkl")
vectorizer = joblib.load("tfidf_vectorizer.pkl")

with open("tarot_card_gpt.json", "r", encoding="utf-8") as file:
    tarot_cards = json.load(file)


def draw_three_tarot_cards():
    #隨機抽三張不同的塔羅牌，並決定是否為正位或逆位
    selected_cards = random.sample(tarot_cards, 3)
    result = []

    for card in selected_cards:
        position = random.choice(["正位", "逆位"])  #隨機決定方向
        meaning = card["upright_meaning"] if position == "正位" else card["reversed_meaning"]

        result.append({
            "name": card["name"],
            "position": position,
            "meaning": meaning
        })

    return result

def check_type(str):
    new_question = str
    
    def jieba_cut(text):
        return " ".join(jieba.cut(text))
    new_X = vectorizer.transform([jieba_cut(new_question)])
    predicted_category = model.predict(new_X)
    return predicted_category[0]

def tarot_reading(question):
    if not question.strip():
        return "請輸入你的問題喵～"
    
    cards = draw_three_tarot_cards()
    type = check_type(question)
    
    instruction = f"請根據抽到的塔羅牌，提供完整的{type}解讀，並加入喵喵叫的風格。"
    input_text = f"問題類別：「{type}」\n抽到的塔羅牌：\n"
    
    result = "# 貓貓塔羅館🐱🐾\n\n"
    result += "喵喵~這邊可愛的塔羅貓幫你抽出了三張牌\n\n"
    result += "## 你的三張牌如下🐱：\n\n"
    
    for i, card in enumerate(cards):
        result += f"**{card['name']}（{card['position']}）**：{card['meaning']}\n\n"
        input_text += f"{card['name']}（{card['position']}）：{card['meaning']}\n"

    prompt = f"### Instruction:\n{instruction}\n### Input:\n{input_text}\n### Response:\n"

    # 產生回答
    result += "## 詳細解牌🐱：\n\n"
    result += "本喵下線思考億下下...\n\n"
    
    try:
        #用調好的模型產生答案
        output = generator(prompt, max_length=1200, pad_token_id=tokenizer.eos_token_id)
        interpretation = output[0]["generated_text"].replace(prompt, "").strip()
        result += interpretation
    except Exception as e:
        result += f"抱歉喵～模型出現問題：{str(e)}"
    
    return result

def create_gradio_interface():
    #自定義CSS樣式
    custom_css = """
    .gradio-container {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        font-family: 'Microsoft JhengHei', sans-serif;
    }
    .gr-button-primary {
        background: linear-gradient(45deg, #ff6b6b, #ffa500) !important;
        border: none !important;
        border-radius: 25px !important;
    }
    .gr-textbox textarea {
        border-radius: 10px !important;
        border: 2px solid #e0e0e0 !important;
    }
    #output-markdown {
        background: rgba(255, 255, 255, 0.9) !important;
        border-radius: 15px !important;
        padding: 20px !important;
        backdrop-filter: blur(10px) !important;
        box-shadow: 0 8px 32px rgba(0, 0, 0, 0.1) !important;
    }
    """
    
    with gr.Blocks(
        title="貓貓塔羅館🐱🔮", 
        theme=gr.themes.Soft(),
        css=custom_css
    ) as demo:
        
        # 標題區域
        gr.HTML("""
        <div style="text-align: center; padding: 30px; background: linear-gradient(45deg, #ff6b6b, #ffa500); 
                    border-radius: 15px; margin-bottom: 20px; color: white;">
            <h1 style="font-size: 2.5em; margin-bottom: 10px; text-shadow: 2px 2px 4px rgba(0,0,0,0.3);">
                貓貓塔羅館 🐱🔮
            </h1>
            <p style="font-size: 1.2em;">讓可愛的塔羅貓為你解讀命運的密碼～喵！</p>
        </div>
        """)
        
        with gr.Row():
            with gr.Column(scale=1):
                question_input = gr.Textbox(
                    label="🐾 請輸入你的問題",
                    placeholder="例如：我和心儀對象的愛情發展如何？",
                    lines=3,
                    max_lines=5
                )
                
                submit_btn = gr.Button(
                    "🔮 抽取塔羅牌", 
                    variant="primary",
                    size="lg"
                )
                
                gr.Examples(
                    examples=[
                        ["我和暗戀對象有機會在一起嗎？"],
                        ["他喜歡我嗎？"],
                        ["我應該和他復合嗎？"],
                        ["吵架了，我們之後會怎麼樣？"],
                        ["未來三個月會有桃花嗎？"]
                    ],
                    inputs=question_input,
                    label="💡 點擊試試這些問題"
                )
        
        with gr.Row():
            output = gr.Markdown(
                label="🐱 塔羅解讀結果",
                elem_id="output-markdown"
            )
        
        submit_btn.click(
            fn=tarot_reading,
            inputs=question_input,
            outputs=output,
            api_name="tarot"
        )
        
        question_input.submit(
            fn=tarot_reading,
            inputs=question_input,
            outputs=output
        )
        
        gr.HTML("""
        <div style="text-align: center; padding: 20px; margin-top: 30px; 
                    background: rgba(255,255,255,0.1); border-radius: 10px;">
            <p style="color: #666;">✨ 塔羅占卜僅供娛樂參考，請以實際行動為準 ✨</p>
        </div>
        """)
    
    return demo

# 啟動應用程式
if __name__ == "__main__":
    demo = create_gradio_interface()
    demo.launch(
        server_name="127.0.0.1",  # 本地跑(想公開就0.0.0.0)
        server_port=7860,       # 設定埠號
        share= False,           #生成公開連結
        debug=True              # 開啟除錯模式
    )