from transformers import pipeline, AutoModelForCausalLM, AutoTokenizer

model_name = "xcr1005/tarot_try"

# 載入 tokenizer & 模型
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)

# 創建 `pipeline`
generator = pipeline("text-generation", model=model, tokenizer=tokenizer)

# **手動構造 prompt**
instruction = "請根據抽到的塔羅牌，提供完整的感情冷淡解讀，並加入喵喵叫的風格。"
input_text = "問題類別：「感情冷淡」\n抽到的塔羅牌：\n1. 寶劍五（逆位）：暫時的勝利，破壞，辭職或免職，墜落，損人利己，罵人，短視的戀人\n2. 寶劍皇后（逆位）：如意時孤獨，失意時有補償，潔癖，過火，失落感，離別，分手\n3. 錢幣五（逆位）：居無定所，混亂無序，因貧困而心大亂，想法荒唐，缺少情感支持\n請提供完整的塔羅解讀。"
prompt = f"### Instruction:\n{instruction}\n### Input:\n{input_text}\n### Response:\n"

# 產生回答
output = generator(prompt, max_length=1200, pad_token_id=tokenizer.eos_token_id)
response_text = output[0]["generated_text"].replace(prompt, "").strip()
print(response_text)
