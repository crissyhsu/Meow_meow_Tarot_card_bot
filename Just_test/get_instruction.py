import json
import random

problems = ["未來感情趨勢","分手後復合","爭吵與摩擦","感情冷淡","關係不明","單戀與告白"]

with open("../tarot_card_gpt.json", "r", encoding="utf-8") as file:
    tarot_cards = json.load(file)

training_data = []

for problem in problems:
    user_question = problem
    for i in range(0,9):
        # 隨機抽取 3 張塔羅牌
        selected_cards = random.sample(tarot_cards, 3)
        tarot_info = "\n".join([
            f"{i+1}. {card['name']}（{random.choice(['正位', '逆位'])}）：{card['upright_meaning'] if random.choice(['正位', '逆位']) == '正位' else card['reversed_meaning']}"
            for i, card in enumerate(selected_cards)
        ])
        training_data.append({
            "input": f"問題類別：「{problem}」\n抽到的塔羅牌：\n{tarot_info}\n請提供完整的塔羅解讀。",
            "output": f""
        })

with open("tarot_card\\tarot_dataset.json", "w", encoding="utf-8") as file:
    json.dump(training_data, file, ensure_ascii=False, indent=4)

print("訓練資料已成功生成並儲存到 `tarot_dataset.json`！")