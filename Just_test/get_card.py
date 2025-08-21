import json
import random

# 讀取塔羅牌資料
with open("../tarot_card_gpt.json", "r", encoding="utf-8") as file:
    tarot_cards = json.load(file)

def draw_three_tarot_cards():
    """隨機抽三張不同的塔羅牌，並決定是否為正位或逆位"""
    selected_cards = random.sample(tarot_cards, 3)  # 隨機選擇 3 張不同的牌
    result = []

    for card in selected_cards:
        position = random.choice(["正位", "逆位"])  # 隨機決定方向
        meaning = card["upright_meaning"] if position == "正位" else card["reversed_meaning"]

        result.append({
            "name": card["name"],
            "position": position,
            "meaning": meaning
        })

    return result

# 測試抽牌
cards = draw_three_tarot_cards()
for i, card in enumerate(cards):
    print(f"第 {i+1} 張：{card['name']}（{card['position']}）")
    print(f"牌義：{card['meaning']}\n")
