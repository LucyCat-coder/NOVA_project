import json
import os

os.makedirs("data/knowledge", exist_ok=True)

with open("nova_knowledge.jsonl", "r", encoding="utf-8") as f_in, \
     open("data/knowledge/train.txt", "w", encoding="utf-8") as f_out:
    for line in f_in:
        if not line.strip():
            continue
        data = json.loads(line)
        messages = data.get("messages", [])
        if len(messages) >= 2:
            user = messages[0].get("content", "")
            assistant = messages[1].get("content", "")
            f_out.write(f"Пользователь: {user}\nНова: {assistant}\n\n")

print("Готово! Файл data/knowledge/train.txt создан.")