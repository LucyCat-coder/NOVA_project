import os
os.environ['HF_HUB_DISABLE_SYMLINKS_WARNING'] = '1'

from inference.assistant import Assistant
from config.train_config import config

assistant = Assistant('out/checkpoint.pt', config)

questions = [
    "Что такое градиентный спуск?",
    "Как работает Docker?",
    "Объясни что такое Git",
]

for q in questions:
    print(f"Вопрос: {q}")
    print(f"Ответ: {assistant.generate(q, max_new_tokens=350, temperature=0.3, top_k_docs=0)}")
    print("---")