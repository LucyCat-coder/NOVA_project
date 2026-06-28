import os
os.environ['HF_HUB_DISABLE_SYMLINKS_WARNING'] = '1'

from inference.assistant import Assistant
from config.train_config import config

assistant = Assistant('out/checkpoint.pt', config)

assistant.index_documents([
    "Градиентный спуск — метод оптимизации, который итеративно обновляет параметры модели в направлении антиградиента функции потерь.",
    "Нейронная сеть обучается путём минимизации функции потерь с помощью обратного распространения ошибки.",
    "PyTorch — популярный фреймворк для глубокого обучения с поддержкой динамических вычислительных графов.",
])

print(assistant.generate("Что такое градиентный спуск?", max_new_tokens=200, temperature=0.3))