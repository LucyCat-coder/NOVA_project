# NOVA — Персональный ИИ-ассистент

**NOVA** — это проект по созданию собственной языковой модели (LLM) с нуля, вдохновлённой Джарвисом из «Железного человека».  
Проект выполнен в рамках дипломной работы по специальности «Разработчик искусственного интеллекта».

---

## Особенности

- Кастомная архитектура — GPT-подобная модель на PyTorch (6 слоёв, 6 голов внимания, 384 эмбеддинга).
- Собственный датасет — 15 000+ технических диалогов, сгенерированных с помощью mistral-nemo через Ollama.
- Гибкий пайплайн генерации — скрипты для создания синтетических диалогов по темам (Python, ML, Git, Linux, Docker и др.).
- Обучение на RTX 4070 Ti SUPER — оптимизировано под 16 ГБ видеопамяти (bfloat16, gradient accumulation, torch.compile).
- Поддержка чекпоинтов — возможность прерывать и возобновлять обучение в любой момент.
- Планируемый RAG — в дальнейшем будет добавлен векторный поиск (FAISS + эмбеддинги).

---

## Структура проекта

```text
nova_project/
├── config/
│   └── train_config.py
├── data/
│   ├── synthetic/
│   ├── knowledge/
│   ├── train.txt
│   └── val.txt
├── model/
│   ├── model.py
│   └── attention.py
├── train/
│   ├── dataset.py
│   └── trainer.py
├── scripts/
│   ├── build_knowledge_dataset.py
│   └── convert_jsonl_to_txt.py
├── main.py
├── .gitignore
└── README.md
```

---

## Быстрый старт

git clone https://github.com/your-username/nova_project.git
cd nova_project

python -m venv .venv
.venv\Scripts\activate  # Windows
# source .venv/bin/activate  # Linux/macOS

pip install --upgrade pip
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124
pip install transformers datasets tiktoken wandb tqdm sentence-transformers faiss-cpu ollama

python main.py

---

## Генерация датасетов

python scripts/build_knowledge_dataset.py
python scripts/convert_jsonl_to_txt.py

---

## Конфигурация

Все параметры в config/train_config.py

---

## Чекпоинты

out/checkpoint.pt

---

## RAG (план)

FAISS + sentence-transformers/all-MiniLM-L6-v2

---

## Автор

Дипломная работа — разработчик ИИ

Вдохновлён идеей создания персонального ассистента, подобного Джарвису.

---

## Лицензия

MIT
