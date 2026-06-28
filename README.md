markdown
# NOVA — Персональный ИИ-ассистент

**NOVA** — это проект по созданию собственной языковой модели (LLM) с нуля, вдохновлённой Джарвисом из «Железного человека».  
Проект выполнен в рамках дипломной работы по специальности «Разработчик искусственного интеллекта».

---

## Особенности

- **Кастомная архитектура** — GPT-подобная модель на PyTorch (6 слоёв, 6 голов внимания, 384 эмбеддинга).
- **Собственный датасет** — 15 000+ технических диалогов, сгенерированных с помощью `mistral-nemo` через Ollama.
- **Гибкий пайплайн генерации** — скрипты для создания синтетических диалогов по темам (Python, ML, Git, Linux, Docker и др.).
- **Обучение на RTX 4070 Ti SUPER** — оптимизировано под 16 ГБ видеопамяти (bfloat16, gradient accumulation, `torch.compile`).
- **Поддержка чекпоинтов** — возможность прерывать и возобновлять обучение в любой момент.
- **Планируемый RAG** — в дальнейшем будет добавлен векторный поиск (FAISS + эмбеддинги) для расширения знаний.

---

## Структура проекта
nova_project/
├── config/
│ └── train_config.py # гиперпараметры модели и обучения
├── data/
│ ├── synthetic/ # (опционально) диалоги личности
│ ├── knowledge/ # технические диалоги (основной датасет)
│ ├── train.txt # тренировочные данные
│ └── val.txt # валидационные данные (10% от train)
├── model/
│ ├── model.py # архитектура GPT (Transformer Decoder)
│ └── attention.py # механизм самовнимания с маской (causal)
├── train/
│ ├── dataset.py # загрузка и токенизация данных (tiktoken)
│ └── trainer.py # цикл обучения с чекпоинтами и валидацией
├── scripts/
│ ├── build_knowledge_dataset.py # генерация технических диалогов (JSONL)
│ └── convert_jsonl_to_txt.py # конвертация JSONL → train.txt
├── main.py # точка входа (обучение / возобновление)
├── .gitignore
└── README.md

text

---

## Быстрый старт

### 1. Клонируйте репозиторий (или создайте проект локально)

```bash
git clone https://github.com/your-username/nova_project.git
cd nova_project
2. Создайте и активируйте виртуальное окружение
Windows:

bash
python -m venv .venv
.venv\Scripts\activate
Linux/macOS:

bash
python3 -m venv .venv
source .venv/bin/activate
3. Установите зависимости
bash
pip install --upgrade pip
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124
pip install transformers datasets tiktoken wandb tqdm sentence-transformers faiss-cpu ollama
Если у вас другая версия CUDA, выберите соответствующую команду на pytorch.org.

4. Подготовьте данные
Убедитесь, что в папке data/ лежат файлы train.txt и val.txt.
Если их нет, сгенерируйте датасет знаний (см. раздел «Генерация датасетов»).

5. Запустите обучение
bash
python main.py
Для возобновления с последнего чекпоинта:

bash
python main.py --resume
Генерация датасетов
5.1. Технические диалоги (корпус знаний)
Скрипт scripts/build_knowledge_dataset.py генерирует 15 000 уникальных технических вопросов и ответов по 20+ темам (Python, ML, Git, Linux, Docker и др.) с использованием ollama и модели mistral-nemo.

Запуск:

bash
python scripts/build_knowledge_dataset.py
Результат: nova_knowledge.jsonl (каждая строка — JSON-диалог в формате {"messages": [...]}).

5.2. Конвертация JSONL → train.txt
Скрипт scripts/convert_jsonl_to_txt.py преобразует JSONL в текстовый формат, совместимый с загрузчиком:

text
Пользователь: вопрос
Нова: ответ
Запуск:

bash
python scripts/convert_jsonl_to_txt.py
Результат: data/knowledge/train.txt.

5.3. Разделение на train / val
В PowerShell (из корня проекта) выполните:

powershell
$lines = Get-Content data\train.txt
$total = $lines.Count
$valCount = [int]($total * 0.1)
$valLines = $lines[0..($valCount-1)]
$trainLines = $lines[$valCount..($total-1)]
$trainLines | Set-Content data\train.txt
$valLines | Set-Content data\val.txt
Теперь у вас есть тренировочные (90%) и валидационные (10%) данные.

Конфигурация
Все ключевые параметры вынесены в config/train_config.py:

Параметр	Значение	Описание
embed_dim	384	Размерность эмбеддингов
num_heads	6	Количество голов внимания
num_layers	6	Количество слоёв Transformer
block_size	256	Длина контекста (в токенах)
dropout	0.1	Вероятность отключения нейронов
batch_size	12	Размер батча
gradient_accumulation_steps	4	Эффективный батч = 48
learning_rate	3e-4	Начальная скорость обучения
max_iters	100000	Количество итераций
weight_decay	0.1	L2-регуляризация
grad_clip	1.0	Обрезка градиентов
device	cuda	Использовать GPU
dtype	bfloat16	Тип данных для ускорения и экономии памяти
compile	True	Включить torch.compile
out_dir	out/	Папка для чекпоинтов
resume	False	Флаг для возобновления (устанавливается через --resume)
Чекпоинты и возобновление
Сохранение: каждые save_interval итераций (по умолчанию 1000) и при улучшении валидационной ошибки.

Файл: out/checkpoint.pt (содержит веса модели, состояние оптимизатора, планировщика, номер шага и лучшую ошибку).

Возобновление: python main.py --resume

Это позволяет безопасно прерывать обучение и продолжать позже, не теряя прогресс.

Планы по RAG
После завершения обучения планируется интеграция RAG (Retrieval-Augmented Generation):

Векторная база: FAISS

Эмбеддинги: sentence-transformers/all-MiniLM-L6-v2

Источник знаний: техническая документация, статьи, заметки — для поиска релевантного контекста и подстановки в промпт перед генерацией ответа.

Это позволит NOVA отвечать на вопросы, выходящие за рамки обучающих данных.

Возможные ошибки и их решение
Ошибка	Вероятная причина	Решение
ModuleNotFoundError: No module named 'train.dataset'	Отсутствует __init__.py в папке train/	Создайте пустой train/__init__.py и model/__init__.py
ValueError: Encountered text corresponding to disallowed special token '<|endoftext|>'	Токенизатор не разрешает специальные токены	В dataset.py замените encode(text) на encode(text, disallowed_special=())
CUDA out of memory	Слишком большой батч или модель	Уменьшите batch_size или block_size; проверьте, что dtype='bfloat16'
ollama не найден	Библиотека не установлена или Ollama не запущен	Установите pip install ollama; убедитесь, что Ollama запущен (ollama serve в фоне)
Автор
Проект разработан в рамках дипломной работы по специальности «Разработчик искусственного интеллекта».
Вдохновлён идеей создания персонального ассистента, подобного Джарвису.

Лицензия
MIT — используйте, модифицируйте и улучшайте свободно.

Благодарности
Андрей Карпати (Andrej Karpathy) за nanoGPT и вдохновляющие лекции.

Команда HuggingFace за инструменты для работы с датасетами и моделями.

Ollama за удобный локальный запуск LLM для генерации данных.

Всем, кто тестировал и давал обратную связь.

Удачи в обучении NOVA! 🚀

text