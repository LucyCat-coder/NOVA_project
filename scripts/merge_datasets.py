import random

# Читаем оба датасета
with open('data/train.txt', 'r', encoding='utf-8') as f:
    original = f.read().strip()

with open('data/alpaca/train.txt', 'r', encoding='utf-8') as f:
    alpaca = f.read().strip()

# Разбиваем на отдельные диалоги
original_dialogs = original.split('\n\n')
alpaca_dialogs = alpaca.split('\n\n')

print(f"Оригинал: {len(original_dialogs)} диалогов")
print(f"Alpaca: {len(alpaca_dialogs)} диалогов")

# Перемешиваем и объединяем
all_dialogs = original_dialogs + alpaca_dialogs
random.shuffle(all_dialogs)

# Сохраняем 90% в train, 10% в val
split = int(len(all_dialogs) * 0.9)
train_dialogs = all_dialogs[:split]
val_dialogs = all_dialogs[split:]

with open('data/train.txt', 'w', encoding='utf-8') as f:
    f.write('\n\n'.join(train_dialogs))

with open('data/val.txt', 'w', encoding='utf-8') as f:
    f.write('\n\n'.join(val_dialogs))

print(f"Итого train: {len(train_dialogs)} диалогов")
print(f"Итого val: {len(val_dialogs)} диалогов")