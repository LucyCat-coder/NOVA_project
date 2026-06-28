from datasets import load_dataset
import os

ds = load_dataset('saillab/alpaca-russian-cleaned')

def format_example(example):
    instruction = example['instruction'].strip()
    inp = example['input'].strip()
    output = example['output'].strip()
    
    # Если есть input — добавляем его к вопросу
    if inp:
        user_text = f"{instruction}\n{inp}"
    else:
        user_text = instruction
    
    return f"Пользователь: {user_text}\nНова: {output}"

os.makedirs('data/alpaca', exist_ok=True)

# Конвертируем train
train_examples = [format_example(ex) for ex in ds['train']]
with open('data/alpaca/train.txt', 'w', encoding='utf-8') as f:
    f.write('\n\n'.join(train_examples))

# Конвертируем test → val
val_examples = [format_example(ex) for ex in ds['test']]
with open('data/alpaca/val.txt', 'w', encoding='utf-8') as f:
    f.write('\n\n'.join(val_examples))

print(f"Train: {len(train_examples)} примеров")
print(f"Val: {len(val_examples)} примеров")
print("\nПример:")
print(train_examples[0])