config = {
    # Модель
    'embed_dim': 384,
    'num_heads': 6,
    'num_layers': 6,
    'block_size': 512,
    'dropout': 0.1,

    # Обучение
    'batch_size': 12,
    'learning_rate': 3e-4,
    'weight_decay': 0.1,
    'grad_clip': 1.0,
    'max_iters': 20000,
    'eval_interval': 200,
    'eval_iters': 10,          # было 200, теперь 10 для быстрой проверки
    'save_interval': 500,
    'log_interval': 50,
    'gradient_accumulation_steps': 4,
    
    # Система
    'device': 'cuda',
    'dtype': 'bfloat16',
    'out_dir': 'out/',
    'resume': False,

    # Данные
    'train_data_path': 'data/train.txt',
    'val_data_path': 'data/val.txt',

    # W&B (опционально)
    'use_wandb': False,
}