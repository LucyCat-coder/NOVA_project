config = {
    # Модель
    'embed_dim': 384,
    'num_heads': 6,
    'num_layers': 6,
    'block_size': 256,
    'dropout': 0.1,

    # Обучение
    'batch_size': 12,
    'learning_rate': 3e-4,
    'weight_decay': 0.1,
    'grad_clip': 1.0,
    'max_iters': 100000,
    'eval_interval': 500,
    'eval_iters': 10,          # было 200, теперь 10 для быстрой проверки
    'save_interval': 1000,
    'log_interval': 50,

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