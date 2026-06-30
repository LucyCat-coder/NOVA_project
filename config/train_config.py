config = {
    # Модель
    'embed_dim': 768,
    'num_heads': 12,
    'num_layers': 12,
    'block_size': 512,
    'dropout': 0.1,

    # Обучение
    'batch_size': 8,
    'learning_rate': 5e-5,
    'weight_decay': 0.1,
    'grad_clip': 1.0,
    'max_iters': 60000,
    'eval_interval': 200,
    'eval_iters': 10,
    'save_interval': 500,
    'log_interval': 50,
    'gradient_accumulation_steps': 4,
    'reset_scheduler': True,

    # Система
    'device': 'cuda',
    'dtype': 'bfloat16',
    'out_dir': 'out/',
    'resume': True,

    # Данные
    'train_data_path': 'data/train.txt',
    'val_data_path': 'data/val.txt',

    # W&B (опционально)
    'use_wandb': False,
}