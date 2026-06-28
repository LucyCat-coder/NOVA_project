import torch
import os
from torch.utils.data import DataLoader
from tqdm import tqdm

class Trainer:
    def __init__(self, model, config, train_dataset, val_dataset):
        self.model = model
        self.config = config
        self.device = config['device']

        self.train_loader = DataLoader(train_dataset, batch_size=config['batch_size'], shuffle=True, num_workers=0)
        self.val_loader = DataLoader(val_dataset, batch_size=config['batch_size'], shuffle=False, num_workers=0)

        self.optimizer = torch.optim.AdamW(model.parameters(), lr=config['learning_rate'], weight_decay=config['weight_decay'])
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=config['max_iters'])

        self.iter_num = 0
        self.best_val_loss = float('inf')
        self.out_dir = config['out_dir']
        os.makedirs(self.out_dir, exist_ok=True)

        # Итератор для корректного прохода по датасету (фикс #4)
        self._train_iter = iter(self.train_loader)

        if config.get('resume', False):
            self.load_checkpoint()

    def save_checkpoint(self):
        ckpt = {
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'iter_num': self.iter_num,
            'best_val_loss': self.best_val_loss,
        }
        torch.save(ckpt, os.path.join(self.out_dir, 'checkpoint.pt'))
        print(f"Чекпоинт сохранён на итерации {self.iter_num}")

    def load_checkpoint(self):
        ckpt_path = os.path.join(self.out_dir, 'checkpoint.pt')
        if os.path.exists(ckpt_path):
            ckpt = torch.load(ckpt_path, map_location=self.device)
            self.model.load_state_dict(ckpt['model_state_dict'])
            self.optimizer.load_state_dict(ckpt['optimizer_state_dict'])
            self.scheduler.load_state_dict(ckpt['scheduler_state_dict'])
            self.iter_num = ckpt['iter_num']
            self.best_val_loss = ckpt['best_val_loss']
            print(f"Загружен чекпоинт с итерации {self.iter_num}")
        else:
            print("Чекпоинт не найден, начинаем с нуля")

    def _get_batch(self):
        """Берёт следующий батч, перезапускает итератор при исчерпании (фикс #4)."""
        try:
            return next(self._train_iter)
        except StopIteration:
            self._train_iter = iter(self.train_loader)
            return next(self._train_iter)

    def train_step(self, batch):
        """Один forward+backward без шага оптимизатора (фикс #1)."""
        x, y = batch
        x, y = x.to(self.device), y.to(self.device)
        _, loss = self.model(x, targets=y)
        loss = loss / self.config.get('gradient_accumulation_steps', 1)
        loss.backward()
        return loss.item()

    @torch.no_grad()
    def eval_step(self):
        """Валидация без лишних принтов (фикс #3)."""
        self.model.eval()
        total_loss = 0.0
        eval_iters = self.config.get('eval_iters', 200)

        for i, batch in enumerate(self.val_loader):
            if i >= eval_iters:
                break
            x, y = batch
            x, y = x.to(self.device), y.to(self.device)
            _, loss = self.model(x, targets=y)
            total_loss += loss.item()

        avg_loss = total_loss / min(eval_iters, i + 1)
        self.model.train()
        return avg_loss

    def train(self):
        total_iters = self.config['max_iters']
        eval_interval = self.config.get('eval_interval', 500)
        save_interval = self.config.get('save_interval', 1000)
        log_interval  = self.config.get('log_interval', 50)
        accum_steps   = self.config.get('gradient_accumulation_steps', 1)

        val_loss = self.eval_step()
        print(f"Начальная валидационная ошибка: {val_loss:.4f}")

        pbar = tqdm(range(self.iter_num, total_iters), initial=self.iter_num, total=total_iters)
        accum_loss = 0.0

        for step in pbar:
            # --- накапливаем градиенты (фикс #1) ---
            for micro_step in range(accum_steps):
                batch = self._get_batch()
                accum_loss += self.train_step(batch)

            # --- один реальный шаг оптимизатора ---
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config['grad_clip'])
            self.optimizer.step()
            self.scheduler.step()
            self.optimizer.zero_grad()

            self.iter_num += 1
            avg_loss = accum_loss / accum_steps
            accum_loss = 0.0

            if self.iter_num % log_interval == 0:
                pbar.set_description(f"loss {avg_loss:.4f}")

            if self.iter_num % eval_interval == 0:
                val_loss = self.eval_step()
                print(f"Шаг {self.iter_num}: val_loss = {val_loss:.4f}")
                if val_loss < self.best_val_loss:
                    self.best_val_loss = val_loss
                    self.save_checkpoint()

            if self.iter_num % save_interval == 0:
                self.save_checkpoint()

        self.save_checkpoint()
        print("Обучение завершено!")