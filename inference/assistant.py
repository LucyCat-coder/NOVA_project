import torch
from rag.retriever import Retriever
from model.model import GPT
import tiktoken

class Assistant:
    def __init__(self, model_path: str, config):
        self.device = config['device']
        self.tokenizer = tiktoken.get_encoding("gpt2")

        self.model = GPT(
            vocab_size=self.tokenizer.n_vocab,
            embed_dim=config['embed_dim'],
            num_heads=config['num_heads'],
            num_layers=config['num_layers'],
            block_size=config['block_size']
        )
        checkpoint = torch.load(model_path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.to(self.device)
        self.model.eval()

        self.retriever = Retriever()

    def index_documents(self, docs: list[str]):
        self.retriever.index_documents(docs)

    def generate(self, query: str, max_new_tokens=100, temperature=0.7,
                 top_k_docs=3, top_k_sampling=None):
        # 1. RAG — ищем релевантные документы
        context_chunks = self.retriever.retrieve(query, top_k=top_k_docs)
        context = "\n".join(context_chunks)

        # 2. Формируем промпт
        prompt = f"Контекст:\n{context}\n\nПользователь: {query}\nНова:"
        input_ids = self.tokenizer.encode(prompt, disallowed_special=())

        # 3. Обрезаем если не влезает в block_size
        max_prompt_len = self.model.block_size - max_new_tokens
        if len(input_ids) > max_prompt_len:
            input_ids = input_ids[-max_prompt_len:]

        generated = torch.tensor([input_ids], dtype=torch.long, device=self.device)

        # 4. Авторегрессивная генерация
        for _ in range(max_new_tokens):
            with torch.no_grad():
                logits, _ = self.model(generated)

            next_logits = logits[:, -1, :] / temperature

            # Опциональный top-k сэмплинг
            if top_k_sampling is not None:
                values, _ = torch.topk(next_logits, top_k_sampling)
                next_logits[next_logits < values[:, [-1]]] = float('-inf')

            probs = torch.softmax(next_logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            generated = torch.cat([generated, next_token], dim=1)

            if next_token.item() == 50256:  # <|endoftext|>
                break

        # 5. Декодируем только ответ модели
        output_text = self.tokenizer.decode(generated[0].tolist())
        if "Нова:" in output_text:
            return output_text.split("Нова:")[-1].strip()
        return output_text