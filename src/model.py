import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import config
from collections import defaultdict


class KoreanChatbotModel:

    def __init__(self, model_path=None):
        self.device = config.DEVICE

        self.history = defaultdict(list)
        self.max_history = 5

        if model_path is None:
            default_path = config.MODELS_DIR / "korean_bot_final"
            if default_path.exists():
                self.model_name_or_path = str(default_path)
            else:
                print("Внимание: Обученная модель не найдена. Загружаю базовую.")
                self.model_name_or_path = config.MODEL_NAME
        else:
            self.model_name_or_path = str(model_path)

        print(f"Загрузка модели из: {self.model_name_or_path}")
        self.load_model(self.model_name_or_path)

    def load_model(self, path):
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(path)
            self.model = AutoModelForSeq2SeqLM.from_pretrained(path)

            if self.tokenizer.pad_token is None:
                if self.tokenizer.eos_token is not None:
                    self.tokenizer.pad_token = self.tokenizer.eos_token
                else:
                    self.tokenizer.add_special_tokens({'pad_token': '[PAD]'})
                    self.model.resize_token_embeddings(len(self.tokenizer))

            self.model.to(self.device)
            self.model.eval()
        except Exception as e:
            print(f"Ошибка загрузки модели: {e}")
            raise e

    def generate_response(self, text: str, user_id: int = 0) -> str:
        if not text.strip():
            return ""

        try:
            if not hasattr(self, 'history'):
                self.history = defaultdict(list)
                self.max_history = 3

            clean_text = text.strip()
            history_items = self.history[user_id][-self.max_history * 2:]
            context_str = " ".join(history_items)

            full_input = f"대화: {context_str} 답변: {clean_text}"

            inputs = self.tokenizer(
                full_input,
                max_length=256,
                truncation=True,
                return_tensors="pt"
            ).to(self.device)


            with torch.no_grad():
                outputs = self.model.generate(
                    inputs["input_ids"],
                    attention_mask=inputs.get("attention_mask"),
                    max_new_tokens=50,


                    num_beams=5,
                    do_sample=False,


                    temperature=1.0,

                    early_stopping=True,
                    no_repeat_ngram_size=2,

                    pad_token_id=self.tokenizer.pad_token_id,
                    eos_token_id=self.tokenizer.eos_token_id
                )

            response = self.tokenizer.decode(outputs[0], skip_special_tokens=True).strip()

            if not response or len(response) < 2:
                response = "네? 다시 말씀해 주시겠어요?"

            response = response.replace("봇:", "").replace("Bot:", "").strip()


            self.history[user_id].append(clean_text)
            self.history[user_id].append(response)

            if len(self.history[user_id]) > self.max_history * 2:
                self.history[user_id] = self.history[user_id][-self.max_history * 2:]

            return response

        except Exception as e:
            print(f"Критическая ошибка генерации для user {user_id}: {e}")
            return "지금은 대답할 수 없어요. 기술적인 문제가 발생했습니다. (Произошла техническая ошибка)"