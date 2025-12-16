import sys
from pathlib import Path

# Добавляем корневую директорию в путь для импорта config
BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR))

import config
from src.utils import load_chatbot_data, split_dataset
from transformers import (
    AutoTokenizer, 
    AutoModelForSeq2SeqLM, 
    Seq2SeqTrainingArguments, 
    Seq2SeqTrainer,
    DataCollatorForSeq2Seq
)
from torch.utils.data import Dataset
import torch

class KoreanChatDataset(Dataset):
    def __init__(self, data, tokenizer, max_length=128):
        self.data = data
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        input_text = str(row['question']).strip()
        target_text = str(row['answer']).strip()
        
        # Проверка на пустые строки
        if not input_text or not target_text:
            # Возвращаем пустую последовательность, если данные некорректны
            input_text = "안녕하세요"
            target_text = "안녕하세요"

        # Токенизация входного текста
        input_encodings = self.tokenizer(
            input_text,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        )

        # Токенизация целевого текста (для labels)
        target_encodings = self.tokenizer(
            target_text,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        )

        # Получаем labels из input_ids
        labels = target_encodings["input_ids"].clone()
        
        # КРИТИЧНО: Заменяем padding token id на -100 для игнорирования в loss
        # Проверяем, что pad_token_id установлен
        pad_token_id = self.tokenizer.pad_token_id
        if pad_token_id is None:
            # Если pad_token не установлен, используем eos_token или unk_token
            pad_token_id = self.tokenizer.eos_token_id if self.tokenizer.eos_token_id is not None else 0
            self.tokenizer.pad_token_id = pad_token_id
        
        labels[labels == pad_token_id] = -100

        return {
            "input_ids": input_encodings["input_ids"].squeeze(0),
            "attention_mask": input_encodings["attention_mask"].squeeze(0),
            "labels": labels.squeeze(0)
        }

def train():
    print("=" * 50)
    print("НАЧАЛО ОБУЧЕНИЯ МОДЕЛИ")
    print("=" * 50)
    
    # 1. Данные
    print(f"\n1. Загрузка данных из {config.CSV_DATA_PATH}...")
    df = load_chatbot_data(str(config.CSV_DATA_PATH))
    
    if df.empty:
        print("ОШИБКА: Датасет пуст или не найден!")
        return
    
    # Ограничиваем размер для быстрого обучения (можно увеличить)
    max_samples = 10000  # Увеличил для лучшего качества
    if len(df) > max_samples:
        print(f"Ограничиваем датасет до {max_samples} примеров для ускорения обучения")
        df = df.sample(n=max_samples, random_state=42).reset_index(drop=True)
    
    print(f"Загружено {len(df)} примеров")
    
    # Проверка данных
    print("\nПроверка данных...")
    print(f"Пример вопроса: {df.iloc[0]['question'][:50]}...")
    print(f"Пример ответа: {df.iloc[0]['answer'][:50]}...")
    
    train_df, val_df = split_dataset(df, test_size=0.1)
    print(f"Обучающая выборка: {len(train_df)} примеров")
    print(f"Валидационная выборка: {len(val_df)} примеров")

    # 2. Модель и Токенизатор
    print(f"\n2. Загрузка модели {config.MODEL_NAME}...")
    print(f"Устройство: {config.DEVICE}")
    
    tokenizer = AutoTokenizer.from_pretrained(config.MODEL_NAME)
    
    # КРИТИЧНО: Устанавливаем pad_token если его нет
    if tokenizer.pad_token is None:
        if tokenizer.eos_token is not None:
            tokenizer.pad_token = tokenizer.eos_token
            print("Установлен pad_token = eos_token")
        else:
            tokenizer.add_special_tokens({'pad_token': '[PAD]'})
            print("Добавлен новый pad_token [PAD]")
    
    model = AutoModelForSeq2SeqLM.from_pretrained(config.MODEL_NAME)
    
    # Если добавили новый pad_token, нужно изменить размер embeddings
    if tokenizer.pad_token_id != model.config.pad_token_id:
        model.resize_token_embeddings(len(tokenizer))
        model.config.pad_token_id = tokenizer.pad_token_id
    
    model.to(config.DEVICE)
    
    # Убеждаемся, что модель в режиме обучения
    model.train()
    print(f"Модель загружена. Параметров: {sum(p.numel() for p in model.parameters()):,}")
    print(f"Обучаемых параметров: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")

    # 3. Датасеты
    print("\n3. Создание датасетов...")
    train_dataset = KoreanChatDataset(train_df, tokenizer, config.TRAIN_CONFIG['max_length'])
    val_dataset = KoreanChatDataset(val_df, tokenizer, config.TRAIN_CONFIG['max_length'])
    
    # Проверка первого примера
    sample = train_dataset[0]
    print(f"Проверка примера:")
    print(f"  input_ids shape: {sample['input_ids'].shape}")
    print(f"  labels shape: {sample['labels'].shape}")
    print(f"  labels не -100: {(sample['labels'] != -100).sum().item()} токенов")
    print(f"  pad_token_id: {tokenizer.pad_token_id}")

    # 4. Аргументы обучения
    print("\n4. Настройка параметров обучения...")
    args = Seq2SeqTrainingArguments(
        output_dir=str(config.MODELS_DIR / "checkpoints"),
        eval_strategy="steps",  # Исправлено: eval_strategy вместо evaluation_strategy
        eval_steps=200,  # Оценка каждые 200 шагов
        save_strategy="steps",
        save_steps=200,
        learning_rate=config.TRAIN_CONFIG['learning_rate'],
        per_device_train_batch_size=config.TRAIN_CONFIG['batch_size'],
        per_device_eval_batch_size=config.TRAIN_CONFIG['batch_size'],
        weight_decay=0.01,
        save_total_limit=2,
        num_train_epochs=config.TRAIN_CONFIG['epochs'],
        predict_with_generate=True,
        fp16=config.USE_FP16,
        logging_dir=str(config.LOGS_DIR),
        logging_steps=config.TRAIN_CONFIG['logging_steps'],
        dataloader_num_workers=0,  # 0 для Windows
        load_best_model_at_end=True,
        metric_for_best_model="loss",
        greater_is_better=False,
        report_to="none",  # Отключаем wandb/tensorboard если не нужны
        remove_unused_columns=False,
        warmup_steps=config.TRAIN_CONFIG.get('warmup_steps', 100),
        gradient_accumulation_steps=1,  # Для экономии памяти
        max_grad_norm=1.0,  # Предотвращает взрыв градиентов
    )

    data_collator = DataCollatorForSeq2Seq(
        tokenizer=tokenizer, 
        model=model,
        padding=True
    )

    trainer = Seq2SeqTrainer(
        model=model,
        args=args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        data_collator=data_collator,
        tokenizer=tokenizer,
    )

    print("\n5. Начало обучения...")
    print("=" * 50)
    
    try:
        trainer.train()
        print("\n" + "=" * 50)
        print("ОБУЧЕНИЕ ЗАВЕРШЕНО УСПЕШНО!")
        print("=" * 50)
    except Exception as e:
        print(f"\nОШИБКА при обучении: {e}")
        import traceback
        traceback.print_exc()
        return

    # 5. Сохранение
    print("\n6. Сохранение модели...")
    final_path = config.MODELS_DIR / "korean_bot_final"
    model.save_pretrained(final_path)
    tokenizer.save_pretrained(final_path)
    print(f"Модель сохранена в: {final_path}")

if __name__ == "__main__":
    train()
