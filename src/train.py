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
    DataCollatorForSeq2Seq,
)
from torch.utils.data import Dataset
import torch
import matplotlib.pyplot as plt


class KoreanChatDataset(Dataset):
    def __init__(self, data, tokenizer, max_length: int = 128):
        self.data = data
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        """
        Возвращает словарь с input_ids, attention_mask, labels.
        Trainer сам считает суммарный loss (здесь он соответствует
        языковому моделированию, т.е. «классификации» токенов).
        Отдельного bbox-лосса в этой модели нет.
        """
        row = self.data.iloc[idx]
        input_text = str(row["question"]).strip()
        target_text = str(row["answer"]).strip()

        if not input_text or not target_text:
            input_text = "안녕하세요"
            target_text = "안녕하세요"

        # Токенизация входа
        model_inputs = self.tokenizer(
            input_text,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
        )

        # Токенизация цели (labels)
        with self.tokenizer.as_target_tokenizer():
            labels = self.tokenizer(
                target_text,
                max_length=self.max_length,
                padding="max_length",
                truncation=True,
            )["input_ids"]

        pad_token_id = self.tokenizer.pad_token_id
        if pad_token_id is None:
            pad_token_id = self.tokenizer.eos_token_id or 0
            self.tokenizer.pad_token_id = pad_token_id

        # Маскируем паддинги как -100, чтобы не входили в loss
        labels = [tid if tid != pad_token_id else -100 for tid in labels]
        model_inputs["labels"] = labels
        return model_inputs


def _plot_losses_from_history(log_history, output_dir: Path):
    """
    Строит график средних train/val потерь по эпохам.

    Для нашей seq2seq модели:
      - потери «классификации» токенов и «total loss» совпадают;
      - отдельного bbox-лосса нет, поэтому на графике — именно суммарный loss.
    """
    train_loss_per_epoch = {}
    eval_loss_per_epoch = {}

    for record in log_history:
        epoch = record.get("epoch")
        if epoch is None:
            continue

        if "loss" in record:
            train_loss_per_epoch.setdefault(epoch, []).append(record["loss"])
        if "eval_loss" in record:
            eval_loss_per_epoch.setdefault(epoch, []).append(record["eval_loss"])

    train_epochs = sorted(train_loss_per_epoch.keys())
    eval_epochs = sorted(eval_loss_per_epoch.keys())

    mean_train = [sum(train_loss_per_epoch[e]) / len(train_loss_per_epoch[e]) for e in train_epochs] if train_epochs else []
    mean_eval = [sum(eval_loss_per_epoch[e]) / len(eval_loss_per_epoch[e]) for e in eval_epochs] if eval_epochs else []

    plots_dir = config.LOGS_DIR / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)
    plot_path = plots_dir / "loss_curves.png"

    plt.figure(figsize=(8, 5))
    if train_epochs:
        plt.plot(train_epochs, mean_train, "-o", label="Train total loss")
    if eval_epochs:
        plt.plot(eval_epochs, mean_eval, "-o", label="Val total loss")

    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Train / Val total loss per epoch")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(plot_path, dpi=150)
    plt.close()

    print(f"График потерь сохранён в: {plot_path}")

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
    print(f"  input_ids length: {len(sample['input_ids'])}")
    print(f"  labels length: {len(sample['labels'])}")
    print(f"  labels != -100: {sum(1 for x in sample['labels'] if x != -100)}")
    print(f"  labels не -100: {(sample['labels'] != -100).sum().item()} токенов")
    print(f"  pad_token_id: {tokenizer.pad_token_id}")

    # 4. Аргументы обучения
    print("\n4. Настройка параметров обучения...")
    args = Seq2SeqTrainingArguments(
        output_dir=str(config.MODELS_DIR / "checkpoints"),
        evaluation_strategy="epoch",  # современный параметр в transformers>=4.36
        save_strategy="epoch",
        learning_rate=config.TRAIN_CONFIG["learning_rate"],
        per_device_train_batch_size=config.TRAIN_CONFIG["batch_size"],
        per_device_eval_batch_size=config.TRAIN_CONFIG["batch_size"],
        weight_decay=0.01,
        save_total_limit=2,
        num_train_epochs=config.TRAIN_CONFIG["epochs"],
        predict_with_generate=True,
        fp16=config.USE_FP16,
        logging_dir=str(config.LOGS_DIR),
        logging_strategy="epoch",
        dataloader_num_workers=0,  # 0 для Windows
        load_best_model_at_end=True,
        metric_for_best_model="loss",
        greater_is_better=False,
        report_to="none",
        remove_unused_columns=False,
        warmup_steps=config.TRAIN_CONFIG.get("warmup_steps", 100),
        gradient_accumulation_steps=1,
        max_grad_norm=1.0,
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
        train_output = trainer.train()
        print("\n" + "=" * 50)
        print("ОБУЧЕНИЕ ЗАВЕРШЕНО УСПЕШНО!")
        print("=" * 50)
    except Exception as e:
        print(f"\nОШИБКА при обучении: {e}")
        import traceback
        traceback.print_exc()
        return

    # 5. Построение графика потерь по эпохам
    print("\n6. Построение графика потерь...")
    _plot_losses_from_history(trainer.state.log_history, config.LOGS_DIR)

    # 6. Сохранение
    print("\n7. Сохранение модели...")
    final_path = config.MODELS_DIR / "korean_bot_final"
    model.save_pretrained(final_path)
    tokenizer.save_pretrained(final_path)
    print(f"Модель сохранена в: {final_path}")

if __name__ == "__main__":
    train()
