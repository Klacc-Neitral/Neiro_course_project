# 📚 Подробное руководство по обучению модели в Google Colab

## 🎯 Цель
Обучить корейский чат-бот на базе модели KETI-AIR/ke-t5-small в Google Colab с использованием современных версий библиотек.

---

## 📋 Шаг 1: Подготовка Google Colab

### 1.1 Открытие Colab
1. Перейдите на [Google Colab](https://colab.research.google.com/)
2. Войдите в свой Google аккаунт
3. Создайте новый ноутбук: **File → New notebook**

### 1.2 Настройка GPU
1. В меню выберите: **Runtime → Change runtime type**
2. В разделе **Hardware accelerator** выберите **GPU** (T4 или лучше)
3. Нажмите **Save**
4. Проверьте GPU: выполните ячейку с кодом:
```python
!nvidia-smi
```

---

## 📦 Шаг 2: Установка библиотек (современные версии)

### 2.1 Установка основных библиотек
Создайте новую ячейку и выполните:

```python
# Установка PyTorch (последняя стабильная версия)
!pip install torch>=2.1.0 torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# Установка transformers и зависимостей
!pip install transformers>=4.36.0 accelerate>=0.25.0 sentencepiece>=0.1.99

# Установка библиотек для работы с данными
!pip install pandas>=2.1.0 numpy>=1.24.0 scikit-learn>=1.3.0

# Утилиты
!pip install tqdm>=4.66.0

# Проверка версий
import torch
import transformers
print(f"PyTorch: {torch.__version__}")
print(f"Transformers: {transformers.__version__}")
print(f"CUDA доступна: {torch.cuda.is_available()}")
print(f"GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'}")
```

### 2.2 Проверка установки
```python
import torch
print(f"CUDA доступна: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"Память GPU: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB")
```

---

## 📁 Шаг 3: Загрузка данных

### 3.1 Загрузка датасета
У вас есть несколько вариантов:

#### Вариант A: Загрузка через Google Drive
```python
from google.colab import drive
drive.mount('/content/drive')

# Скопируйте ваш CSV файл в Google Drive
# Путь к файлу (замените на ваш путь)
csv_path = '/content/drive/MyDrive/Neiro_course/data/final_chatbot_data.csv'
```

#### Вариант B: Загрузка напрямую в Colab
```python
from google.colab import files
uploaded = files.upload()

# Файл будет загружен в /content/
import os
csv_path = '/content/final_chatbot_data.csv'  # Или другое имя файла
```

#### Вариант C: Создание структуры проекта
```python
# Создаем структуру папок
!mkdir -p /content/Neiro_course/{data,models,logs,src}

# Загружаем файлы (используйте один из методов выше)
# Затем скопируйте файлы:
# !cp /content/drive/MyDrive/Neiro_course/data/final_chatbot_data.csv /content/Neiro_course/data/
```

---

## 💻 Шаг 4: Создание файлов проекта

### 4.1 Создание config.py
```python
%%writefile /content/Neiro_course/config.py
import torch
from pathlib import Path
import os

# Пути
BASE_DIR = Path("/content/Neiro_course")
DATA_DIR = BASE_DIR / "data"
MODELS_DIR = BASE_DIR / "models"
LOGS_DIR = BASE_DIR / "logs"

# Создаем папки
DATA_DIR.mkdir(exist_ok=True, parents=True)
MODELS_DIR.mkdir(exist_ok=True, parents=True)
LOGS_DIR.mkdir(exist_ok=True, parents=True)

# Путь к данным (ИЗМЕНИТЕ НА ВАШ ПУТЬ!)
CSV_DATA_PATH = DATA_DIR / "final_chatbot_data.csv"

# Конфигурация модели
MODEL_NAME = "KETI-AIR/ke-t5-small"

# Определение устройства
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
USE_FP16 = True  # ВКЛЮЧАЕМ для GPU в Colab

# Параметры обучения
TRAIN_CONFIG = {
    "max_length": 128,
    "batch_size": 16,        # Увеличиваем для GPU
    "learning_rate": 3e-4,
    "epochs": 5,
    "warmup_steps": 200,
    "logging_steps": 50,
}
```

### 4.2 Создание utils.py
```python
%%writefile /content/Neiro_course/src/utils.py
import pandas as pd
from sklearn.model_selection import train_test_split
from typing import Tuple

def load_chatbot_data(csv_path: str) -> pd.DataFrame:
    """Загрузка и очистка данных"""
    try:
        df = pd.read_csv(csv_path, encoding='utf-8')
        df.columns = [c.lower().strip() for c in df.columns]
        
        if 'question' in df.columns and 'answer' in df.columns:
            df = df[['question', 'answer']].copy()
        elif 'q' in df.columns and 'a' in df.columns:
            df = df.rename(columns={'q': 'question', 'a': 'answer'})[['question', 'answer']].copy()
        else:
            raise ValueError(f"CSV должен содержать колонки 'question' и 'answer'. Найдены: {df.columns.tolist()}")
        
        df = df.dropna()
        df['question'] = df['question'].astype(str).str.strip()
        df['answer'] = df['answer'].astype(str).str.strip()
        df = df[(df['question'].str.len() > 0) & (df['answer'].str.len() > 0)]
        df = df.drop_duplicates(subset=['question', 'answer'])
        
        print(f"Загружено строк: {len(df)}")
        return df.reset_index(drop=True)
    except Exception as e:
        print(f"Ошибка чтения файла: {e}")
        import traceback
        traceback.print_exc()
        return pd.DataFrame()

def split_dataset(df: pd.DataFrame, test_size=0.1) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Разделение на train/val"""
    return train_test_split(df, test_size=test_size, random_state=42)
```

### 4.3 Создание train.py
```python
%%writefile /content/Neiro_course/src/train.py
import sys
from pathlib import Path

# Добавляем корневую директорию в путь
BASE_DIR = Path("/content/Neiro_course")
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
        question = str(row['question']).strip()
        answer = str(row['answer']).strip()
        
        if not question or not answer:
            question = "안녕하세요"
            answer = "안녕하세요"

        # КРИТИЧНО: T5 требует префикс
        input_text = f"question: {question}"
        target_text = answer

        input_encodings = self.tokenizer(
            input_text,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        )

        target_encodings = self.tokenizer(
            target_text,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        )

        labels = target_encodings["input_ids"].clone()
        pad_token_id = self.tokenizer.pad_token_id
        if pad_token_id is None:
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
    print("НАЧАЛО ОБУЧЕНИЯ МОДЕЛИ В GOOGLE COLAB")
    print("=" * 50)
    
    # 1. Данные
    print(f"\n1. Загрузка данных из {config.CSV_DATA_PATH}...")
    df = load_chatbot_data(str(config.CSV_DATA_PATH))
    
    if df.empty:
        print("ОШИБКА: Датасет пуст или не найден!")
        return
    
    max_samples = 10000
    if len(df) > max_samples:
        print(f"Ограничиваем датасет до {max_samples} примеров")
        df = df.sample(n=max_samples, random_state=42).reset_index(drop=True)
    
    print(f"Загружено {len(df)} примеров")
    print(f"Пример вопроса: {df.iloc[0]['question'][:50]}...")
    print(f"Пример ответа: {df.iloc[0]['answer'][:50]}...")
    
    train_df, val_df = split_dataset(df, test_size=0.1)
    print(f"Обучающая выборка: {len(train_df)} примеров")
    print(f"Валидационная выборка: {len(val_df)} примеров")

    # 2. Модель и Токенизатор
    print(f"\n2. Загрузка модели {config.MODEL_NAME}...")
    print(f"Устройство: {config.DEVICE}")
    
    tokenizer = AutoTokenizer.from_pretrained(config.MODEL_NAME)
    
    if tokenizer.pad_token is None:
        if tokenizer.eos_token is not None:
            tokenizer.pad_token = tokenizer.eos_token
            print("Установлен pad_token = eos_token")
        else:
            tokenizer.add_special_tokens({'pad_token': '[PAD]'})
            print("Добавлен новый pad_token [PAD]")
    
    model = AutoModelForSeq2SeqLM.from_pretrained(config.MODEL_NAME)
    
    if tokenizer.pad_token_id != model.config.pad_token_id:
        model.resize_token_embeddings(len(tokenizer))
        model.config.pad_token_id = tokenizer.pad_token_id
    
    model.to(config.DEVICE)
    model.train()
    
    print(f"Модель загружена. Параметров: {sum(p.numel() for p in model.parameters()):,}")

    # 3. Датасеты
    print("\n3. Создание датасетов...")
    train_dataset = KoreanChatDataset(train_df, tokenizer, config.TRAIN_CONFIG['max_length'])
    val_dataset = KoreanChatDataset(val_df, tokenizer, config.TRAIN_CONFIG['max_length'])
    
    sample = train_dataset[0]
    print(f"Проверка примера:")
    print(f"  input_ids shape: {sample['input_ids'].shape}")
    print(f"  labels не -100: {(sample['labels'] != -100).sum().item()} токенов")

    # 4. Аргументы обучения
    print("\n4. Настройка параметров обучения...")
    args = Seq2SeqTrainingArguments(
        output_dir=str(config.MODELS_DIR / "checkpoints"),
        eval_strategy="steps",
        eval_steps=200,
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
        dataloader_num_workers=2,
        load_best_model_at_end=True,
        metric_for_best_model="loss",
        greater_is_better=False,
        report_to="none",
        remove_unused_columns=False,
        warmup_steps=config.TRAIN_CONFIG.get('warmup_steps', 200),
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
    
    # 6. Сохранение в Google Drive (опционально)
    print("\n7. Сохранение в Google Drive...")
    try:
        from google.colab import drive
        drive.mount('/content/drive')
        drive_path = Path("/content/drive/MyDrive/Neiro_course/models/korean_bot_final")
        drive_path.parent.mkdir(exist_ok=True, parents=True)
        
        import shutil
        if drive_path.exists():
            shutil.rmtree(drive_path)
        shutil.copytree(final_path, drive_path)
        print(f"Модель также сохранена в Google Drive: {drive_path}")
    except Exception as e:
        print(f"Не удалось сохранить в Google Drive: {e}")

if __name__ == "__main__":
    train()
```

---

## 🚀 Шаг 5: Запуск обучения

### 5.1 Запуск скрипта обучения
```python
# Убедитесь, что путь к данным правильный в config.py!
# Затем запустите:
import sys
sys.path.insert(0, '/content/Neiro_course')

from src.train import train
train()
```

### 5.2 Мониторинг обучения
Обучение займет примерно **2-3 часа** на GPU T4. Вы будете видеть логи каждые 50 шагов.

---

## 💾 Шаг 6: Сохранение и загрузка модели

### 6.1 Сохранение в Google Drive
```python
from google.colab import drive
drive.mount('/content/drive')

import shutil
from pathlib import Path

# Путь к обученной модели
model_path = Path("/content/Neiro_course/models/korean_bot_final")
drive_path = Path("/content/drive/MyDrive/Neiro_course/models/korean_bot_final")

# Создаем папку в Drive
drive_path.parent.mkdir(exist_ok=True, parents=True)

# Копируем модель
if model_path.exists():
    if drive_path.exists():
        shutil.rmtree(drive_path)
    shutil.copytree(model_path, drive_path)
    print(f"Модель сохранена в: {drive_path}")
else:
    print("Модель не найдена!")
```

### 6.2 Загрузка модели из Drive
```python
from google.colab import drive
drive.mount('/content/drive')

import shutil
from pathlib import Path

drive_path = Path("/content/drive/MyDrive/Neiro_course/models/korean_bot_final")
local_path = Path("/content/Neiro_course/models/korean_bot_final")

if drive_path.exists():
    local_path.parent.mkdir(exist_ok=True, parents=True)
    if local_path.exists():
        shutil.rmtree(local_path)
    shutil.copytree(drive_path, local_path)
    print(f"Модель загружена из Drive в: {local_path}")
else:
    print("Модель не найдена в Drive!")
```

---

## 🧪 Шаг 7: Тестирование модели

### 7.1 Тест обученной модели
```python
import sys
sys.path.insert(0, '/content/Neiro_course')

from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import torch

# Загрузка модели
model_path = "/content/Neiro_course/models/korean_bot_final"
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForSeq2SeqLM.from_pretrained(model_path)
model.to("cuda")
model.eval()

# Тест
test_question = "안녕하세요, 어떻게 지내세요?"
input_text = f"question: {test_question}"

inputs = tokenizer(input_text, return_tensors="pt", truncation=True, max_length=128).to("cuda")

with torch.no_grad():
    outputs = model.generate(
        inputs["input_ids"],
        attention_mask=inputs.get("attention_mask"),
        max_new_tokens=64,
        num_beams=4,
        early_stopping=True,
        no_repeat_ngram_size=3,
        do_sample=True,
        temperature=0.7,
        top_p=0.9,
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=tokenizer.eos_token_id,
        repetition_penalty=1.2
    )

response = tokenizer.decode(outputs[0], skip_special_tokens=True)
print(f"Вопрос: {test_question}")
print(f"Ответ: {response}")
```

---

## ⚠️ Важные замечания

1. **Время сессии**: Colab бесплатные сессии ограничены ~12 часами. Сохраняйте прогресс в Google Drive.

2. **Память GPU**: Если получаете ошибки памяти, уменьшите `batch_size` в config.py.

3. **Прерывание**: Если обучение прервалось, можно продолжить с чекпоинта:
   ```python
   trainer.train(resume_from_checkpoint=True)
   ```

4. **Версии библиотек**: Всегда используйте указанные версии для совместимости.

---

## 📊 Ожидаемые результаты

После обучения вы должны увидеть:
- **Loss**: должен снизиться с ~5-6 до ~1-2
- **Время обучения**: ~2-3 часа на 10000 примеров
- **Качество ответов**: модель должна отвечать на простые вопросы на корейском

---

## 🆘 Решение проблем

### Проблема: "Out of memory"
**Решение**: Уменьшите `batch_size` в config.py до 8 или 4

### Проблема: "File not found"
**Решение**: Проверьте путь к CSV файлу в config.py

### Проблема: "CUDA out of memory"
**Решение**: 
- Перезапустите runtime: **Runtime → Restart runtime**
- Уменьшите batch_size
- Используйте gradient_accumulation_steps=2

---

## ✅ Чеклист перед запуском

- [ ] GPU активирован в Colab
- [ ] Все библиотеки установлены
- [ ] CSV файл загружен и путь указан правильно
- [ ] Все файлы проекта созданы
- [ ] config.py настроен правильно
- [ ] Готовы к обучению 2-3 часа

---

**Удачи в обучении! 🚀**

