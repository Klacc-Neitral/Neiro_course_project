import pandas as pd
from sklearn.model_selection import train_test_split
from typing import Tuple, List

def load_chatbot_data(csv_path: str) -> pd.DataFrame:
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
