import time
import pandas as pd
import re
import nltk
import torch
import pymorphy3

from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from sklearn.model_selection import train_test_split
from torch.optim import AdamW
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer, BertForSequenceClassification, get_scheduler


nltk.download('stopwords')
nltk.download('punkt')
nltk.download('punkt_tab')

stop_words = set(stopwords.words('russian'))
morph = pymorphy3.MorphAnalyzer()

class SentimentDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_len=256):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = str(self.texts[idx])
        label = int(self.labels[idx])

        encoding = self.tokenizer(
            text,
            padding="max_length",
            truncation=True,
            max_length=self.max_len,
            return_tensors="pt"
        )

        return {
            "input_ids": encoding["input_ids"].flatten(),
            "attention_mask": encoding["attention_mask"].flatten(),
            "labels": torch.tensor(label, dtype=torch.long)
        }

def initData():
    df = pd.read_csv('sentiment_dataset.csv')
    pd.set_option('display.max_columns', None)
    pd.set_option("display.width", 1000)

    print("\nВид датасета:")
    print(df)
    print("\nПропущенные значения в столбцах:")
    print(df.isnull().sum())
    print("\nКоличество дубликатов в тексте:")
    print(df.duplicated(subset='text').sum())
    df.drop_duplicates(subset='text', inplace=True)

    print("Начинаем предобработку текста...")
    start = time.time()
    df['processed_text'] = df['text'].apply(preprocessText)
    end = time.time() - start

    print(f"Предобработка завершена за: {end:.4f} секунд")
    print("\nПримеры обработанных текстов:")

    for i in range(3):
        print(f"Оригинал: {df['text'].iloc[i][:100]}...")
        print(f"Обработанный: {df['processed_text'].iloc[i]}")
        print(f"Метка: {df['label'].iloc[i]}")
        print("\n----------------------------------------")

    print(f"\nСтатистика датасета:")
    print(f"Всего записей: {len(df)}")
    print(f"Нейтральные (0): {len(df[df['label'] == 0])}")
    print(f"Позитивные (1): {len(df[df['label'] == 1])}")
    print(f"Негативные (2): {len(df[df['label'] == 2])}")

    df.to_csv('preprocessed_reviews.csv', index=False, encoding='utf-8')
    print("\nОбработанные данные сохранены в preprocessed_reviews.csv")

    return df

def initDataTest():
    df = pd.read_csv('preprocessed_reviews.csv')
    print("\nВид датасета:")
    print(df)
    return df

def preprocessText(text):
    if not isinstance(text, str):
        return ""

    text = text.lower()
    text = re.sub(r'[^а-яёa-z\s]', ' ', text)
    text = re.sub(r'\s+', ' ', text)
    text = text.strip()
    tokens = word_tokenize(text, language='russian')
    tokens = [token for token in tokens if token not in stop_words]
    tokens = [morph.parse(token)[0].normal_form for token in tokens]

    tokens = [token for token in tokens if len(token) > 2]
    return ' '.join(tokens)

def vectorizedData(df, batch_size=16):
    X_trainval, X_test, y_trainval, y_test = train_test_split(
        df["processed_text"], df["label"],
        test_size=0.2, random_state=42, stratify=df["label"]
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_trainval, y_trainval,
        test_size=0.125,
        random_state=42,
        stratify=y_trainval
    )

    tokenizer = BertTokenizer.from_pretrained("bert-base-multilingual-cased")

    train_dataset = SentimentDataset(X_train.tolist(), y_train.tolist(), tokenizer)
    val_dataset = SentimentDataset(X_val.tolist(), y_val.tolist(), tokenizer)
    test_dataset = SentimentDataset(X_test.tolist(), y_test.tolist(), tokenizer)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)
    test_loader = DataLoader(test_dataset, batch_size=batch_size)

    return train_loader, val_loader, test_loader, tokenizer

def trainNeuron(train_loader, val_loader, test_loader, tokenizer, save_path, patience=2, num_epochs=3):
    print("CUDA доступна:", torch.cuda.is_available())
    if torch.cuda.is_available():
        print("Имя GPU:", torch.cuda.get_device_name(0))
        print("Количество GPU:", torch.cuda.device_count())
        print("Текущий GPU:", torch.cuda.current_device())
    else:
        print("Работаем на CPU")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = BertForSequenceClassification.from_pretrained(
        "bert-base-multilingual-cased",
        num_labels=3
    ).to(device)

    optimizer = AdamW(model.parameters(), lr=2e-5)
    num_training_steps = len(train_loader) * num_epochs
    lr_scheduler = get_scheduler(
        "linear", optimizer=optimizer, num_warmup_steps=0, num_training_steps=num_training_steps
    )

    best_val_loss = float("inf")
    best_epoch = 0
    patience_counter = 0

    start = time.time()
    for epoch in range(num_epochs):
        print(f"\nEpoch {epoch+1}/{num_epochs}")
        model.train()
        total_loss = 0

        for batch in train_loader:
            optimizer.zero_grad()
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss
            total_loss += loss.item()
            loss.backward()

            optimizer.step()
            lr_scheduler.step()

        avg_train_loss = total_loss / len(train_loader)
        print(f"Train Loss: {avg_train_loss:.4f}")

        model.eval()
        val_loss, val_correct, val_total = 0, 0, 0
        with torch.no_grad():
            for batch in val_loader:
                input_ids = batch["input_ids"].to(device)
                attention_mask = batch["attention_mask"].to(device)
                labels = batch["labels"].to(device)

                outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
                val_loss += outputs.loss.item()

                preds = torch.argmax(outputs.logits, dim=1)
                val_correct += (preds == labels).sum().item()
                val_total += labels.size(0)

        avg_val_loss = val_loss / len(val_loader)
        val_acc = val_correct / val_total
        print(f"Validation Loss: {avg_val_loss:.4f} | Accuracy: {val_acc:.2%}")

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            best_epoch = epoch
            patience_counter = 0
            model.save_pretrained(save_path)
            tokenizer.save_pretrained(save_path)
            print("Сохранена новая лучшая модель!")
        else:
            patience_counter += 1
            print(f"Потеря валидации не улучшилась ({patience_counter}/{patience})")
            if patience_counter >= patience:
                print("Early stopping сработал")
                break


    end = time.time() - start
    print(f"Время обучения: {end:.4f} секунд")
    print(f"Лучшая эпоха: {best_epoch+1}, Val Loss = {best_val_loss:.4f}")

    print("\nОценка на тестовых данных")
    model = BertForSequenceClassification.from_pretrained(save_path).to(device)
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for batch in test_loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            preds = torch.argmax(outputs.logits, dim=1)

            correct += (preds == labels).sum().item()
            total += labels.size(0)

    print(f"Test Accuracy: {correct / total:.2%}")

    return save_path, device

def predict_text(model_path, text):
    tokenizer = BertTokenizer.from_pretrained(model_path)
    model = BertForSequenceClassification.from_pretrained(model_path)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()

    processed = preprocessText(text)
    encoding = tokenizer(
        processed,
        return_tensors="pt",
        padding="max_length",
        truncation=True,
        max_length=128
    )

    input_ids = encoding["input_ids"].to(device)
    attention_mask = encoding["attention_mask"].to(device)

    with torch.no_grad():
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        prediction = torch.argmax(outputs.logits, dim=1).item()
        probs = torch.softmax(outputs.logits, dim=1)
        confidence = torch.max(probs).item()

    mapping = {
        0: "нейтрально",
        1: "позитив",
        2: "негатив"
    }

    return {
        "prediction": mapping.get(prediction),
        "confidence": confidence,
        "probabilities": {
            "нейтрально": probs[0][0].item(),
            "позитив": probs[0][1].item(),
            "негатив": probs[0][2].item()
        }
    }

if __name__ == '__main__':
    df = initData()
    save_path = "./sentiment_model"
    train_loader, val_loader, test_loader, tokenizer = vectorizedData(df)
    trainNeuron(train_loader, val_loader, test_loader, tokenizer, save_path)

    test_reviews = [
        "Фильм оказался ужасным, еле досмотрел до конца.",
        "Доставка пришла с опозданием, коробка была повреждена.",
        "Очень разочарован сервисом, сотрудники грубили и не помогли.",
        "Я купил книгу вчера, пока ещё не читал.",
        "Приложение работает нормально, как и ожидалось.",
        "Посетил магазин, ассортимент стандартный, ничего особенного.",
        "Фильм был потрясающий, я получил огромное удовольствие!",
        "Товар превзошёл ожидания, качество на высоте.",
        "Очень понравилось обслуживание, всё быстро и вежливо.",
        "Отличный ресторан, еда вкусная, персонал дружелюбный."
    ]

    for i, review in enumerate(test_reviews, 1):
        prediction = predict_text(save_path, review)
        print(f"{i}. {review}")
        print(f"Предсказание: {prediction}\n")
