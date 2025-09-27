import time
import pandas as pd
import re
import nltk
import torch
import pymorphy3
import matplotlib.pyplot as plt
import os
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from sklearn.model_selection import train_test_split
from torch.optim import AdamW
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer, BertForSequenceClassification, get_scheduler
import seaborn as sns
from wordcloud import WordCloud
from sklearn.metrics import f1_score, confusion_matrix, ConfusionMatrixDisplay
import numpy as np
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
    print(df.head())
    print("\nПропущенные значения в столбцах:")
    print(df.isnull().sum())
    print("\nКоличество дубликатов в тексте:")
    print(df.duplicated(subset='text').sum())
    df.drop_duplicates(subset='text', inplace=True)

    plt.figure(figsize=(15, 5))

    plt.subplot(1, 3, 1)
    sns.countplot(x="label", data=df, palette="Set2")
    plt.title("Распределение классов")

    plt.subplot(1, 3, 2)
    df["text_len"] = df["text"].astype(str).apply(len)
    sns.histplot(df["text_len"], bins=30, kde=True, color="skyblue")
    plt.title("Длина исходных текстов")

    plt.subplot(1, 3, 3)
    sns.boxplot(x="label", y="text_len", data=df, palette="Set3")
    plt.title("Длина текстов по классам")

    plt.tight_layout()
    plt.savefig("dataset_overview.png")
    plt.close()
    print("Сохранён график: dataset_overview.png")

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

    plt.figure(figsize=(12, 5))

    df["processed_len"] = df["processed_text"].apply(lambda x: len(str(x).split()))
    plt.subplot(1, 2, 1)
    sns.histplot(df["processed_len"], bins=30, kde=True, color="orange")
    plt.title("Длина обработанных текстов (в словах)")

    text_corpus = " ".join(df["processed_text"])
    wordcloud = WordCloud(width=800, height=400, background_color="white").generate(text_corpus)
    plt.subplot(1, 2, 2)
    plt.imshow(wordcloud, interpolation="bilinear")
    plt.axis("off")
    plt.title("Часто встречающиеся слова")

    plt.tight_layout()
    plt.savefig("processed_text_analysis.png")
    plt.close()
    print("Сохранён график: processed_text_analysis.png")

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

    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    split_data = {
        "Train": y_train,
        "Validation": y_val,
        "Test": y_test
    }
    df_split = pd.concat([
        pd.DataFrame({"label": y_train, "split": "Train"}),
        pd.DataFrame({"label": y_val, "split": "Validation"}),
        pd.DataFrame({"label": y_test, "split": "Test"})
    ])
    sns.countplot(x="label", hue="split", data=df_split, palette="Set2")
    plt.title("Распределение классов по сплитам")

    plt.subplot(1, 2, 2)
    sizes = [len(y_train), len(y_val), len(y_test)]
    labels = ["Train", "Validation", "Test"]
    plt.bar(labels, sizes, color=["skyblue", "orange", "green"])
    plt.title("Размеры датасетов")
    for i, v in enumerate(sizes):
        plt.text(i, v + 5, str(v), ha="center")

    plt.tight_layout()
    plt.savefig("dataset_splits.png")
    plt.close()
    print("Сохранён график: dataset_splits.png")

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

    history = {
        "train_loss": [],
        "val_loss": [],
        "val_acc": []
    }

    os.makedirs(os.path.join(save_path, "plots"), exist_ok=True)

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
        history["train_loss"].append(avg_train_loss)
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
        history["val_loss"].append(avg_val_loss)
        history["val_acc"].append(val_acc)

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

        epochs_range = range(1, len(history["train_loss"]) + 1)

        plt.figure(figsize=(12, 5))

        plt.subplot(1, 2, 1)
        plt.plot(epochs_range, history["train_loss"], label="Train Loss")
        plt.plot(epochs_range, history["val_loss"], label="Validation Loss")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.title("Training & Validation Loss")
        plt.legend()

        plt.subplot(1, 2, 2)
        plt.plot(epochs_range, history["val_acc"], label="Validation Accuracy")
        plt.xlabel("Epoch")
        plt.ylabel("Accuracy")
        plt.title("Validation Accuracy")
        plt.legend()

        plt.suptitle(f"Epoch {epoch+1}/{num_epochs}", fontsize=14)
        plt.tight_layout()

        plt.savefig(os.path.join(save_path, "plots", f"training_epoch_{epoch+1}.png"))
        plt.close()

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

    test_acc = correct / total
    print(f"Test Accuracy: {test_acc:.2%}")

    epochs_range = range(1, len(history["train_loss"]) + 1)

    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    plt.plot(epochs_range, history["train_loss"], label="Train Loss")
    plt.plot(epochs_range, history["val_loss"], label="Validation Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training & Validation Loss")
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(epochs_range, history["val_acc"], label="Validation Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.title("Validation Accuracy")
    plt.legend()

    plt.suptitle(f"Training Summary (Test Accuracy: {test_acc:.2%})", fontsize=14)
    plt.tight_layout()

    plt.savefig(os.path.join(save_path, "plots", "training_summary.png"))
    plt.close()

    return save_path, device

def optimizedNeuron(train_loader, val_loader, test_loader, tokenizer, save_path,
                patience=2, num_epochs=4, lr=2e-5, weight_decay=0.01):

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

    optimizer = AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    num_training_steps = len(train_loader) * num_epochs
    num_warmup_steps = int(0.1 * num_training_steps)
    lr_scheduler = get_scheduler(
        "linear",
        optimizer=optimizer,
        num_warmup_steps=num_warmup_steps,
        num_training_steps=num_training_steps
    )

    best_val_loss = float("inf")
    best_epoch = 0
    patience_counter = 0

    history = {"train_loss": [], "val_loss": [], "val_acc": [], "lr": []}

    os.makedirs(os.path.join(save_path, "plots"), exist_ok=True)

    scaler = torch.amp.GradScaler("cuda")

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

            with torch.amp.autocast("cuda"):
                outputs = model(input_ids=input_ids,
                                attention_mask=attention_mask,
                                labels=labels)
                loss = outputs.loss

            scaler.scale(loss).backward()


            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

            scaler.step(optimizer)
            scaler.update()
            lr_scheduler.step()

            total_loss += loss.item()
            history["lr"].append(lr_scheduler.get_last_lr()[0])

        avg_train_loss = total_loss / len(train_loader)
        history["train_loss"].append(avg_train_loss)
        print(f"Train Loss: {avg_train_loss:.4f}")

        model.eval()
        val_loss, val_correct, val_total = 0, 0, 0
        with torch.no_grad():
            for batch in val_loader:
                input_ids = batch["input_ids"].to(device)
                attention_mask = batch["attention_mask"].to(device)
                labels = batch["labels"].to(device)

                with torch.amp.autocast("cuda"):
                    outputs = model(input_ids=input_ids,
                                    attention_mask=attention_mask,
                                    labels=labels)
                val_loss += outputs.loss.item()

                preds = torch.argmax(outputs.logits, dim=1)
                val_correct += (preds == labels).sum().item()
                val_total += labels.size(0)

        avg_val_loss = val_loss / len(val_loader)
        val_acc = val_correct / val_total
        history["val_loss"].append(avg_val_loss)
        history["val_acc"].append(val_acc)

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

        epochs_range = range(1, len(history["train_loss"]) + 1)

        plt.figure(figsize=(15, 5))

        plt.subplot(1, 3, 1)
        plt.plot(epochs_range, history["train_loss"], label="Train Loss")
        plt.plot(epochs_range, history["val_loss"], label="Validation Loss")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.title("Training & Validation Loss")
        plt.legend()

        plt.subplot(1, 3, 2)
        plt.plot(epochs_range, history["val_acc"], label="Validation Accuracy")
        plt.xlabel("Epoch")
        plt.ylabel("Accuracy")
        plt.title("Validation Accuracy")
        plt.legend()

        plt.subplot(1, 3, 3)
        plt.plot(history["lr"], label="Learning Rate")
        plt.xlabel("Step")
        plt.ylabel("LR")
        plt.title("LR Schedule")
        plt.legend()

        plt.suptitle(f"Epoch {epoch+1}/{num_epochs}", fontsize=14)
        plt.tight_layout()
        plt.savefig(os.path.join(save_path, "plots", f"training_epoch_{epoch+1}.png"))
        plt.close()

    end = time.time() - start
    print(f"Время обучения: {end:.4f} секунд")
    print(f"Лучшая эпоха: {best_epoch+1}, Val Loss = {best_val_loss:.4f}")

    print("\nОценка на тестовых данных")
    model = BertForSequenceClassification.from_pretrained(save_path).to(device)
    model.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for batch in test_loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            with torch.amp.autocast("cuda"):
                outputs = model(input_ids=input_ids, attention_mask=attention_mask)

            preds = torch.argmax(outputs.logits, dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    test_acc = np.mean(np.array(all_preds) == np.array(all_labels))
    test_f1 = f1_score(all_labels, all_preds, average="macro")
    print(f"Test Accuracy: {test_acc:.2%}")
    print(f"Test F1-score (macro): {test_f1:.4f}")

    cm = confusion_matrix(all_labels, all_preds)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=[0, 1, 2])
    disp.plot(cmap="Blues", values_format="d")
    plt.title("Confusion Matrix (Test Set)")
    plt.savefig(os.path.join(save_path, "plots", "confusion_matrix.png"))
    plt.close()

    epochs_range = range(1, len(history["train_loss"]) + 1)
    plt.figure(figsize=(15, 5))

    plt.subplot(1, 3, 1)
    plt.plot(epochs_range, history["train_loss"], label="Train Loss")
    plt.plot(epochs_range, history["val_loss"], label="Validation Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training & Validation Loss")
    plt.legend()

    plt.subplot(1, 3, 2)
    plt.plot(epochs_range, history["val_acc"], label="Validation Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.title("Validation Accuracy")
    plt.legend()

    plt.subplot(1, 3, 3)
    plt.plot(history["lr"], label="Learning Rate")
    plt.xlabel("Step")
    plt.ylabel("LR")
    plt.title("LR Schedule")
    plt.legend()

    plt.suptitle(f"Training Summary (Test Acc: {test_acc:.2%}, F1: {test_f1:.4f})", fontsize=14)
    plt.tight_layout()
    plt.savefig(os.path.join(save_path, "plots", "training_summary.png"))
    plt.close()

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

    optimizedNeuron(train_loader, val_loader, test_loader, tokenizer, save_path)
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