
import streamlit as st  # Библиотека для создания веб-приложений Streamlit
import cv2  # OpenCV для работы с изображениями
import numpy as np  # NumPy для работы с массивами
from PIL import Image  # Pillow (PIL) для работы с изображениями
import torch  # PyTorch - фреймворк для машинного обучения
from torchvision import transforms, datasets  # Transforms и Datasets из torchvision
import os  # Для работы с файловой системой
from model import FaceRecognitionModel, save_model, load_model  # Импортируем классы и функции из файла model.py
from utils import detect_and_align_face  # Импортируем функцию для обнаружения и выравнивания лиц
from torch.utils.data import DataLoader  # DataLoader для загрузки данных батчами

# Параметры
IMG_SIZE = 224  # Размер изображения для обучения
DATA_DIR = "E:\\Project\\app_face\\data\\lfw-deepfunneled\\students"  # Путь к директории с данными (замените на свой)
EMBEDDING_SIZE = 128  # Размерность эмбеддинга лица
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")  # Используем GPU, если доступен, иначе CPU
MODEL_PATH = 'models/face_recognition_model.pth'  # Путь для сохранения и загрузки модели

# Вспомогательные функции
def load_data(data_dir, transform, batch_size):
    """Загружает данные из директории, применяет преобразования и создает DataLoader."""
    dataset = datasets.ImageFolder(data_dir, transform=transform)  # Создаем ImageFolder датасет
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)  # Создаем DataLoader
    return dataloader, dataset.class_to_idx  # Возвращаем DataLoader и соответствие классов индексам

# Функция обучения
def train_model(model, data_dir, device, epochs=5, batch_size=32, callback=None):
    """Обучает модель распознавания лиц."""
    try:
        # Определяем преобразования данных
        data_transforms = transforms.Compose([
            transforms.Resize((IMG_SIZE, IMG_SIZE)),  # Изменяем размер изображения
            transforms.ToTensor(),  # Преобразуем в тензор PyTorch
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])  # Нормализуем данные
        ])

        # Загружаем данные
        dataloader, class_to_idx = load_data(data_dir, data_transforms, batch_size)  # Загружаем данные с помощью DataLoader
        num_classes = len(class_to_idx)  # Определяем количество классов

        # Функция потерь и оптимизатор
        criterion = torch.nn.CrossEntropyLoss()  # Используем кросс-энтропию в качестве функции потерь
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)  # Используем Adam в качестве оптимизатора (скорость обучения можно настроить)

        model.train()  # Переводим модель в режим обучения

        total_batches = len(dataloader)  # Общее количество батчей

        for epoch in range(epochs):  # Итерируемся по эпохам
            for batch_idx, (inputs, labels) in enumerate(dataloader):  # Итерируемся по батчам
                inputs = inputs.to(device)  # Переносим входные данные на устройство (GPU или CPU)
                labels = labels.to(device)  # Переносим метки на устройство

                optimizer.zero_grad()  # Обнуляем градиенты параметров

                outputs = model(inputs)  # Прямой проход (forward pass)
                loss = criterion(outputs, labels)  # Вычисляем функцию потерь

                loss.backward()  # Обратный проход (backward pass)
                optimizer.step()  # Обновляем параметры

                if callback:
                    callback(epoch, batch_idx, loss.item(), total_batches)  # Вызываем callback-функцию, если она задана

        return model  # Возвращаем обученную модель
    except Exception as e:
        st.error(f"Ошибка во время обучения: {e}")  # Выводим сообщение об ошибке в Streamlit
        return None  # Возвращаем None в случае ошибки

# Загрузка модели
@st.cache_resource  # Кэшируем функцию загрузки модели, чтобы она не вызывалась каждый раз
def cached_load_model(num_classes, embedding_size, model_path, device):
    """Загружает предварительно обученную модель."""
    try:
        return load_model(num_classes, embedding_size, model_path, device)  # Загружаем модель
    except Exception as e:
        st.error(f"Ошибка при загрузке модели: {e}")  # Выводим сообщение об ошибке в Streamlit
        return None  # Возвращаем None в случае ошибки

# Streamlit App
st.title("Распознавание лиц (PyTorch)")  # Заголовок приложения

# Убеждаемся, что модель загружена только один раз
if 'model' not in st.session_state:  # Проверяем, есть ли модель в session_state
    NUM_CLASSES = len(os.listdir(DATA_DIR))  # Определяем количество классов
    st.session_state['model'] = cached_load_model(NUM_CLASSES, EMBEDDING_SIZE, MODEL_PATH, DEVICE)  # Загружаем модель и сохраняем в session_state

model = st.session_state['model']  # Получаем модель из session_state

if model is None:
    st.error("Модель не загружена. Проверьте наличие файла модели.")  # Выводим сообщение об ошибке, если модель не загружена
else:
    # Загрузка изображения для распознавания
    uploaded_file = st.file_uploader("Загрузите изображение", type=["jpg", "jpeg", "png"])  # Загрузчик файлов в Streamlit

    if uploaded_file is not None:
        # ... (остальной код для загрузки и распознавания изображения)
        pass  # замените это кодом из вашего исходного приложения

    # Кнопка для обучения модели
    if st.button("Обучить модель"):  # Создаем кнопку "Обучить модель"
        try:
            # 1. Retrain Model (Переобучаем модель)
            NUM_CLASSES = len(os.listdir(DATA_DIR))  # Определяем количество классов
            st.write(f"Количество классов: {NUM_CLASSES}")  # Выводим отладочную информацию
            model = FaceRecognitionModel(NUM_CLASSES, EMBEDDING_SIZE).to(DEVICE)  # Создаем новый экземпляр модели и переносим на устройство

            # Streamlit progress bar (Индикатор прогресса Streamlit)
            progress_bar = st.progress(0)  # Создаем индикатор прогресса
            status_text = st.empty()  # Создаем пустое место для текста статуса
            epochs = 5  # Определяем количество эпох обучения
            batch_size = 32  # Определяем размер батча

            def training_callback(epoch, batch_idx, loss, total_batches):
                """Callback-функция для отображения прогресса обучения."""
                # Вычисляем прогресс
                progress = (epoch * total_batches + batch_idx) / (epochs * total_batches)
                progress_bar.progress(progress)  # Обновляем индикатор прогресса
                status_text.text(f"Эпоха: {epoch + 1}, Пакет: {batch_idx + 1}/{total_batches}, Loss: {loss:.4f}")  # Выводим информацию о текущем состоянии

            trained_model = train_model(model, DATA_DIR, DEVICE, epochs=epochs, batch_size=batch_size, callback=training_callback)  # Запускаем обучение

            if trained_model:
                save_model(trained_model, MODEL_PATH)  # Сохраняем обученную модель
                st.success("Модель сохранена!")  # Выводим сообщение об успехе

                # Load the model (Загружаем модель)
                model = load_model(NUM_CLASSES, EMBEDDING_SIZE, MODEL_PATH, DEVICE)  # Загружаем модель

                # Store the model in Streamlit's session state (Сохраняем модель в session_state)
                st.session_state['model'] = model  # Обновляем модель в session_state
                st.success("Модель успешно обучена!")  # Выводим сообщение об успехе
            else:
                st.error("Обучение не удалось.")  # Выводим сообщение об ошибке

        except Exception as e:
            st.error(f"Ошибка во время обучения: {e}")  # Выводим сообщение об ошибке

# Add New Face UI
# st.subheader("Добавить новое лицо")
# new_face_name = st.text_input("Имя нового лица")
# new_face_image = st.file_uploader("Загрузите фотографию нового лица", type=["jpg", "jpeg", "png"])

# if new_face_image is not None and new_face_name:
#     new_image = Image.open(new_face_image)
#     new_image = np.array(new_image)
#     st.image(new_image, caption="Фотография нового лица", use_column_width=True)

#     if st.button("Добавить лицо и переобучить модель"):
#         # ... (весь код для сохранения нового изображения и переобучения)
