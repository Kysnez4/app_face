
import json
import os
from PyQt5.QtWidgets import (QWidget, QPushButton, QLabel,
                             QFileDialog, QLineEdit, QVBoxLayout, QHBoxLayout,
                             QTextEdit, QProgressBar, QComboBox, QMessageBox, QSizePolicy)
from PyQt5.QtCore import QThread, pyqtSignal, Qt
from PyQt5.QtGui import QPixmap, QImage
from src.utils import (create_dataset_csv, FaceCompareModel, CustomModel, load_image,
                       contrastive_loss, save_model, load_model, copy_model, import_model)
import torch
from torchvision import transforms
from torch.optim import Adam
from torch.utils.data import Dataset, DataLoader
import pandas as pd
from tqdm import tqdm
import matplotlib.pyplot as plt
import io
import numpy as np


class TrainingWorker(QThread):
    progress_signal = pyqtSignal(int)
    log_signal = pyqtSignal(str)
    plot_signal = pyqtSignal(str)
    finished_signal = pyqtSignal(str)

    def __init__(self, params, csv_path):
        super().__init__()
        self.params = params
        self.csv_path = csv_path
        self.stop_training = False
        self.device = torch.device("cuda" if torch.cuda.is_available() and self.params['use_gpu'] else "cpu")
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    def _create_model(self):
        try:
            model_class = self.params['model_type']
            return model_class(self.params['embedding_size']).to(self.device)
        except Exception as e:
            raise ValueError(f"Не удалось создать модель: {e}")

    def stop(self):
        self.stop_training = True

    def run(self):
        self.log_signal.emit(f"Начинаем обучение... Используем устройство: {self.device}")
        if self.params['auto_train_models'] > 1 and self.params['auto_train_iterations'] > 1:
            self._auto_train()
        else:
            self._train()

    def _create_dataloader(self):
        class FaceDataset(Dataset):
            def __init__(self, csv_path, transform, resize_size):
                self.df = pd.read_csv(csv_path)
                self.transform = transform
                self.resize_size = resize_size

            def __len__(self):
                return len(self.df)

            def __getitem__(self, idx):
                row = self.df.iloc[idx]
                img1_path = row['image1']
                img2_path = row['image2']
                label = torch.tensor(row['label'], dtype=torch.float32)
                img1 = load_image(img1_path, self.transform, self.params['resize_size'])
                img2 = load_image(img2_path, self.transform, self.params['resize_size'])
                return (img1, img2, label) if img1 is not None and img2 is not None else None

        dataset = FaceDataset(self.csv_path, self.transform, self.params['resize_size'])
        dataset = [item for item in dataset if item is not None]
        return DataLoader(dataset, batch_size=self.params['batch_size'], shuffle=True)

    def _train_epoch(self, model, optimizer, dataloader):
        model.train()
        total_loss = 0
        for i, (img1_batch, img2_batch, label_batch) in enumerate(tqdm(dataloader)):
            optimizer.zero_grad()
            img1_batch = img1_batch.to(self.device)
            img2_batch = img2_batch.to(self.device)
            output1 = model(img1_batch)
            output2 = model(img2_batch)
            loss = contrastive_loss(output1, output2, label_batch.to(self.device))
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            self.progress_signal.emit(int((i + 1) / len(dataloader) * 100))
        return total_loss / len(dataloader)

    def _auto_train_epoch(self, model, optimizer, dataloader, model_idx, models_count):
        model.train()
        total_loss = 0
        for i, (img1_batch, img2_batch, label_batch) in enumerate(tqdm(dataloader)):
            optimizer.zero_grad()
            img1_batch = img1_batch.to(self.device)
            img2_batch = img2_batch.to(self.device)
            output1 = model(img1_batch)
            output2 = model(img2_batch)
            loss = contrastive_loss(output1, output2, label_batch.to(self.device))
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            self.progress_signal.emit(
                int((i + 1) / len(dataloader) * 100) // models_count + model_idx * (100 // models_count))
        return total_loss / len(dataloader)

    def _auto_train(self):
        dataloader = self._create_dataloader()
        best_model = None
        best_loss = float('inf')
        losses = []

        for iteration in range(self.params['auto_train_iterations']):
            self.log_signal.emit(f"Авто-обучение итерация: {iteration + 1}/{self.params['auto_train_iterations']}")

            models = [self._create_model() for _ in range(self.params['auto_train_models'])]
            optimizers = [Adam(model.parameters(), lr=self.params['learning_rate']) for model in models]
            iter_losses = []

            for model_idx, model in enumerate(models):
                self.log_signal.emit(f"Обучение модели {model_idx + 1}/{len(models)} в итерации {iteration + 1}...")
                model_losses = []
                for epoch in range(self.params['epochs']):
                    if self.stop_training:
                        self.log_signal.emit("Авто-обучение остановлено пользователем.")
                        return
                    avg_loss = self._auto_train_epoch(model, optimizers[model_idx], dataloader, model_idx, len(models))
                    model_losses.append(avg_loss)
                    self.log_signal.emit(
                        f"Эпоха {epoch + 1}/{self.params['epochs']}, Loss: {avg_loss:.4f} модель {model_idx + 1}")
                iter_losses.append(model_losses)
            losses.append(iter_losses)

            avg_losses = [sum(model_losses) / len(model_losses) for model_losses in iter_losses]
            min_loss_idx = avg_losses.index(min(avg_losses))
            if best_model is None or avg_losses[min_loss_idx] < best_loss:
                best_model = copy_model(models[min_loss_idx])
                best_loss = avg_losses[min_loss_idx]
                self.log_signal.emit(
                    f"Итерация {iteration + 1}: Найдена новая лучшая модель со средним loss: {best_loss:.4f}")
            else:
                self.log_signal.emit(f"Итерация {iteration + 1}: Нет улучшений, используем лучшую модель")

        plot_path = os.path.join(self.params['output_dir'], 'training_loss.png')
        self._plot_losses(losses, plot_path)
        self.plot_signal.emit(plot_path)
        save_model(best_model, self.params['model_path'])
        self.log_signal.emit(f"Авто-обучение завершено. Лучшая модель сохранена в: {self.params['model_path']}")
        self.finished_signal.emit("Авто-обучение завершено!")

    def _train(self):
        dataloader = self._create_dataloader()
        model = self._create_model()
        optimizer = Adam(model.parameters(), lr=self.params['learning_rate'])
        losses = []

        for epoch in range(self.params['epochs']):
            if self.stop_training:
                self.log_signal.emit("Обучение остановлено пользователем.")
                return
            avg_loss = self._train_epoch(model, optimizer, dataloader)
            losses.append(avg_loss)
            self.log_signal.emit(f"Эпоха {epoch + 1}/{self.params['epochs']}, Loss: {avg_loss:.4f}")

        plot_path = os.path.join(self.params['output_dir'], 'training_loss.png')
        self._plot_losses([losses], plot_path)
        self.plot_signal.emit(plot_path)
        save_model(model, self.params['model_path'])
        self.log_signal.emit(f"Обучение завершено. Модель сохранена в: {self.params['model_path']}")
        self.finished_signal.emit("Обучение завершено!")

    def _plot_losses(self, losses, plot_path):
        plt.figure(figsize=(10, 5))
        for i, model_losses in enumerate(losses):
            if isinstance(model_losses[0], list):  # Handle auto-train losses
                for j, loss in enumerate(model_losses):
                     plt.plot(loss, label=f"Модель {j + 1} - итерация {i + 1}")
            else:
                plt.plot(model_losses, label=f"Модель {i + 1}")

        plt.title('График потерь во время обучения')
        plt.xlabel('Эпохи')
        plt.ylabel('Потери')
        plt.legend()
        plt.savefig(plot_path)


class FaceRecognitionApp(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Тренер распознавания лиц")
        self.setGeometry(100, 100, 800, 600)
        self.current_config = self._load_config()
        self._init_ui()

    def _init_ui(self):
        tooltips = {
            'data_dir': "Путь к каталогу, содержащему изображения для обучения.",
            'output_dir': "Каталог, в который будут сохранены результаты обучения (модели, графики, CSV)",
            'model_path': "Путь к файлу, в который будет сохранена или загружена обученная модель",
            'model_type': "Тип модели для обучения. Выберите из доступных опций.",
            'negative_pairs': "Количество отрицательных пар, которые будут сгенерированы в CSV",
            'epochs': "Количество эпох обучения",
            'batch_size': "Размер пакета данных для обучения",
            'learning_rate': "Скорость обучения",
            'embedding_size': "Размерность векторного представления лица",
            'resize_size': "Размер изображения для обучения",
            'auto_train_models': "Количество моделей, которые будут обучены в режиме автоматического обучения",
            'auto_train_iterations': "Количество итераций автоматического обучения",
            'gpu': "Использовать GPU для обучения, если доступно"
        }

        self.data_dir_label = QLabel("Каталог с данными:")
        self.data_dir_label.setToolTip(tooltips['data_dir'])
        self.data_dir_line = QLineEdit(self.current_config.get('data_dir', ''))
        self.data_dir_button = QPushButton("Обзор", clicked=self._browse_data_dir)

        self.output_dir_label = QLabel("Каталог вывода:")
        self.output_dir_label.setToolTip(tooltips['output_dir'])
        self.output_dir_line = QLineEdit(self.current_config.get('output_dir', ''))
        self.output_dir_button = QPushButton("Обзор", clicked=self._browse_output_dir)

        self.model_path_label = QLabel("Путь к модели:")
        self.model_path_label.setToolTip(tooltips['model_path'])
        self.model_path_line = QLineEdit(self.current_config.get('model_path', ''))
        self.model_path_button = QPushButton("Обзор", clicked=self._browse_model_path)

        self.model_type_label = QLabel("Тип модели:")
        self.model_type_label.setToolTip(tooltips['model_type'])
        self.model_type_combo = QComboBox()
        available_models = {'FaceCompareModel': FaceCompareModel, 'CustomModel': CustomModel}
        self.model_type_combo.addItems(list(available_models.keys()))
        default_model = self.current_config.get('model_type', 'FaceCompareModel')
        self.model_type_combo.setCurrentText(default_model)
        self.available_models = available_models


        self.negative_pairs_label = QLabel("Отрицательных пар:")
        self.negative_pairs_label.setToolTip(tooltips['negative_pairs'])
        self.negative_pairs_line = QLineEdit(str(self.current_config.get('negative_pairs', 1000)))

        self.epochs_label = QLabel("Эпохи:")
        self.epochs_label.setToolTip(tooltips['epochs'])
        self.epochs_line = QLineEdit(str(self.current_config.get('epochs', 10)))

        self.batch_size_label = QLabel("Размер пакета:")
        self.batch_size_label.setToolTip(tooltips['batch_size'])
        self.batch_size_line = QLineEdit(str(self.current_config.get('batch_size', 32)))

        self.learning_rate_label = QLabel("Скорость обучения:")
        self.learning_rate_label.setToolTip(tooltips['learning_rate'])
        self.learning_rate_line = QLineEdit(str(self.current_config.get('learning_rate', 0.001)))

        self.embedding_size_label = QLabel("Размер эмбеддинга:")
        self.embedding_size_label.setToolTip(tooltips['embedding_size'])
        self.embedding_size_line = QLineEdit(str(self.current_config.get('embedding_size', 128)))

        self.resize_size_label = QLabel("Размер изображения:")
        self.resize_size_label.setToolTip(tooltips['resize_size'])
        self.resize_size_line = QLineEdit(str(self.current_config.get('resize_size', 224)))


        self.auto_train_models_label = QLabel("Количество моделей для авто-обучения:")
        self.auto_train_models_label.setToolTip(tooltips['auto_train_models'])
        self.auto_train_models_line = QLineEdit(str(self.current_config.get('auto_train_models', 1)))

        self.auto_train_iterations_label = QLabel("Количество итераций авто-обучения:")
        self.auto_train_iterations_label.setToolTip(tooltips['auto_train_iterations'])
        self.auto_train_iterations_line = QLineEdit(str(self.current_config.get('auto_train_iterations', 1)))


        self.gpu_label = QLabel("Использовать GPU:")
        self.gpu_label.setToolTip(tooltips['gpu'])
        self.gpu_combo = QComboBox()
        self.gpu_combo.addItems(["False", "True"])
        self.gpu_combo.setCurrentText(str(self.current_config.get('use_gpu', False)))

        self.generate_csv_button = QPushButton("Создать CSV", clicked=self._generate_csv)
        self.train_button = QPushButton("Начать обучение", clicked=self._start_training, enabled=False)
        self.stop_button = QPushButton("Остановить обучение", clicked=self._stop_training, enabled=False)
        self.load_model_button = QPushButton("Загрузить модель", clicked=self._load_model_func)

        self.log_area = QTextEdit(readOnly=True)
        self.progress_bar = QProgressBar(value=0)
        self.image_label = QLabel()
        self.image_label.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.image_label.setAlignment(Qt.AlignCenter)

        # Layout setup
        data_layout = QHBoxLayout()
        data_layout.addWidget(self.data_dir_label)
        data_layout.addWidget(self.data_dir_line)
        data_layout.addWidget(self.data_dir_button)

        model_type_layout = QHBoxLayout()
        model_type_layout.addWidget(self.model_type_label)
        model_type_layout.addWidget(self.model_type_combo)

        output_layout = QHBoxLayout()
        output_layout.addWidget(self.output_dir_label)
        output_layout.addWidget(self.output_dir_line)
        output_layout.addWidget(self.output_dir_button)

        model_layout = QHBoxLayout()
        model_layout.addWidget(self.model_path_label)
        model_layout.addWidget(self.model_path_line)
        model_layout.addWidget(self.model_path_button)

        neg_pair_layout = QHBoxLayout()
        neg_pair_layout.addWidget(self.negative_pairs_label)
        neg_pair_layout.addWidget(self.negative_pairs_line)

        epochs_layout = QHBoxLayout()
        epochs_layout.addWidget(self.epochs_label)
        epochs_layout.addWidget(self.epochs_line)

        batch_layout = QHBoxLayout()
        batch_layout.addWidget(self.batch_size_label)
        batch_layout.addWidget(self.batch_size_line)

        lr_layout = QHBoxLayout()
        lr_layout.addWidget(self.learning_rate_label)
        lr_layout.addWidget(self.learning_rate_line)

        emb_layout = QHBoxLayout()
        emb_layout.addWidget(self.embedding_size_label)
        emb_layout.addWidget(self.embedding_size_line)

        resize_layout = QHBoxLayout()
        resize_layout.addWidget(self.resize_size_label)
        resize_layout.addWidget(self.resize_size_line)

        auto_train_layout = QHBoxLayout()
        auto_train_layout.addWidget(self.auto_train_models_label)
        auto_train_layout.addWidget(self.auto_train_models_line)

        auto_train_iter_layout = QHBoxLayout()
        auto_train_iter_layout.addWidget(self.auto_train_iterations_label)
        auto_train_iter_layout.addWidget(self.auto_train_iterations_line)

        gpu_layout = QHBoxLayout()
        gpu_layout.addWidget(self.gpu_label)
        gpu_layout.addWidget(self.gpu_combo)

        button_layout = QHBoxLayout()
        button_layout.addWidget(self.generate_csv_button)
        button_layout.addWidget(self.train_button)
        button_layout.addWidget(self.stop_button)
        button_layout.addWidget(self.load_model_button)

        main_layout = QVBoxLayout()
        main_layout.addLayout(data_layout)
        main_layout.addLayout(model_type_layout)
        main_layout.addLayout(output_layout)
        main_layout.addLayout(model_layout)
        main_layout.addLayout(neg_pair_layout)
        main_layout.addLayout(epochs_layout)
        main_layout.addLayout(batch_layout)
        main_layout.addLayout(lr_layout)
        main_layout.addLayout(emb_layout)
        main_layout.addLayout(resize_layout)
        main_layout.addLayout(auto_train_layout)
        main_layout.addLayout(auto_train_iter_layout)
        main_layout.addLayout(gpu_layout)
        main_layout.addLayout(button_layout)
        main_layout.addWidget(self.progress_bar)
        main_layout.addWidget(self.log_area)
        main_layout.addWidget(self.image_label)
        self.setLayout(main_layout)

    def _browse_data_dir(self):
        dir_path = QFileDialog.getExistingDirectory(self, "Выберите каталог с данными")
        if dir_path:
            self.data_dir_line.setText(dir_path)

    def _browse_output_dir(self):
        dir_path = QFileDialog.getExistingDirectory(self, "Выберите каталог вывода")
        if dir_path:
            self.output_dir_line.setText(dir_path)

    def _browse_model_path(self):
        file_path, _ = QFileDialog.getSaveFileName(self, "Сохранить путь к модели", "", "PyTorch модель (*.pth)")
        if file_path:
            self.model_path_line.setText(file_path)

    def _generate_csv(self):
        data_dir = self.data_dir_line.text()
        output_dir = self.output_dir_line.text()
        num_negative = self.negative_pairs_line.text()

        if not all([data_dir, output_dir, num_negative]):
            QMessageBox.warning(self, "Предупреждение", "Пожалуйста, заполните все каталоги и параметры")
            return

        try:
            num_negative = int(num_negative)
        except ValueError:
            QMessageBox.warning(self, "Ошибка", "Неверные параметры для отрицательных пар")
            return

        if not os.path.exists(os.path.join(data_dir, 'students')):
            QMessageBox.warning(self, "Предупреждение", "Каталог с данными должен содержать подкаталог 'students'")
            return

        self.log_area.append("Начинаем создание csv...")
        csv_path = create_dataset_csv(data_dir, output_dir, num_negative)
        if csv_path:
            self.log_area.append(f"CSV файл создан: {csv_path}")
            self.train_button.setEnabled(True)
            self._save_config()
            self.csv_path = csv_path
        else:
            self.log_area.append("Не удалось создать csv файл!")

    def _start_training(self):
        if not hasattr(self, 'csv_path') or not self.csv_path:
            QMessageBox.warning(self, "Предупреждение", "Пожалуйста, создайте CSV файл перед началом обучения.")
            return

        params = {
            'data_dir': self.data_dir_line.text(),
            'output_dir': self.output_dir_line.text(),
            'model_type': self.available_models[self.model_type_combo.currentText()],
            'model_path': self.model_path_line.text(),
            'epochs': self.epochs_line.text(),
            'batch_size': self.batch_size_line.text(),
            'learning_rate': self.learning_rate_line.text(),
            'use_gpu': self.gpu_combo.currentText(),
            'embedding_size': self.embedding_size_line.text(),
            'resize_size': self.resize_size_line.text(),
            'auto_train_models': self.auto_train_models_line.text(),
            'auto_train_iterations': self.auto_train_iterations_line.text()
        }

        try:
            params['epochs'] = int(params['epochs'])
            params['batch_size'] = int(params['batch_size'])
            params['learning_rate'] = float(params['learning_rate'])
            params['embedding_size'] = int(params['embedding_size'])
            params['resize_size'] = int(params['resize_size'])
            params['auto_train_models'] = int(params['auto_train_models'])
            params['auto_train_iterations'] = int(params['auto_train_iterations'])
            params['use_gpu'] = params['use_gpu'] == "True"

        except ValueError as e:
            QMessageBox.warning(self, "Ошибка", f"Неверные параметры обучения: {e}")
            return

        self.log_area.append("Начинаем обучение...")
        self.progress_bar.setValue(0)
        self.train_button.setEnabled(False)
        self.stop_button.setEnabled(True)

        self.training_worker = TrainingWorker(params=params, csv_path=self.csv_path)
        self.training_worker.progress_signal.connect(self._update_progress)
        self.training_worker.log_signal.connect(self._update_log)
        self.training_worker.plot_signal.connect(self._display_plot)
        self.training_worker.finished_signal.connect(self._training_finished)
        self.training_worker.start()
        self._save_config()

    def _stop_training(self):
        if hasattr(self, 'training_worker') and self.training_worker.isRunning():
            self.training_worker.stop()
            self.log_area.append("Останавливаем обучение...")
            self.stop_button.setEnabled(False)

    def _load_model_func(self):
        model_path = self.model_path_line.text()
        embedding_size = self.embedding_size_line.text()
        if not all([model_path, embedding_size]):
            QMessageBox.warning(self, "Предупреждение",
                                "Пожалуйста, укажите путь к модели и размер эмбеддинга перед загрузкой.")
            return
        try:
            embedding_size = int(embedding_size)
        except ValueError:
            QMessageBox.warning(self, "Ошибка", "Неверный размер эмбеддинга")
            return

        try:
            load_model(model_path, embedding_size)
            self.log_area.append(f"Модель загружена из {model_path}")
        except Exception as e:
            self.log_area.append(f"Ошибка загрузки модели: {e}")

    def _update_progress(self, value):
        self.progress_bar.setValue(value)

    def _update_log(self, message):
        self.log_area.append(message)

    def _display_plot(self, plot_path):
        try:
            pixmap = self._load_plot_pixmap(plot_path)
            self.image_label.setPixmap(pixmap)
            self.image_label.setAlignment(Qt.AlignCenter)
            self.image_label.show()
        except Exception as e:
            self.log_area.append(f"Ошибка отображения графика: {e}")


    def _load_plot_pixmap(self, plot_path):
        try:
            img_data = plt.imread(plot_path)
            height, width, channel = img_data.shape
            bytes_per_line = channel * width
            q_img = QImage(img_data.data, width, height, bytes_per_line, QImage.Format_RGB888)
            return QPixmap.fromImage(q_img)
        except Exception as e:
            self.log_area.append(f"Ошибка преобразования графика в QPixmap: {e}")
            return QPixmap()


    def _training_finished(self, message):
        self.log_area.append(message)
        self.train_button.setEnabled(True)
        self.stop_button.setEnabled(False)

    def _save_config(self):
        config = {
            'data_dir': self.data_dir_line.text(),
            'output_dir': self.output_dir_line.text(),
            'model_path': self.model_path_line.text(),
            'negative_pairs': self.negative_pairs_line.text(),
            'epochs': self.epochs_line.text(),
            'batch_size': self.batch_size_line.text(),
            'learning_rate': self.learning_rate_line.text(),
            'use_gpu': self.gpu_combo.currentText(),
            'embedding_size': self.embedding_size_line.text(),
            'resize_size': self.resize_size_line.text(),
            'auto_train_models': self.auto_train_models_line.text(),
            'auto_train_iterations': self.auto_train_iterations_line.text(),
             'model_type': self.model_type_combo.currentText()
        }
        with open('src/config.json', 'w') as f:
            json.dump(config, f)
        self.current_config = config

    def _load_config(self):
        try:
            with open('src/config.json', 'r') as f:
                config = json.load(f)
                if 'model_type' in config:
                    return config
                return {}
        except FileNotFoundError:
            return {}


    def closeEvent(self, event):
        self._save_config()
        event.accept()

