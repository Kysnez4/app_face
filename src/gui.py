import json
import os
from PyQt5.QtWidgets import (QWidget, QPushButton, QLabel,
                             QFileDialog, QLineEdit, QVBoxLayout, QHBoxLayout,
                             QTextEdit, QProgressBar, QComboBox, QMessageBox)
from PyQt5.QtCore import QThread, pyqtSignal
from src.utils import (create_dataset_csv, FaceCompareModel, load_image,
                       contrastive_loss, save_model, load_model, copy_model)
import torch
from torchvision import transforms
from torch.optim import Adam
from torch.utils.data import Dataset, DataLoader
import pandas as pd
from tqdm import tqdm
import matplotlib.pyplot as plt



class TrainingWorker(QThread):
    progress_signal = pyqtSignal(int)
    log_signal = pyqtSignal(str)
    plot_signal = pyqtSignal(str)
    finished_signal = pyqtSignal(str)

    def __init__(self, data_dir, output_dir, csv_path, epochs, batch_size, learning_rate, use_gpu, model_path,
                 embedding_size, resize_size, auto_train_models, auto_train_iterations):
        super().__init__()
        self.data_dir = data_dir
        self.output_dir = output_dir
        self.csv_path = csv_path
        self.epochs = epochs
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.use_gpu = use_gpu
        self.model_path = model_path
        self.stop_training = False
        self.embedding_size = embedding_size
        self.resize_size = resize_size
        self.auto_train_models = auto_train_models
        self.auto_train_iterations = auto_train_iterations

    def stop(self):
        self.stop_training = True

    def run(self):
        self.log_signal.emit("Starting training...")
        if self.auto_train_models > 1 and self.auto_train_iterations > 1:
            self.auto_train()
        else:
            self.train()

    def auto_train(self):
        device = torch.device("cuda" if torch.cuda.is_available() and self.use_gpu else "cpu")
        self.log_signal.emit(f"Using device: {device}")

        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

        class FaceDataset(Dataset):
            def __init__(self, csv_path, transform=transform, resize_size=self.resize_size):
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

                img1 = load_image(img1_path, self.transform, self.resize_size)
                img2 = load_image(img2_path, self.transform, self.resize_size)

                if img1 is None or img2 is None:
                    return None

                return img1, img2, label

        dataset = FaceDataset(self.csv_path)
        dataset = [item for item in dataset if item is not None]
        dataloader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)
        best_model = None
        best_loss = float('inf')

        for iteration in range(self.auto_train_iterations):
            self.log_signal.emit(f"Auto-train iteration: {iteration + 1}/{self.auto_train_iterations}")
            models = [FaceCompareModel(self.embedding_size).to(device) for _ in range(self.auto_train_models)]
            optimizers = [Adam(model.parameters(), lr=self.learning_rate) for model in models]

            losses = []

            for model_idx, model in enumerate(models):
                self.log_signal.emit(f"Training model {model_idx + 1}/{len(models)} in iteration {iteration + 1}...")
                model_losses = []
                for epoch in range(self.epochs):
                    if self.stop_training:
                        self.log_signal.emit("Auto-training stopped by user.")
                        return

                    model.train()
                    total_loss = 0
                    for i, (img1_batch, img2_batch, label_batch) in enumerate(tqdm(dataloader)):
                        optimizers[model_idx].zero_grad()
                        img1_batch = img1_batch.to(device)
                        img2_batch = img2_batch.to(device)

                        output1 = model(img1_batch)
                        output2 = model(img2_batch)

                        loss = contrastive_loss(output1, output2, label_batch.to(device))
                        loss.backward()
                        optimizers[model_idx].step()
                        total_loss += loss.item()
                        self.progress_signal.emit(
                            int((i + 1) / len(dataloader) * 100) // len(models) + model_idx * (100 // len(models)))
                    avg_loss = total_loss / len(dataloader)
                    model_losses.append(avg_loss)
                    self.log_signal.emit(f"Epoch {epoch + 1}/{self.epochs}, Loss: {avg_loss:.4f} model {model_idx + 1}")

                losses.append(model_losses)

            avg_losses = [sum(model_losses) / len(model_losses) for model_losses in losses]
            min_loss_idx = avg_losses.index(min(avg_losses))
            if best_model is None or avg_losses[min_loss_idx] < best_loss:
                best_model = copy_model(models[min_loss_idx])
                best_loss = avg_losses[min_loss_idx]
                self.log_signal.emit(f"Iteration {iteration + 1}: New best model found with avg loss: {best_loss:.4f}")
            else:
                self.log_signal.emit(f"Iteration {iteration + 1}: No improvement, using best model")

        plot_path = os.path.join(self.output_dir, 'training_loss.png')
        self.plot_losses(losses, plot_path)
        self.plot_signal.emit(plot_path)

        save_model(best_model, self.model_path)
        self.log_signal.emit(f"Auto-training finished. Best model saved to: {self.model_path}")
        self.finished_signal.emit("Auto-training completed!")

    def train(self):
        device = torch.device("cuda" if torch.cuda.is_available() and self.use_gpu else "cpu")
        self.log_signal.emit(f"Using device: {device}")

        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

        class FaceDataset(Dataset):
            def __init__(self, csv_path, transform=transform, resize_size=self.resize_size):
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

                img1 = load_image(img1_path, self.transform, self.resize_size)
                img2 = load_image(img2_path, self.transform, self.resize_size)

                if img1 is None or img2 is None:
                    return None

                return img1, img2, label

        dataset = FaceDataset(self.csv_path)
        dataset = [item for item in dataset if item is not None]
        dataloader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)

        model = FaceCompareModel(self.embedding_size).to(device)
        optimizer = Adam(model.parameters(), lr=self.learning_rate)

        losses = []
        for epoch in range(self.epochs):
            if self.stop_training:
                self.log_signal.emit("Training stopped by user.")
                return
            model.train()
            total_loss = 0
            for i, (img1_batch, img2_batch, label_batch) in enumerate(tqdm(dataloader)):
                optimizer.zero_grad()
                img1_batch = img1_batch.to(device)
                img2_batch = img2_batch.to(device)

                output1 = model(img1_batch)
                output2 = model(img2_batch)

                loss = contrastive_loss(output1, output2, label_batch.to(device))
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
                self.progress_signal.emit(int((i + 1) / len(dataloader) * 100))

            avg_loss = total_loss / len(dataloader)
            losses.append(avg_loss)

            self.log_signal.emit(f"Epoch {epoch + 1}/{self.epochs}, Loss: {avg_loss:.4f}")

        plt.figure(figsize=(10, 5))
        plt.plot(losses)
        plt.title('Training Loss')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plot_path = os.path.join(self.output_dir, 'training_loss.png')
        plt.savefig(plot_path)
        self.plot_signal.emit(plot_path)

        save_model(model, self.model_path)
        self.log_signal.emit(f"Training finished. Model saved to: {self.model_path}")
        self.finished_signal.emit("Training completed!")

    def plot_losses(self, losses, plot_path):
        plt.figure(figsize=(10, 5))
        for i, model_losses in enumerate(losses):
            plt.plot(model_losses, label=f"Model {i + 1}")
        plt.title('Training Loss')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.legend()
        plt.savefig(plot_path)


class FaceRecognitionApp(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Face Recognition Trainer")
        self.setGeometry(100, 100, 800, 600)

        self.data_dir = None
        self.output_dir = None
        self.csv_path = None
        self.model_path = None
        self.current_config = {}  # Initialize with default empty dict

        self.load_config()
        self.init_ui()

    def init_ui(self):
        self.data_dir_label = QLabel("Data Directory:")
        self.data_dir_line = QLineEdit()
        self.data_dir_line.setText(self.current_config.get('data_dir', ''))
        self.data_dir_button = QPushButton("Browse")
        self.data_dir_button.clicked.connect(self.browse_data_dir)

        self.output_dir_label = QLabel("Output Directory:")
        self.output_dir_line = QLineEdit()
        self.output_dir_line.setText(self.current_config.get('output_dir', ''))
        self.output_dir_button = QPushButton("Browse")
        self.output_dir_button.clicked.connect(self.browse_output_dir)

        self.model_path_label = QLabel("Model Path:")
        self.model_path_line = QLineEdit()
        self.model_path_line.setText(self.current_config.get('model_path', ''))
        self.model_path_button = QPushButton("Browse")
        self.model_path_button.clicked.connect(self.browse_model_path)

        self.negative_pairs_label = QLabel("Negative Pairs:")
        self.negative_pairs_line = QLineEdit()
        self.negative_pairs_line.setText(str(self.current_config.get('negative_pairs', 1000)))

        self.epochs_label = QLabel("Epochs:")
        self.epochs_line = QLineEdit()
        self.epochs_line.setText(str(self.current_config.get('epochs', 10)))

        self.batch_size_label = QLabel("Batch Size:")
        self.batch_size_line = QLineEdit()
        self.batch_size_line.setText(str(self.current_config.get('batch_size', 32)))

        self.learning_rate_label = QLabel("Learning Rate:")
        self.learning_rate_line = QLineEdit()
        self.learning_rate_line.setText(str(self.current_config.get('learning_rate', 0.001)))

        self.embedding_size_label = QLabel("Embedding Size:")
        self.embedding_size_line = QLineEdit()
        self.embedding_size_line.setText(str(self.current_config.get('embedding_size', 128)))

        self.resize_size_label = QLabel("Resize Size:")
        self.resize_size_line = QLineEdit()
        self.resize_size_line.setText(str(self.current_config.get('resize_size', 224)))

        self.auto_train_models_label = QLabel("Auto Train Models:")
        self.auto_train_models_line = QLineEdit()
        self.auto_train_models_line.setText(str(self.current_config.get('auto_train_models', 1)))

        self.auto_train_iterations_label = QLabel("Auto Train Iterations:")
        self.auto_train_iterations_line = QLineEdit()
        self.auto_train_iterations_line.setText(str(self.current_config.get('auto_train_iterations', 1)))

        self.gpu_label = QLabel("Use GPU:")
        self.gpu_combo = QComboBox()
        self.gpu_combo.addItems(["False", "True"])
        self.gpu_combo.setCurrentText(str(self.current_config.get('use_gpu', False)))

        self.generate_csv_button = QPushButton("Generate CSV")
        self.generate_csv_button.clicked.connect(self.generate_csv)

        self.train_button = QPushButton("Start Training")
        self.train_button.clicked.connect(self.start_training)
        self.train_button.setEnabled(False)

        self.stop_button = QPushButton("Stop Training")
        self.stop_button.clicked.connect(self.stop_training)
        self.stop_button.setEnabled(False)

        self.load_model_button = QPushButton("Load Model")
        self.load_model_button.clicked.connect(self.load_model_func)

        self.log_area = QTextEdit()
        self.log_area.setReadOnly(True)

        self.progress_bar = QProgressBar()
        self.progress_bar.setValue(0)

        self.image_label = QLabel()

        # Layout setup
        data_layout = QHBoxLayout()
        data_layout.addWidget(self.data_dir_label)
        data_layout.addWidget(self.data_dir_line)
        data_layout.addWidget(self.data_dir_button)

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

    def browse_data_dir(self):
        dir_path = QFileDialog.getExistingDirectory(self, "Select Data Directory")
        if dir_path:
            self.data_dir_line.setText(dir_path)
            self.data_dir = dir_path

    def browse_output_dir(self):
        dir_path = QFileDialog.getExistingDirectory(self, "Select Output Directory")
        if dir_path:
            self.output_dir_line.setText(dir_path)
            self.output_dir = dir_path

    def browse_model_path(self):
        file_path, _ = QFileDialog.getSaveFileName(self, "Save Model Path", "", "PyTorch Model (*.pth)")
        if file_path:
            self.model_path_line.setText(file_path)
            self.model_path = file_path

    def generate_csv(self):
        data_dir = self.data_dir_line.text()
        output_dir = self.output_dir_line.text()

        num_negative = self.negative_pairs_line.text()

        if not all([data_dir, output_dir, num_negative]):
            QMessageBox.warning(self, "Warning", "Please fill in all directory and parameter")
            return

        try:
            num_negative = int(num_negative)
        except ValueError:
            QMessageBox.warning(self, "Error", "Invalid parameters for negative pairs")
            return

        if not os.path.exists(os.path.join(data_dir, 'students')):
            QMessageBox.warning(self, "Warning", "Data Directory must contain a 'students' subdirectory")
            return

        self.log_area.append("Starting generate csv...")

        self.csv_path = create_dataset_csv(data_dir, output_dir, num_negative)
        if self.csv_path:
            self.log_area.append(f"CSV file created: {self.csv_path}")
            self.train_button.setEnabled(True)
            self.save_config()
        else:
            self.log_area.append("Failed to create csv file!")

    def start_training(self):
        if not self.csv_path:
            QMessageBox.warning(self, "Warning", "Please generate CSV file before starting training.")
            return

        data_dir = self.data_dir_line.text()
        output_dir = self.output_dir_line.text()
        model_path = self.model_path_line.text()
        epochs = self.epochs_line.text()
        batch_size = self.batch_size_line.text()
        learning_rate = self.learning_rate_line.text()
        use_gpu = self.gpu_combo.currentText()
        embedding_size = self.embedding_size_line.text()
        resize_size = self.resize_size_line.text()
        auto_train_models = self.auto_train_models_line.text()
        auto_train_iterations = self.auto_train_iterations_line.text()

        if not all([data_dir, output_dir, model_path, epochs, batch_size, learning_rate, embedding_size, resize_size,
                    auto_train_models, auto_train_iterations]):
            QMessageBox.warning(self, "Warning", "Please fill in all directory and training parameters.")
            return

        try:
            epochs = int(epochs)
            batch_size = int(batch_size)
            learning_rate = float(learning_rate)
            embedding_size = int(embedding_size)
            resize_size = int(resize_size)
            auto_train_models = int(auto_train_models)
            auto_train_iterations = int(auto_train_iterations)
            use_gpu = use_gpu == "True"
        except ValueError:
            QMessageBox.warning(self, "Error", "Invalid training parameters")
            return

        self.log_area.append("Starting training...")
        self.progress_bar.setValue(0)
        self.train_button.setEnabled(False)
        self.stop_button.setEnabled(True)

        self.training_worker = TrainingWorker(
            data_dir=data_dir,
            output_dir=output_dir,
            csv_path=self.csv_path,
            epochs=epochs,
            batch_size=batch_size,
            learning_rate=learning_rate,
            use_gpu=use_gpu,
            model_path=model_path,
            embedding_size=embedding_size,
            resize_size=resize_size,
            auto_train_models=auto_train_models,
            auto_train_iterations=auto_train_iterations
        )
        self.training_worker.progress_signal.connect(self.update_progress)
        self.training_worker.log_signal.connect(self.update_log)
        self.training_worker.plot_signal.connect(self.display_plot)
        self.training_worker.finished_signal.connect(self.training_finished)
        self.training_worker.start()
        self.save_config()

    def stop_training(self):
        if hasattr(self, 'training_worker') and self.training_worker.isRunning():
            self.training_worker.stop()
            self.log_area.append("Stopping training...")
            self.stop_button.setEnabled(False)

    def load_model_func(self):
        model_path = self.model_path_line.text()
        embedding_size = self.embedding_size_line.text()
        if not all([model_path, embedding_size]):
            QMessageBox.warning(self, "Warning", "Please specify the Model Path and Embedding Size before loading.")
            return
        try:
            embedding_size = int(embedding_size)
        except ValueError:
            QMessageBox.warning(self, "Error", "Invalid Embedding size")
            return

        try:
            model = load_model(model_path, embedding_size)
            self.log_area.append(f"Model loaded from {model_path}")
        except Exception as e:
            self.log_area.append(f"Error loading model: {e}")
            return

    def update_progress(self, value):
        self.progress_bar.setValue(value)

    def update_log(self, message):
        self.log_area.append(message)

    def display_plot(self, plot_path):
        try:
            plt.figure()
            img = plt.imread(plot_path)
            plt.imshow(img)
            plt.axis('off')
            plt.show(block=False)
        except Exception as e:
            self.log_area.append(f"Error displaying plot: {e}")

    def training_finished(self, message):
        self.log_area.append(message)
        self.train_button.setEnabled(True)
        self.stop_button.setEnabled(False)

    def save_config(self):
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
            'auto_train_iterations': self.auto_train_iterations_line.text()
        }

        with open('src/config.json', 'w') as f:
            json.dump(config, f)
        self.current_config = config

    def load_config(self):
        try:
            with open('src/config.json', 'r') as f:
                self.current_config = json.load(f)
        except FileNotFoundError:
            self.current_config = {}

    def closeEvent(self, event):
        self.save_config()
        event.accept()
