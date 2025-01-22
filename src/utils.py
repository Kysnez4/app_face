import os
import random
import pandas as pd
from PIL import Image
import torch
import torch.nn as nn
from efficientnet_pytorch import EfficientNet
import cv2
from copy import deepcopy


def create_pairs(data_dir):
    """Создает положительные пары изображений."""
    positive_pairs = []
    labels = []
    for student_dir in os.listdir(data_dir):
        student_path = os.path.join(data_dir, student_dir)
        if os.path.isdir(student_path):
            images = [os.path.join(student_path, img)
                      for img in os.listdir(student_path) if img.lower().endswith(('.png', '.jpg', '.jpeg'))]
            if len(images) < 2:
                print(f"Warning: Student {student_dir} has less than 2 images.")
                continue  # Skip to the next folder if there are not enough images

            # Creating positive pairs for a single student with shuffling
            pairs = list(zip(images[::2], images[1::2]))
            positive_pairs.extend(pairs)
            labels.extend([1] * len(pairs))

    return positive_pairs, labels


def create_negative_pairs(data_dir, num_negative, positive_pairs):
    """Создает отрицательные пары изображений."""
    all_images = []
    for student_dir in os.listdir(data_dir):
        student_path = os.path.join(data_dir, student_dir)
        if os.path.isdir(student_path):
            images = [os.path.join(student_path, img)
                      for img in os.listdir(student_path) if img.lower().endswith(('.png', '.jpg', '.jpeg'))]
            all_images.extend(images)

    negative_pairs = []
    labels = []
    existing_pairs = set()
    for pair in positive_pairs:
        existing_pairs.add(tuple(sorted(pair)))

    for _ in range(num_negative):
        while True:
            img1 = random.choice(all_images)
            img2 = random.choice(all_images)
            if img1 != img2:
                sorted_pair = tuple(sorted((img1, img2)))
                if sorted_pair not in existing_pairs:
                    negative_pairs.append((img1, img2))
                    labels.append(0)
                    break

    return negative_pairs, labels


def create_dataset_csv(data_dir, output_dir, num_negative):
    """Создает CSV файл с парами изображений и метками."""
    positive_pairs, positive_labels = create_pairs(os.path.join(data_dir, 'students'))
    negative_pairs, negative_labels = create_negative_pairs(os.path.join(data_dir, 'students'),
                                                            num_negative, positive_pairs)

    all_pairs = positive_pairs + negative_pairs
    all_labels = positive_labels + negative_labels

    df = pd.DataFrame({'image1': [p[0] for p in all_pairs],
                       'image2': [p[1] for p in all_pairs],
                       'label': all_labels})
    csv_path = os.path.join(output_dir, 'pairs.csv')
    df.to_csv(csv_path, index=False)
    return csv_path


def load_image(image_path, transform=None, resize_size=None):
    """Загружает и преобразует изображение."""
    try:
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # Convert to RGB
        if resize_size:
            image = cv2.resize(image, (resize_size, resize_size))
        image = Image.fromarray(image)  # Convert back to PIL
        if transform:
            image = transform(image)
        return image
    except Exception as e:
        print(f"Error loading image: {image_path}, Error: {e}")
        return None


class FaceCompareModel(nn.Module):
    def __init__(self, embedding_size=128):
        super(FaceCompareModel, self).__init__()
        self.efficientnet = EfficientNet.from_pretrained('efficientnet-b0')
        self.efficientnet._fc = nn.Linear(self.efficientnet._fc.in_features, embedding_size)
        self.embedding_size = embedding_size

    def forward(self, x):
        embedding = self.efficientnet(x)
        return embedding


def contrastive_loss(output1, output2, label, margin=1.0):
    euclidean_distance = torch.nn.functional.pairwise_distance(output1, output2)
    loss_contrastive = torch.mean((label) * torch.pow(euclidean_distance, 2) +
                                  (1 - label) * torch.pow(torch.clamp(margin - euclidean_distance, min=0.0), 2))
    return loss_contrastive


def save_model(model, model_path):
    torch.save(model.state_dict(), model_path)


def load_model(model_path, embedding_size=128):
    model = FaceCompareModel(embedding_size)
    model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
    model.eval()
    return model


def copy_model(model):
    return deepcopy(model)
