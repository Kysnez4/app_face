import torch
import torch.nn as nn
import torchvision.models as models
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import os
import torch.optim as optim


class FaceRecognitionModel(nn.Module):
    def __init__(self, num_classes, embedding_size=128, pretrained=True):
        super(FaceRecognitionModel, self).__init__()

        self.efficientnet = models.efficientnet_b7(pretrained=pretrained)

        # Freeze parameters (optional)
        for param in self.efficientnet.parameters():
            param.requires_grad = False

        #  Fix to access in_features correctly
        in_features = self.efficientnet.classifier._modules['1'].in_features # for newer pytorch

        self.efficientnet.classifier = nn.Sequential(
            nn.Linear(in_features, embedding_size),
            nn.ReLU(),
            nn.Linear(embedding_size, num_classes)
        )

    def forward(self, x):
        return self.efficientnet(x)


class FaceDataset(Dataset):
    def __init__(self, data_dir, transform=None):
        self.data_dir = data_dir
        self.image_paths = []
        self.labels = []
        self.transform = transform
        self.class_names = sorted(os.listdir(data_dir))  # Get class names

        for i, person_name in enumerate(self.class_names):  # Iterate through class names instead of os.listdir(data_dir)
            person_dir = os.path.join(data_dir, person_name)
            if os.path.isdir(person_dir):
                for image_name in os.listdir(person_dir):
                    image_path = os.path.join(person_dir, image_name)
                    self.image_paths.append(image_path)
                    self.labels.append(i)

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        label = self.labels[idx]

        image = Image.open(image_path).convert('RGB')

        if self.transform:
            image = self.transform(image)

        return image, torch.tensor(label, dtype=torch.long)


def train_model(model, data_dir, device, epochs=10, img_height=224, img_width=224, batch_size=32, callback=None):
    data_transforms = transforms.Compose([
        transforms.Resize((img_height, img_width)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    dataset = FaceDataset(data_dir, transform=data_transforms)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    model = model.to(device)

    for epoch in range(epochs):
        running_loss = 0.0
        for i, data in enumerate(dataloader, 0):
            inputs, labels = data[0].to(device), data[1].to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            if callback:
                callback(epoch, i, loss.item()) # call training callback
            if i % 10 == 9:
                print(f'[{epoch + 1}, {i + 1:5d}] loss: {running_loss / 10:.3f}')
                running_loss = 0.0

    print('Finished Training')
    return model

def save_model(model, model_path):
  torch.save(model.state_dict(), model_path)
  print(f"Модель сохранена в {model_path}")

def load_model(num_classes, embedding_size, model_path, device):
    model = FaceRecognitionModel(num_classes, embedding_size)
    try:
        model.load_state_dict(torch.load(model_path, map_location=device))
        model.eval()
        model = model.to(device)
        print("Модель загружена!")
    except:
        print("Не удалось загрузить модель.  Используется необученная модель.")
    return model

def load_data(data_dir, transform, batch_size):
    dataset = datasets.ImageFolder(data_dir, transform=transform)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    return dataloader, dataset.class_to_idx

# Training Function (moved outside for clarity)
def train_model(model, data_dir, device, epochs=5, batch_size=32, callback=None):  # Added batch_size
    """Trains the face recognition model."""
    try:
        # Define transformations
        data_transforms = transforms.Compose([
            transforms.Resize((IMG_SIZE, IMG_SIZE)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])

        # Load data
        dataloader, class_to_idx = load_data(data_dir, data_transforms, batch_size)
        num_classes = len(class_to_idx)

        # Loss function and optimizer
        criterion = torch.nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)  # Adjust learning rate as needed

        model.train()  # Set model to training mode

        total_batches = len(dataloader)

        for epoch in range(epochs):
            for batch_idx, (inputs, labels) in enumerate(dataloader):
                inputs = inputs.to(device)
                labels = labels.to(device)

                optimizer.zero_grad()  # Zero the parameter gradients

                outputs = model(inputs)  # Forward pass
                loss = criterion(outputs, labels)  # Calculate loss

                loss.backward()  # Backward pass
                optimizer.step()  # Optimize

                if callback:
                    callback(epoch, batch_idx, loss.item(), total_batches)

        return model
    except Exception as e:
        st.error(f"Ошибка во время обучения: {e}")
        return None
