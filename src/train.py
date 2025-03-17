import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from torchvision import datasets
from torch.utils.data import DataLoader, random_split
import os
from datetime import datetime
from PIL import Image
from models.cnn_model import CatDogCNN

def verify_images(directory):
    corrupted = []
    total_images = 0
    print(f"\nVerifying images in {directory}...")
    for root, dirs, files in os.walk(directory):
        for filename in files:
            if filename.endswith(('.jpg', '.jpeg', '.png')):
                total_images += 1
                filepath = os.path.join(root, filename)
                try:
                    img = Image.open(filepath)
                    img.verify()
                except Exception as e:
                    print(f"Corrupted image found: {filepath}")
                    print(f"Error: {str(e)}")
                    corrupted.append(filepath)
                    os.remove(filepath)
    print(f"Verification complete: {total_images} images checked")
    print(f"{len(corrupted)} corrupted images removed")
    return len(corrupted)

class RobustImageFolder(datasets.ImageFolder):
    def __getitem__(self, index):
        try:
            return super().__getitem__(index)
        except Exception as e:
            print(f"Error loading image at index {index}: {str(e)}")
            return self.__getitem__((index + 1) % len(self))

def main():
    print("\n" + "="*50)
    print("Starting Cat-Dog Classifier Training")
    print("="*50 + "\n")
    
    start_time = datetime.now()
    print(f"Training started at: {start_time.strftime('%Y-%m-%d %H:%M:%S')}")
    
    verify_images("../data/PetImages")
    
    transform = transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    print("\nLoading dataset...")
    dataset = RobustImageFolder(root="../data/PetImages", transform=transform)
    print(f"Dataset loaded: {len(dataset)} total images")
    print(f"Classes: {dataset.classes}")
    print(f"Class to index mapping: {dataset.class_to_idx}")

    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
    print(f"\nDataset split:")
    print(f"Training set: {train_size} images")
    print(f"Validation set: {val_size} images")

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

    print("\nInitializing model...")
    model = CatDogCNN()
    print("Model architecture:")
    print(model)
    print(f"Total parameters: {sum(p.numel() for p in model.parameters()):,}")

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    print(f"\nTraining configuration:")
    print(f"Optimizer: {optimizer.__class__.__name__}")
    print(f"Learning rate: {optimizer.param_groups[0]['lr']}")
    print(f"Loss function: {criterion.__class__.__name__}")

    num_epochs = 10
    best_val_acc = 0.0

    os.makedirs('saved_models', exist_ok=True)
    print("\nStarting training...")
    print("="*50)

    for epoch in range(num_epochs):
        epoch_start_time = datetime.now()
        print(f"\nEpoch [{epoch+1}/{num_epochs}]")
        print("-"*50)
        
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0
        batch_count = 0
        
        for images, labels in train_loader:
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            train_total += labels.size(0)
            train_correct += (predicted == labels).sum().item()
            batch_count += 1
            
            if batch_count % 10 == 0:
                current_acc = 100 * train_correct / train_total
                print(f"Batch [{batch_count}/{len(train_loader)}] - Loss: {loss.item():.4f} - Acc: {current_acc:.2f}%")
        
        train_acc = 100 * train_correct / train_total
        avg_train_loss = train_loss / len(train_loader)
        
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        
        print("\nValidating...")
        with torch.no_grad():
            for images, labels in val_loader:
                outputs = model(images)
                loss = criterion(outputs, labels)
                
                val_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                val_total += labels.size(0)
                val_correct += (predicted == labels).sum().item()
        
        val_acc = 100 * val_correct / val_total
        avg_val_loss = val_loss / len(val_loader)
        
        epoch_duration = datetime.now() - epoch_start_time
        print(f"\nEpoch Summary:")
        print(f"Training Loss: {avg_train_loss:.4f}")
        print(f"Training Accuracy: {train_acc:.2f}%")
        print(f"Validation Loss: {avg_val_loss:.4f}")
        print(f"Validation Accuracy: {val_acc:.2f}%")
        print(f"Duration: {epoch_duration.total_seconds():.1f}s")
        
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), 'saved_models/best_model.pth')
            print(f"\nNew best model saved!")
            print(f"Validation accuracy: {val_acc:.2f}%")
            print(f"Model saved to: saved_models/best_model.pth")

    total_duration = datetime.now() - start_time
    print("\n" + "="*50)
    print("Training completed!")
    print(f"Finished at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Total duration: {total_duration.total_seconds()/60:.1f} minutes")
    print(f"Best validation accuracy: {best_val_acc:.2f}%")
    print("="*50 + "\n")

if __name__ == "__main__":
    main() 