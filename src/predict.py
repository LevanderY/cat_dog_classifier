import torch
import torchvision.transforms as transforms
from PIL import Image
from models.cnn_model import CatDogCNN

def predict_image(image_path, model_path='../saved_models/best_model.pth'):
    transform = transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    image = Image.open(image_path)
    image = transform(image).unsqueeze(0)
    
    model = CatDogCNN()
    model.load_state_dict(torch.load(model_path))
    model.eval()
    
    with torch.no_grad():
        outputs = model(image)
        _, predicted = torch.max(outputs.data, 1)
        
    return "It is a cat!" if predicted.item() == 0 else "It is a dog!"

if __name__ == "__main__":
    image_path = "../assets/dog_1.jpg"
    result = predict_image(image_path)
    print(result) 