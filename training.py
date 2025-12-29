import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image

device = torch.device("cpu")

model = models.resnet34(weights=None)
model.fc = nn.Linear(512, 6)
model.to(device)

state_dict = torch.load("pcb_resnet34(#2).pth", map_location=device)
model.load_state_dict(state_dict)
model.eval()

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],   
        std=[0.229, 0.224, 0.225]
    )
])

image = Image.open("12_spurious_copper_05.jpg").convert("RGB")
input_tensor = transform(image).unsqueeze(0)  
input_tensor = input_tensor.to(device)

with torch.no_grad():
    outputs = model(input_tensor)
    probabilities = torch.softmax(outputs, dim=1)
    predicted_class = torch.argmax(probabilities, dim=1).item()

print("Predicted class index:", predicted_class)
print("Class probabilities:", probabilities.squeeze().tolist())
