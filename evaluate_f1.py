import torch
import torch.nn as nn
from torchvision import models, datasets, transforms
from sklearn.metrics import classification_report, confusion_matrix
import numpy as np

device = torch.device("cpu")

model = models.resnet34(weights=None)
model.fc = nn.Linear(512, 6)

state_dict = torch.load("pcb_resnet34(#2).pth", map_location=device)
model.load_state_dict(state_dict)
model.to(device)
model.eval()

test_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

test_dataset = datasets.ImageFolder(
    root="train",
    transform=test_transform
)

test_loader = torch.utils.data.DataLoader(
    test_dataset,
    batch_size=32,
    shuffle=False
)

class_names = test_dataset.classes
print("Class mapping:", test_dataset.class_to_idx)

y_true = []
y_pred = []

with torch.no_grad():
    for images, labels in test_loader:
        images = images.to(device)
        labels = labels.to(device)

        outputs = model(images)
        preds = torch.argmax(outputs, dim=1)

        y_true.extend(labels.cpu().numpy())
        y_pred.extend(preds.cpu().numpy())
        
top2_correct = 0
total = 0

with torch.no_grad():
    for images, labels in test_loader:
        outputs = model(images)
        top2 = torch.topk(outputs, k=2, dim=1).indices

        for i in range(labels.size(0)):
            if labels[i].item() in top2[i]:
                top2_correct += 1
            total += 1

print("Top-2 Accuracy:", top2_correct / total)


print("Total samples evaluated:", len(y_true))
print("Unique predicted labels:", np.unique(y_pred))

if len(y_true) == 0:
    print("ERROR: No samples found. Check dataset path.")
    exit()

print("\nClassification Report:\n")
print(classification_report(
    y_true,
    y_pred,
    target_names=class_names,
    digits=4,
    zero_division=0
))

print("\nConfusion Matrix:\n")
print(confusion_matrix(y_true, y_pred))
