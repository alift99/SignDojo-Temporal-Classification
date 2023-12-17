import torch
from train_trail_classifier import get_trail_classifier
from dataloader import TrailImageDataset
from torch.utils.data import DataLoader


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = get_trail_classifier().to(device)

test_dataset = TrailImageDataset(root='images/test')
test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=0)

# Eval loop
model.eval()
correct = 0
total = 0

with torch.no_grad():
    for inputs, labels in test_loader:
        inputs, labels = inputs.to(device), labels.to(device)

        outputs = model(inputs)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

accuracy = correct / total
print(f"Test Accuracy: {accuracy * 100:.2f}%")