import torch
import torch.nn as nn
import torch.optim as optim
from torchvision.models.detection import detr
from torchvision.transforms import functional as F
from torchvision.datasets import CocoDetection
from torchvision import transforms
from torch.utils.data import DataLoader

class AerialRescue(torch.utils.data.Dataset):
    def __init__(self, data_dir, transforms=None):
        self.data_dir = data_dir
        self.transforms = transforms
        self.image_paths = []  
        self.annotations = []  
        self.load_data()
    def load_data(self):
        annotation_file = os.path.join(self.data_dir, 'annotations.json')
        
        with open(annotation_file, 'r') as json_file:
            self.annotations = json.load(json_file)
        
        for annotation in self.annotations:
            image_path = os.path.join(self.data_dir, annotation['image_file'])
            self.image_paths.append(image_path)
    def __len__(self):
        return len(self.image_paths)
    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        image = self.load_image(image_path)
        if self.transforms:
            image = self.transforms(image)
        annotation = self.load_annotations(idx)
        return image, annotation
    def load_image(self, image_path):
        image = Image.open(image_path)
		
        return image
    def load_annotations(self, idx):
        annotation = self.annotations[idx]
        bbox = annotation['bbox']
        return annotation
def get_transform(train):
    transforms_list = []
    transforms_list.append(transforms.ToTensor())
    
    if train:
        transforms_list.append(transforms.RandomHorizontalFlip(p=0.5))
        
        transforms_list.append(transforms.RandomRotation(degrees=(-10, 10))
        transforms_list.append(transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1))
        
        
    return transforms.Compose(transforms_list)

train_dataset = AerialRescue(data_dir='path_to_train_data', transforms=get_transform(train=True))
val_dataset = AerialRescue(data_dir='path_to_val_data', transforms=get_transform(train=False))
train_loader = DataLoader(train_dataset, batch_size=2, shuffle=True, num_workers=4)
val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False, num_workers=4)
model = detr.DETR(num_classes=num_classes, pretrained=True)
criterion = nn.CrossEntropyLoss() 
optimizer = optim.AdamW(model.parameters(), lr=0.0001)
num_epochs = 10

for epoch in range(num_epochs):
    model.train()
    for images, targets in train_loader:
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs['pred_logits'], targets)
        loss.backward()
        optimizer.step()
    model.eval()
    with torch.no_grad():
        for val_images, val_targets in val_loader:
            val_outputs = model(val_images)
    print(f"Epoch [{epoch + 1}/{num_epochs}] - Loss: {loss.item()}")
    
torch.save(model.state_dict(), 'aerial_rescue_detection_model.pth')