import os
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def build_transfer_learning_model(num_classes=2):
    # 1. تحميل نموذج ResNet18 مجهز بالأوزان المسبقة
    model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)

    # 2. تجميد أوزان الطبقات الأولى (Feature Extractor) لتسريع التدريب
    for param in model.parameters():
        param.requires_grad = False

    # 3. استبدال الطبقة الأخيرة (Fully Connected Layer) لتناسب عدد الفئات لدينا
    num_ftrs = model.fc.in_features
    model.fc = nn.Sequential(
        nn.Linear(num_ftrs, 128),
        nn.ReLU(),
        nn.Dropout(0.2),
        nn.Linear(128, num_classes)
    )
    return model.to(device)

if __name__ == '__main__':
    print("--- 1. بناء نموذج Transfer Learning بـ ResNet18 ---")
    model = build_transfer_learning_model(num_classes=2)
    print(model.fc)  # طباعة الطبقة الأخيرة المعدلة

    # 4. محاكاة تجهيز صورة عشوائية للتحقق من التمرير الأمامي (Forward Pass)
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    # إنشاء صورة تجريبية في الذاكرة
    dummy_img = Image.new('RGB', (300, 300), color='red')
    input_tensor = transform(dummy_img).unsqueeze(0).to(device) # إضافة بُعد الـ Batch

    model.eval()
    with torch.no_grad():
        output = model(input_tensor)
        print("\nمخرجات النموذج (Logits) للصورة التجريبية:", output.cpu().numpy())