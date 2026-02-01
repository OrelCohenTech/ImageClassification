import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report
import numpy as np
import os
import sys

# ייבוא המחלקות שלנו
from datast import DualStreamDataset
from model import FakeDetectDualNet

# --- הגדרות ---
MODEL_PATH = "fake_image_classifier.pth"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BATCH_SIZE = 16

# שמות המחלקות לפי הסדר (0, 1, 2)
class_names = ['REAL', '2D', '3D']

def evaluate_model():
    print(f" Starting Evaluation on {DEVICE}...")

    # 1. טעינת הדאטה (בדיוק כמו באימון)
    current_script_path = os.path.dirname(os.path.abspath(__file__))
    data_path = os.path.join(current_script_path, '..', 'data')

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ])

    print("Loading dataset...")
    # כאן אנחנו טוענים את כל הדאטה. בפרויקט אמיתי עדיף לטעון רק את ה-Test Set
    # אבל לצורך הדגמה נריץ על הכל כדי לראות את המטריצה המלאה
    dataset = DualStreamDataset(root_dir=data_path, transform=transform)
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

    # 2. טעינת המודל
    model = FakeDetectDualNet(num_classes=3).to(DEVICE)
    if os.path.exists(MODEL_PATH):
        model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE, weights_only=True))
        print(" Model loaded successfully.")
    else:
        print(f" Error: Model file '{MODEL_PATH}' not found.")
        return

    model.eval() # מצב בדיקה

    all_preds = []
    all_labels = []

    print("Running inference on all images (this might take a minute)...")
    
    # 3. מעבר על כל התמונות
    with torch.no_grad():
        for rgb_imgs, freq_imgs, labels in loader:
            rgb_imgs, freq_imgs = rgb_imgs.to(DEVICE), freq_imgs.to(DEVICE)
            
            outputs = model(rgb_imgs, freq_imgs)
            _, preds = torch.max(outputs, 1)
            
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.numpy())

    # 4. חישוב מדדים
    print("\n" + "="*40)
    print(" Classification Report")
    print("="*40)
    print(classification_report(all_labels, all_preds, target_names=class_names))

    # 5. ציור Confusion Matrix
    cm = confusion_matrix(all_labels, all_preds)
    
    plt.figure(figsize=(10, 8))
    # יצירת מפת חום (Heatmap)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=class_names, yticklabels=class_names)
    
    plt.xlabel('Predicted Label (מה המודל חשב)')
    plt.ylabel('True Label (מה זה באמת)')
    plt.title('Confusion Matrix - DeepFake Detection')
    
    # שמירת הגרף
    plt.savefig('confusion_matrix.png')
    print(" Confusion Matrix saved as 'confusion_matrix.png'")
    
    # הצגת הגרף על המסך
    plt.show()

if __name__ == "__main__":
    evaluate_model()
