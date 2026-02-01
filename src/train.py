import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from torchvision import transforms
import os

# ייבוא מהקבצים שלכם
from datast import DualStreamDataset
from model import FakeDetectDualNet

# --- הגדרות (Hyperparameters) ---
BATCH_SIZE = 16        # כמה תמונות מעבדים במכה
LEARNING_RATE = 0.0001 # קצב למידה (עדיף נמוך ל-Fine Tuning)
NUM_EPOCHS = 10        # כמה פעמים עוברים על כל הדאטה
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def train_model():
    print(f" Starting training on device: {DEVICE}")

    # 1. מציאת נתיב הדאטה באופן דינמי
    current_script_path = os.path.dirname(os.path.abspath(__file__))
    data_path = os.path.join(current_script_path, '..', 'data')

    # 2. הכנת הנתונים
    # שינוי גודל ל-224x224 (דרישה של EfficientNet) והמרה לטנזור
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ])

    print("Loading Dataset...")
    full_dataset = DualStreamDataset(root_dir=data_path, transform=transform)
    
    # חלוקה ל-Train (80%) ו-Validation (20%)
    train_size = int(0.8 * len(full_dataset))
    val_size = len(full_dataset) - train_size
    train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])

    # יצירת ה-Loaders
    # num_workers=0 חשוב בווינדוס כדי למנוע תקלות
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

    print(f"Data loaded: {len(train_dataset)} training samples, {len(val_dataset)} validation samples.")

    # 3. טעינת המודל (3 מחלקות: REAL, 2D, 3D)
    model = FakeDetectDualNet(num_classes=3).to(DEVICE)

    # 4. פונקציית הפסד ואופטימייזר
    criterion = nn.CrossEntropyLoss() # מתאים לסיווג רב-מחלקתי
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    # 5. לולאת האימון
    for epoch in range(NUM_EPOCHS):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        print(f"\nEpoch {epoch+1}/{NUM_EPOCHS}")
        print("-" * 30)

        for batch_idx, (rgb_imgs, freq_imgs, labels) in enumerate(train_loader):
            # העברה ל-GPU/CPU
            rgb_imgs, freq_imgs, labels = rgb_imgs.to(DEVICE), freq_imgs.to(DEVICE), labels.to(DEVICE)

            # איפוס נגזרות
            optimizer.zero_grad()

            # Forward
            outputs = model(rgb_imgs, freq_imgs)
            loss = criterion(outputs, labels)

            # Backward
            loss.backward()
            optimizer.step()

            # סטטיסטיקות
            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1) # מציאת המחלקה עם הציון הגבוה ביותר
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

            if (batch_idx + 1) % 5 == 0:
                print(f"Batch {batch_idx+1}/{len(train_loader)}: Loss = {loss.item():.4f}")

        train_acc = 100 * correct / total
        train_loss = running_loss / len(train_loader)
        print(f"End of Epoch {epoch+1} -> Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%")

        # ולידציה בסוף כל Epoch
        validate(model, val_loader, criterion)

    # שמירת המודל
    print("\nSaving model...")
    torch.save(model.state_dict(), "fake_image_classifier.pth")
    print(" Model saved as 'fake_image_classifier.pth'")

def validate(model, loader, criterion):
    model.eval()
    val_loss = 0.0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for rgb_imgs, freq_imgs, labels in loader:
            rgb_imgs, freq_imgs, labels = rgb_imgs.to(DEVICE), freq_imgs.to(DEVICE), labels.to(DEVICE)
            
            outputs = model(rgb_imgs, freq_imgs)
            loss = criterion(outputs, labels)
            
            val_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
    val_acc = 100 * correct / total
    val_loss = val_loss / len(loader)
    print(f"Validation -> Loss: {val_loss:.4f}, Accuracy: {val_acc:.2f}%")

if __name__ == "__main__":
    train_model()
