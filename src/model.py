import torch
import torch.nn as nn
import torchvision.models as models

class FakeDetectDualNet(nn.Module):
    def __init__(self, num_classes=3):
        super(FakeDetectDualNet, self).__init__()
        
        print("🏗️ Building Dual-Stream Model...")
        
        # --- ענף א': RGB (וויזואלי) - EfficientNet ---
        # מודל חזק מאוד שמצליח לזהות פרטים עדינים
        self.rgb_branch = models.efficientnet_b0(weights='IMAGENET1K_V1')
        
        # אנחנו מסירים את השכבה האחרונה (הסיווג) כדי לקבל רק את הפיצ'רים
        # ב-EfficientNet B0 גודל הפיצ'רים הוא 1280
        self.rgb_feature_dim = self.rgb_branch.classifier[1].in_features
        # מחליפים את הראש ב-Identity (כלומר, לא עושה כלום, רק מעביר את המידע)
        self.rgb_branch.classifier = nn.Identity()
        
        # --- ענף ב': תדרים (Frequency) - ResNet18 ---
        # מודל קליל ומהיר לזיהוי תבניות גיאומטריות בספקטרום
        self.freq_branch = models.resnet18(weights='IMAGENET1K_V1')
        
        # ב-ResNet18 גודל הפיצ'רים לפני הסוף הוא 512
        self.freq_feature_dim = self.freq_branch.fc.in_features
        self.freq_branch.fc = nn.Identity()
        
        # --- המוח המשלב (Fusion Head) ---
        # חיבור הגדלים של שני הענפים (1280 + 512 = 1792)
        combined_features = self.rgb_feature_dim + self.freq_feature_dim
        
        self.classifier = nn.Sequential(
            nn.Linear(combined_features, 512),
            nn.BatchNorm1d(512), # ייצוב האימון
            nn.ReLU(),
            nn.Dropout(0.3),     # מניעת Overfitting
            nn.Linear(512, 128),
            nn.ReLU(),
            nn.Linear(128, num_classes) # יציאה ל-3 מחלקות (0, 1, 2)
        )

    def forward(self, x_rgb, x_freq):
        # 1. הרצת התמונה הרגילה בענף הראשון
        rgb_features = self.rgb_branch(x_rgb)
        
        # 2. הרצת תמונת התדרים בענף השני
        freq_features = self.freq_branch(x_freq)
        
        # 3. איחוד המידע (Concatenation)
        combined = torch.cat((rgb_features, freq_features), dim=1)
        
        # 4. קבלת החלטה סופית
        output = self.classifier(combined)
        
        return output

# --- בדיקה עצמית (לוודא שהמודל נבנה בלי שגיאות) ---
if __name__ == "__main__":
    # יצירת דאטה פיקטיבי - שים לב: שינינו את המספר הראשון ל-2 (Batch Size)
    # זה קריטי כי BatchNorm לא עובד עם דוגמה אחת בלבד
    model = FakeDetectDualNet(num_classes=3)
    dummy_rgb = torch.randn(2, 3, 224, 224) 
    dummy_freq = torch.randn(2, 3, 224, 224)
    
    print("Testing forward pass...")
    output = model(dummy_rgb, dummy_freq)
    print(f"\n Model Output Shape: {output.shape}")
    
    # הפלט צריך להיות [2, 3] (2 דוגמאות * 3 מחלקות)
    if output.shape == (2, 3):
        print("Status: Model is ready for training! ")
    else:
        print("Status: Something is wrong with dimensions.")
