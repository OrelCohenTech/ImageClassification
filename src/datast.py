import os
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset
from PIL import Image

class DualStreamDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        """
        root_dir: הנתיב לתיקיית הדאטה שמכילה את התיקיות: REAL, 2D, 3D
        """
        self.root_dir = root_dir
        self.transform = transform
        self.samples = []
        
        # --- עדכון: 3 מחלקות נפרדות (Multi-Class) ---
        # המודל יצטרך להוציא וקטור של 3 הסתברויות בסוף
        folder_mapping = {
            'REAL': 0,
            '2D': 1,
            '3D': 2
        }
        
        # המרה הפוכה (כדי שנוכל להדפיס שמות של מחלקות בבדיקה)
        self.idx_to_class = {v: k for k, v in folder_mapping.items()}
        
        print(f"Scanning data in: {os.path.abspath(root_dir)}")
        
        for folder_name, label in folder_mapping.items():
            folder_path = os.path.join(root_dir, folder_name)
            
            # בדיקה שהתיקייה אכן קיימת (לא רגיש לאותיות גדולות/קטנות בווינדוס, אבל בלינוקס כן)
            if not os.path.exists(folder_path):
                print(f" Warning: Folder '{folder_name}' not found in {root_dir}")
                continue
            
            # ספירת תמונות
            count = 0
            for img_name in os.listdir(folder_path):
                if img_name.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tif', '.webp')):
                    self.samples.append((os.path.join(folder_path, img_name), label))
                    count += 1
            
            print(f"Loaded {count} images from '{folder_name}' (Label: {label})")

    def __len__(self):
        return len(self.samples)

    def create_frequency_spectrum(self, img_path):
        """ יצירת תמונת תדרים (FFT) """
        # טעינה בשחור לבן
        img = cv2.imread(img_path, 0)
        
        if img is None:
            return Image.new('RGB', (224, 224))
            
        # המרת פורייה
        f = np.fft.fft2(img)
        fshift = np.fft.fftshift(f)
        
        magnitude_spectrum = 20 * np.log(np.abs(fshift) + 1e-9)
        magnitude_spectrum = cv2.normalize(magnitude_spectrum, None, 0, 255, cv2.NORM_MINMAX)
        magnitude_spectrum = np.uint8(magnitude_spectrum)
        
        # שכפול לערוצי RGB
        magnitude_spectrum = cv2.cvtColor(magnitude_spectrum, cv2.COLOR_GRAY2RGB)
        
        return Image.fromarray(magnitude_spectrum)

    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        
        try:
            image_rgb = Image.open(img_path).convert('RGB')
            image_freq = self.create_frequency_spectrum(img_path)
            
            if self.transform:
                image_rgb = self.transform(image_rgb)
                image_freq = self.transform(image_freq)
                
            return image_rgb, image_freq, label
            
        except Exception as e:
            print(f"Error loading image {img_path}: {e}")
            return torch.zeros(3, 224, 224), torch.zeros(3, 224, 224), label

# --- בדיקה עצמית ---
if __name__ == "__main__":
    # טריק למציאת הנתיב המדויק:
    # 1. קח את המיקום של הקובץ הזה (src)
    current_script_path = os.path.dirname(os.path.abspath(__file__))
    # 2. לך תיקייה אחת אחורה ותיכנס ל-data
    data_path = os.path.join(current_script_path, '..', 'data')
    
    print(f" Looking for data in: {os.path.abspath(data_path)}")
    
    # בדיקה האם התיקייה הראשית קיימת
    if not os.path.exists(data_path):
        print(f" Error: The folder '{data_path}' does not exist!")
    else:
        # יצירת ה-Dataset עם הנתיב המוחלט
        ds = DualStreamDataset(root_dir=data_path) 
        
        if len(ds) > 0:
            print(f"\n Success! Total images: {len(ds)}")
            
            # בדיקת דוגמה אחת כדי לראות שהכל תקין
            try:
                rgb, freq, lbl = ds[0]
                class_name = ds.idx_to_class[lbl]
                print(f"Sample loaded -> Label: {lbl} ({class_name})")
                print(f"RGB Shape: {rgb.size if not isinstance(rgb, torch.Tensor) else rgb.shape}")
            except Exception as e:
                print(f"Error loading sample: {e}")
        else:
            print("\n Error: No images found inside the folders.")
            print("Please check: inside 'data' -> do you have 'REAL', '2D', '3D' folders?")
