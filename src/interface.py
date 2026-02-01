import torch
import cv2
import numpy as np
from PIL import Image
from torchvision import transforms
import os

# ייבוא המודל והפונקציה של התדרים שבנינו
from model import FakeDetectDualNet
from datast import DualStreamDataset 

# --- הגדרות ---
MODEL_PATH = "fake_image_classifier.pth" # ודאו שהקובץ הזה קיים בתיקייה הראשית
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# מיפוי המחלקות (חייב להיות זהה למה שהגדרנו ב-Dataset)
CLASSES = {0: 'REAL (Authentic)', 1: '2D (Artificial)', 2: '3D (Artificial)'}

def load_trained_model():
    print(f"Loading model from {MODEL_PATH}...")
    model = FakeDetectDualNet(num_classes=3)
    # טעינת המשקולות (map_location=DEVICE חשוב אם מאמנים ב-GPU ובודקים ב-CPU)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE, weights_only=True))
    model.to(DEVICE)
    model.eval() # מעבר למצב חיזוי (מכבה Dropout וכו')
    return model

def prepare_image(image_path):
    """ הכנת התמונה בדיוק כמו באימון (Resize + FFT) """
    # 1. טרנספורמציות בסיסיות
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ])
    
    # 2. טעינת RGB
    try:
        rgb_img = Image.open(image_path).convert('RGB')
    except Exception as e:
        print(f"Error opening image: {e}")
        return None, None

    # 3. יצירת FFT (משתמשים בלוגיקה מ-Dataset אבל באופן ידני כאן לפשטות)
    # נשתמש בפונקציית עזר קטנה כדי לא ליצור אובייקט Dataset שלם
    freq_img_pil = create_fft_single(image_path)

    # החלת הטרנספורמציות והוספת מימד Batch (מ-[3, 224, 224] ל-[1, 3, 224, 224])
    rgb_tensor = transform(rgb_img).unsqueeze(0).to(DEVICE)
    freq_tensor = transform(freq_img_pil).unsqueeze(0).to(DEVICE)
    
    return rgb_tensor, freq_tensor

def create_fft_single(img_path):
    """ אותה לוגיקה בדיוק כמו ב-dataset.py """
    img = cv2.imread(img_path, 0)
    if img is None: return Image.new('RGB', (224, 224))
    f = np.fft.fft2(img)
    fshift = np.fft.fftshift(f)
    magnitude_spectrum = 20 * np.log(np.abs(fshift) + 1e-9)
    magnitude_spectrum = cv2.normalize(magnitude_spectrum, None, 0, 255, cv2.NORM_MINMAX)
    magnitude_spectrum = np.uint8(magnitude_spectrum)
    magnitude_spectrum = cv2.cvtColor(magnitude_spectrum, cv2.COLOR_GRAY2RGB)
    return Image.fromarray(magnitude_spectrum)

def predict(image_path):
    # טעינת מודל
    model = load_trained_model()
    
    # הכנת תמונה
    rgb_input, freq_input = prepare_image(image_path)
    if rgb_input is None: return

    print(f"\n Analyzing image: {image_path}")
    
    with torch.no_grad():
        # הרצת המודל
        outputs = model(rgb_input, freq_input)
        
        # חישוב הסתברויות (Softmax)
        probabilities = torch.nn.functional.softmax(outputs, dim=1)
        
        # קבלת ההחלטה הסופית
        score, predicted_idx = torch.max(probabilities, 1)
        predicted_class = CLASSES[predicted_idx.item()]
        confidence = score.item() * 100

    print("-" * 30)
    print(f" Result: {predicted_class}")
    print(f" Confidence: {confidence:.2f}%")
    
    # הדפסת כל ההסתברויות
    print("\nDetailed Probabilities:")
    for i, class_name in CLASSES.items():
        prob = probabilities[0][i].item() * 100
        print(f"  {class_name}: {prob:.2f}%")
    print("-" * 30)

if __name__ == "__main__":
    # --- כאן אתם בוחרים איזו תמונה לבדוק ---
    # שלב א': תורידו תמונה מהאינטרנט ותשמרו אותה בתיקיית הפרויקט
    # שלב ב': שנו את השם כאן לשם הקובץ שלכם
    image_to_test = "test_image.jpg" 
    
    # בדיקה אם הקובץ קיים לפני שמריצים
    if os.path.exists(image_to_test):
        predict(image_to_test)
    else:
        print(f" Please create/download an image named '{image_to_test}' in the project folder to test.")
