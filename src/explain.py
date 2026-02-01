import torch
import torch.nn.functional as F
import cv2
import numpy as np
import matplotlib.pyplot as plt
from model import FakeDetectDualNet
from interface import prepare_image, CLASSES, MODEL_PATH, DEVICE
import os

# --- מחלקה לחישוב Grad-CAM (בלי ספריות חיצוניות) ---
class GradCAM:
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None
        
        # הרשמה לאירועים (Hooks) כדי לתפוס את המידע באמצע הרשת
        target_layer.register_forward_hook(self.save_activation)
        target_layer.register_full_backward_hook(self.save_gradient)

    def save_activation(self, module, input, output):
        self.activations = output

    def save_gradient(self, module, grad_input, grad_output):
        self.gradients = grad_output[0]

    def __call__(self, rgb_input, freq_input, class_idx):
        # 1. הרצה קדימה
        output = self.model(rgb_input, freq_input)
        
        # 2. איפוס גרדיאנטים
        self.model.zero_grad()
        
        # 3. הרצה אחורה (Backward) רק עבור המחלקה שמעניינת אותנו
        target = output[0][class_idx]
        target.backward()
        
        # 4. חישוב מפת החום (הקסם של Grad-CAM)
        gradients = self.gradients.data.cpu().numpy()[0]
        activations = self.activations.data.cpu().numpy()[0]
        
        # ממוצע של הגרדיאנטים (Global Average Pooling)
        weights = np.mean(gradients, axis=(1, 2))
        
        # הכפלת האקטיבציות במשקולות
        cam = np.zeros(activations.shape[1:], dtype=np.float32)
        for i, w in enumerate(weights):
            cam += w * activations[i]
            
        # ניקוי רעשים (ReLU)
        cam = np.maximum(cam, 0)
        
        # נרמול לטווח 0-1
        cam = cv2.resize(cam, (224, 224))
        cam = cam - np.min(cam)
        cam = cam / np.max(cam)
        return cam

def visualize_prediction(image_path):
    print(f" Generating Heatmap for: {image_path}...")
    
    # 1. טעינת מודל
    model = FakeDetectDualNet(num_classes=3).to(DEVICE)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE, weights_only=True))
    model.eval()
    
    # 2. הכנת התמונה
    rgb_input, freq_input = prepare_image(image_path)
    if rgb_input is None: return

    # 3. הגדרת היעד ל-Grad-CAM
    # אנחנו רוצים להסתכל על השכבה האחרונה של ה-EfficientNet (ענף ה-RGB)
    # שם המודל "רואה" את הצורות והטקסטורות
    target_layer = model.rgb_branch.features[-1]
    grad_cam = GradCAM(model, target_layer)
    
    # 4. ביצוע החיזוי כדי לדעת איזו מחלקה להסביר
    output = model(rgb_input, freq_input)
    pred_idx = torch.argmax(output, dim=1).item()
    pred_class = CLASSES[pred_idx]
    
    print(f"Prediction: {pred_class}")
    
    # 5. הפעלת Grad-CAM
    heatmap = grad_cam(rgb_input, freq_input, pred_idx)
    
    # 6. יצירת תמונה משולבת
    # טעינת התמונה המקורית לצורך תצוגה
    original_img = cv2.imread(image_path)
    original_img = cv2.resize(original_img, (224, 224))
    original_img = np.float32(original_img) / 255
    
    # המרת מפת החום לצבעים (JET colormap)
    heatmap_colored = cv2.applyColorMap(np.uint8(255 * heatmap), cv2.COLORMAP_JET)
    heatmap_colored = np.float32(heatmap_colored) / 255
    
    # שילוב (Overlay)
    cam_result = heatmap_colored + original_img
    cam_result = cam_result / np.max(cam_result)
    
    # 7. שמירת התוצאה
    output_filename = "gradcam_result3.jpg"
    cv2.imwrite(output_filename, np.uint8(255 * cam_result))
    print(f" Saved result to: {output_filename}")
    
    # אופציונלי: פתיחת התמונה מיד (רק בווינדוס)
    try:
        os.startfile(output_filename)
    except:
        pass

if __name__ == "__main__":
    # שימו כאן את שם התמונה שבדקתם קודם
    img_name = "test_image3.jpg"
    
    if os.path.exists(img_name):
        visualize_prediction(img_name)
    else:
        print(f" Image '{img_name}' not found.")
