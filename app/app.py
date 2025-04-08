import os
import io
import sys
import base64
import torch
import numpy as np
from PIL import Image, ImageDraw
import cv2
from flask import Flask, request, jsonify, render_template, send_from_directory

app = Flask(__name__, static_folder='static')

UPLOAD_FOLDER = 'uploads'
RESULT_FOLDER = 'static/results'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(RESULT_FOLDER, exist_ok=True)

parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(parent_dir)

MODEL_PATH = os.path.join(parent_dir, 'yolov5/runs/train/exp10/weights/best.pt')

model = None

def load_model():
    global model
    try:
        model = torch.hub.load('ultralytics/yolov5', 'custom', path=MODEL_PATH)
        model.conf = 0.4  
        model.iou = 0.45 
        print(f"Model loaded successfully! Classes: {model.names}")
        return model
    except Exception as e:
        print(f"Error loading model: {e}")
        import traceback
        traceback.print_exc()
        return None

COLORS = {
    0: (255, 0, 0),    
    1: (0, 0, 255)     
}

CLASS_NAMES = {
    0: "red",
    1: "blue"
}

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(UPLOAD_FOLDER, filename)

@app.route('/detect', methods=['POST'])
def detect():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'})
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'})
    
    filename = os.path.join(UPLOAD_FOLDER, file.filename)
    file.save(filename)
    
    try:
        img = Image.open(filename)
        img_np = np.array(img)
        
        if model is None:
            load_model()
            if model is None:
                return jsonify({'error': 'Failed to load the model'})
        
        results = model(img_np)
        
        result_img = img.copy()
        draw = ImageDraw.Draw(result_img)
        
        plates = []
        detections = results.pandas().xyxy[0]
        
        for i, detection in detections.iterrows():
            x1, y1, x2, y2 = int(detection['xmin']), int(detection['ymin']), int(detection['xmax']), int(detection['ymax'])
            conf = float(detection['confidence'])
            cls = int(detection['class'])
            
            color_rgb = COLORS.get(cls, (0, 255, 0)) 
            color = (color_rgb[0], color_rgb[1], color_rgb[2])  
            class_name = CLASS_NAMES.get(cls, f"class_{cls}")
            
            draw.rectangle([(x1, y1), (x2, y2)], outline=color, width=3)
            
            plates.append({
                'color': class_name,
                'confidence': conf,
                'box': [x1, y1, x2, y2]
            })
        
        result_filename = os.path.join(RESULT_FOLDER, f"result_{file.filename}")
        result_img.save(result_filename)
        
        buffer = io.BytesIO()
        img.save(buffer, format='JPEG')
        img_str = base64.b64encode(buffer.getvalue()).decode('utf-8')
        
        buffer = io.BytesIO()
        result_img.save(buffer, format='JPEG')
        result_str = base64.b64encode(buffer.getvalue()).decode('utf-8')
        
        return jsonify({
            'original': f"data:image/jpeg;base64,{img_str}",
            'result': f"data:image/jpeg;base64,{result_str}",
            'plates': plates
        })
    
    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({'error': str(e)})

if __name__ == '__main__':
    load_model() 
    app.run(debug=True)