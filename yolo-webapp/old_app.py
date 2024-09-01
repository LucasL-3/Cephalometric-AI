from flask import Flask, request, render_template, send_from_directory
import torch
import detect

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return 'No file part'
    file = request.files['file']
    if file.filename == '':
        return 'No selected file'
    if file:
        filepath = f'uploads/{file.filename}'
        file.save(filepath)
        # Run YOLO detection
        detect.run(source=filepath, weights='yolov5s.pt', save_txt=True, save_conf=True)
        return send_from_directory('runs/detect/exp', file.filename)

if __name__ == '__main__':
    app.run(debug=True, host='127.0.0.1', port=8080)

from flask import Flask, request, render_template, send_from_directory
import torch
import detect

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return 'No file part'
    file = request.files['file']
    if file.filename == '':
        return 'No selected file'
    if file:
        filepath = f'uploads/{file.filename}'
        file.save(filepath)
        # Run YOLO detection
        detect.run(source=filepath, weights='yolov5s.pt', save_txt=True, save_conf=True)
        return send_from_directory('runs/detect/exp', file.filename)

if __name__ == '__main__':
    app.run(debug=True, host='127.0.0.1', port=8080)

