from flask import Flask, request, redirect, url_for, render_template, session
from werkzeug.utils import secure_filename
import os
from image_processor import yolo_inference

app = Flask(__name__)
app.secret_key = 'your_secret_key_here'  # Replace with your actual secret key

UPLOAD_FOLDER = 'static/uploads/'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif'}

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Ensure upload folder exists
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/')
def home():
    return redirect(url_for('upload_file'))

@app.route('/upload', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        if 'file' not in request.files:
            return redirect(request.url)
        file = request.files['file']
        if file.filename == '':
            return redirect(request.url)
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            save_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(save_path)

            yolo_path = os.path.join(app.config['UPLOAD_FOLDER'], f"rotated_{filename}")
            ANBtype, SNBtype, SNAtype, ODItype, APDItype, FHItype, FMAtype, mwtype = yolo_inference(save_path, yolo_path)
            
            # Store the data list in the session
            session['data'] = [ANBtype, SNBtype, SNAtype, ODItype, APDItype, FHItype, FMAtype, mwtype]
            
            # Redirect to the uploaded_file route
            return redirect(url_for('uploaded_file', filename=filename))
    return render_template('upload.html')

@app.route('/uploads/<filename>')
def uploaded_file(filename):
    # Retrieve the 'data' list from the session
    data = session.get('data', [])
    return render_template('uploaded.html', filename=filename, data=data)

@app.route('/edit/<filename>')
def edit_landmarks(filename):
    return render_template('edit_landmarks.html', filename=filename)


if __name__ == '__main__':
    app.run(debug=False, host='0.0.0.0', port=8080)
