from flask import Flask, request, redirect, url_for, render_template, jsonify
from werkzeug.utils import secure_filename
import os
from image_processor import (
    yolo_inference,
    get_placeholder_measurements,
    analysis_from_landmarks,
    read_points_xy,
    read_points_file,
    get_keypoint_names,
)

app = Flask(__name__)
app.secret_key = 'your_secret_key_here'  # Replace with your actual secret key

UPLOAD_FOLDER = 'static/uploads/'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif'}

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# In-memory store for analysis results (avoids session/cookie issues on redirect)
_analysis_store = {}

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
            app_dir = os.path.dirname(os.path.abspath(__file__))
            upload_dir = os.path.join(app_dir, app.config['UPLOAD_FOLDER'])
            os.makedirs(upload_dir, exist_ok=True)
            save_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(os.path.join(app_dir, save_path))
            yolo_path = os.path.join(app_dir, app.config['UPLOAD_FOLDER'], f"rotated_{filename}")
            abs_save_path = os.path.join(app_dir, save_path)
            try:
                steiner_data = yolo_inference(abs_save_path, yolo_path)
                measurements = steiner_data.get('measurements') or []
                skeletal_class = steiner_data.get('skeletal_class', 'N/A')
                if not measurements:
                    measurements = get_placeholder_measurements()
                _analysis_store[filename] = {
                    'skeletal_class': skeletal_class,
                    'measurements': measurements,
                    'error': steiner_data.get('error', ''),
                    'quality': steiner_data.get('quality'),
                    'keypoint_confidence': steiner_data.get('keypoint_confidence'),
                }
            except Exception as e:
                app.logger.exception("Analysis failed")
                _analysis_store[filename] = {
                    'skeletal_class': 'N/A',
                    'measurements': get_placeholder_measurements(),
                    'error': str(e),
                    'quality': None,
                    'keypoint_confidence': None,
                }
            return redirect(url_for('uploaded_file', filename=filename))
    return render_template('upload.html')

@app.route('/uploads/<filename>')
def uploaded_file(filename):
    stored = _analysis_store.get(filename, {})
    data = stored.get('measurements', get_placeholder_measurements())
    skeletal_class = stored.get('skeletal_class', 'N/A')
    error = stored.get('error', '')
    quality = stored.get('quality')
    keypoint_confidence = stored.get('keypoint_confidence')
    return render_template('uploaded.html', filename=filename, data=data, skeletal_class=skeletal_class, error=error, quality=quality, keypoint_confidence=keypoint_confidence)

def _points_path_for_result_image(filename):
    """Path to the points file for a result image e.g. rotated_test.png -> .../rotated_test_points.txt."""
    app_dir = os.path.dirname(os.path.abspath(__file__))
    base = filename.rsplit(".", 1)[0]
    return os.path.join(app_dir, app.config["UPLOAD_FOLDER"], base + "_points.txt")


def _original_filename(result_filename):
    """rotated_test.png -> test.png."""
    if result_filename.startswith("rotated_"):
        return result_filename[8:]
    return result_filename


@app.route("/edit/<filename>")
def edit_landmarks(filename):
    points_path = _points_path_for_result_image(filename)
    points = read_points_xy(points_path)
    names = get_keypoint_names()
    return render_template(
        "edit_landmarks.html",
        filename=filename,
        points=points,
        names=names,
    )


@app.route("/edit/<filename>/save", methods=["POST"])
def save_landmarks(filename):
    data = request.get_json()
    if not data or "points" not in data:
        return jsonify({"ok": False, "error": "Missing points"}), 400
    points_path = _points_path_for_result_image(filename)
    os.makedirs(os.path.dirname(points_path), exist_ok=True)
    # Write landmark_id,x,y format
    with open(points_path, "w") as f:
        for row in data["points"]:
            f.write(f"{row[0]},{float(row[1])},{float(row[2])}\n")
    try:
        coords = read_points_file(points_path)
        skeletal_class, measurements = analysis_from_landmarks(coords)
        base = _original_filename(filename)
        _analysis_store[base] = {
            "skeletal_class": skeletal_class,
            "measurements": measurements,
            "error": "",
            "quality": None,
            "keypoint_confidence": None,
        }
    except Exception as e:
        app.logger.exception("Re-analysis failed")
        return jsonify({"ok": False, "error": str(e)}), 500
    return jsonify({"ok": True, "redirect": url_for("uploaded_file", filename=_original_filename(filename))})


if __name__ == '__main__':
    app.run(debug=False, host='0.0.0.0', port=8080)
