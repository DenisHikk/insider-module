from flask import Flask, request, render_template, send_file
import cv2
import numpy as np
import os

app = Flask(__name__)
UPLOAD_FOLDER = 'uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

@app.route('/', methods=['GET', 'POST'])
def index():
    result_img = None
    if request.method == 'POST':
        f = request.files['image']
        block_size = int(request.form.get('block_size', 11))
        c_val = int(request.form.get('c_val', 2))

        filepath = os.path.join(UPLOAD_FOLDER, f.filename)
        f.save(filepath)

        image = cv2.imread(filepath, cv2.IMREAD_GRAYSCALE)
        if block_size % 2 == 0:
            block_size += 1
        processed = cv2.adaptiveThreshold(
            image, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, block_size, c_val
        )
        out_path = os.path.join(UPLOAD_FOLDER, 'processed.png')
        cv2.imwrite(out_path, processed)
        result_img = 'processed.png'

    return render_template('index.html', result=result_img)

@app.route('/images/<filename>')
def uploaded_file(filename):
    return send_file(os.path.join(UPLOAD_FOLDER, filename), mimetype='image/png')

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)