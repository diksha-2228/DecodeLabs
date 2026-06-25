import os
import sys
import traceback
from flask import Flask, request, jsonify, render_template, send_file
import pandas as pd

app = Flask(__name__)

# Configure upload and output folders
UPLOAD_FOLDER = os.path.abspath(os.path.join(os.path.dirname(__file__), 'data'))
OUTPUT_FOLDER = os.path.abspath(os.path.join(os.path.dirname(__file__), 'outputs'))

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['OUTPUT_FOLDER'] = OUTPUT_FOLDER

# Ensure src/ is in python path
src_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), 'src'))
if src_dir not in sys.path:
    sys.path.append(src_dir)

try:
    from pipeline_core import run_pipeline
except ImportError:
    from src.pipeline_core import run_pipeline


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return jsonify({"error": "No file part in the request"}), 400
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({"error": "No selected file"}), 400
        
    if not file.filename.endswith('.csv'):
        return jsonify({"error": "Only CSV files are supported"}), 400
        
    try:
        # Save to data/uploaded.csv
        uploaded_path = os.path.join(app.config['UPLOAD_FOLDER'], 'uploaded.csv')
        file.save(uploaded_path)
        
        # Verify it can be parsed as a CSV
        try:
            pd.read_csv(uploaded_path, nrows=5)
        except Exception as e:
            return jsonify({"error": f"Invalid CSV file format: {str(e)}"}), 400
            
        return jsonify({"message": "File uploaded successfully", "filename": file.filename}), 200
    except Exception as e:
        return jsonify({"error": f"Failed to save uploaded file: {str(e)}"}), 500


@app.route('/run-pipeline', methods=['POST'])
def run_data_pipeline():
    uploaded_path = os.path.join(app.config['UPLOAD_FOLDER'], 'uploaded.csv')
    if not os.path.exists(uploaded_path):
        return jsonify({"error": "No uploaded dataset found. Please upload a CSV first."}), 400
        
    try:
        # Run the pipeline core
        results = run_pipeline(uploaded_path, app.config['OUTPUT_FOLDER'])
        return jsonify(results), 200
    except ValueError as ve:
        # Handle custom validation errors (e.g. missing required columns)
        return jsonify({"error": str(ve)}), 400
    except Exception as e:
        # Catch-all for unexpected pipeline errors
        tb = traceback.format_exc()
        print("Pipeline Execution Error:")
        print(tb)
        return jsonify({"error": f"An error occurred during pipeline execution: {str(e)}"}), 500


@app.route('/download-cleaned', methods=['GET'])
def download_cleaned():
    cleaned_path = os.path.join(app.config['OUTPUT_FOLDER'], 'cleaned_house_listings.csv')
    if not os.path.exists(cleaned_path):
        return jsonify({"error": "No cleaned dataset found. Please run the pipeline first."}), 400
    
    return send_file(
        cleaned_path,
        mimetype='text/csv',
        as_attachment=True,
        download_name='cleaned_house_listings.csv'
    )


if __name__ == '__main__':
    app.run(host='127.0.0.1', port=5000, debug=True)
