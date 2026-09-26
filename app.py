"""
Flask application for displaying FPL AI pipeline results.
Run with:  python app.py
Open:      http://127.0.0.1:5000
"""

import os
from flask import Flask, render_template, request, redirect, url_for, flash
from werkzeug.utils import secure_filename

from pipeline_runner import run_pipeline
import prototype_fpl as fpl


# ------------------------------------------------------------
# App config
# ------------------------------------------------------------
app = Flask(__name__)
app.secret_key = "fpl-ai-secret-key-change-me"

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
UPLOAD_FOLDER = os.path.join(BASE_DIR, "uploads")
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER
app.config["MAX_CONTENT_LENGTH"] = 200 * 1024 * 1024   # 200 MB

ALLOWED_EXTENSIONS = {"mp3", "wav", "m4a", "ogg", "flac"}


def allowed_file(filename):
    return "." in filename and \
        filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS


# ------------------------------------------------------------
# Routes
# ------------------------------------------------------------
@app.route("/")
def index():
    return render_template("index.html")


@app.route("/analyze", methods=["POST"])
def analyze():
    audio_path = None

    # If a file was uploaded, save and use it
    if "audio" in request.files:
        f = request.files["audio"]
        if f and f.filename:
            if not allowed_file(f.filename):
                flash("Unsupported file type. Please upload MP3, WAV, M4A, "
                      "OGG or FLAC.", "error")
                return redirect(url_for("index"))
            filename = secure_filename(f.filename)
            saved_path = os.path.join(app.config["UPLOAD_FOLDER"], filename)
            f.save(saved_path)
            audio_path = saved_path
            flash("Uploaded {}.".format(filename), "success")

    # Otherwise fall back to the configured default clip
    if audio_path is None:
        audio_path = fpl.AUDIO_FILE

    # Run the pipeline (this blocks until finished)
    try:
        results = run_pipeline(audio_path)
    except Exception as e:
        flash("Pipeline error: {}".format(e), "error")
        return redirect(url_for("index"))

    return render_template("results.html", **results)


# ------------------------------------------------------------
# Entry point
# ------------------------------------------------------------
if __name__ == "__main__":
    app.run(debug=True, host="127.0.0.1", port=5000)