"""
Bone Age Prediction Web Application
Run: python app.py
Then open: http://localhost:5000
"""

from flask import Flask, render_template, request, jsonify
from predictor import predict_bone_age

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 16 * 1024 * 1024  # 16 MB limit

ALLOWED_EXTENSIONS = {"png", "jpg", "jpeg", "bmp", "tiff", "tif"}


def allowed_file(filename: str) -> bool:
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():
    if "image" not in request.files:
        return jsonify({"error": "Görüntü dosyası gönderilmedi."}), 400

    file   = request.files["image"]
    gender = request.form.get("gender", "").lower()

    if file.filename == "":
        return jsonify({"error": "Dosya seçilmedi."}), 400

    if not allowed_file(file.filename):
        return jsonify({"error": "Geçersiz dosya türü. PNG, JPG veya BMP yükleyin."}), 400

    if gender not in ("male", "female"):
        return jsonify({"error": "Cinsiyet seçimi gerekli (Erkek / Kız)."}), 400

    try:
        image_bytes = file.read()
        result      = predict_bone_age(image_bytes, gender)
        return jsonify(result)
    except FileNotFoundError as e:
        return jsonify({"error": str(e)}), 503
    except ValueError as e:
        return jsonify({"error": str(e)}), 400
    except Exception as e:
        return jsonify({"error": f"Tahmin sırasında hata oluştu: {str(e)}"}), 500


if __name__ == "__main__":
    print("=" * 55)
    print("  Kemik Yaşı Tahmin Sistemi başlatılıyor...")
    print("  Model yükleniyor (ilk istek biraz sürebilir)")
    print("  Adres: http://localhost:5000")
    print("=" * 55)
    app.run(debug=False, host="0.0.0.0", port=5000)
