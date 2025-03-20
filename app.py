import os
import tensorflow as tf
import numpy as np
from PIL import Image
from flask import Flask, render_template, request, redirect, url_for, flash, send_file
from flask_sqlalchemy import SQLAlchemy
from flask_migrate import Migrate
from flask_login import LoginManager, UserMixin, login_user, login_required, logout_user, current_user
from werkzeug.security import generate_password_hash, check_password_hash
from io import BytesIO
from datetime import datetime
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)
app.secret_key = os.getenv('SECRET_KEY', 'your-secret-key')
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///database.db'
app.config['UPLOAD_FOLDER'] = 'static/uploads'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

db = SQLAlchemy(app)
migrate = Migrate(app, db)
login_manager = LoginManager(app)
login_manager.login_view = 'login'

logger.info("Loading TensorFlow model...")
try:
    model = tf.keras.models.load_model('model.keras')
    logger.info("Model loaded successfully")
except Exception as e:
    logger.error(f"Failed to load model: {str(e)}")
    raise

IMG_SIZE = 128
class_names = ['COVID', 'NORMAL', 'PNEUMONIA']

class User(UserMixin, db.Model):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(50), unique=True, nullable=False)
    password = db.Column(db.String(100), nullable=False)
    role = db.Column(db.String(10), default='user')

class Prediction(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)
    filename = db.Column(db.String(100), nullable=False)
    predicted_class = db.Column(db.String(20), nullable=False)
    confidence = db.Column(db.Float, nullable=False)
    timestamp = db.Column(db.DateTime, default=datetime.utcnow)

@login_manager.user_loader
def load_user(user_id):
    return User.query.get(int(user_id))

with app.app_context():
    db.create_all()
    if not User.query.filter_by(username='admin').first():
        admin = User(username='admin', password=generate_password_hash('admin123'), role='admin')
        db.session.add(admin)
        db.session.commit()

def predict_image(image_path):
    try:
        img = Image.open(image_path)
        original_size = img.size
        if original_size[0] < 128 or original_size[1] < 128:
            return None, "low_resolution", "Image resolution too low (minimum 128x128 pixels)"
        img = img.resize((IMG_SIZE, IMG_SIZE)).convert('RGB')
        img_array = np.array(img) / 255.0
        img_array = np.expand_dims(img_array, 0)
        pred = model.predict(img_array, verbose=0)
        return pred, None, None
    except Exception as e:
        logger.error(f"Exception in predict_image: {str(e)}")
        return None, "invalid_format", "Invalid image: Corrupted file or unsupported format"

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        user = User.query.filter_by(username=username).first()
        if user and check_password_hash(user.password, password):
            login_user(user)
            return redirect(url_for('predict'))
        flash('Invalid credentials')
    return render_template('login.html')

@app.route('/logout')
@login_required
def logout():
    logout_user()
    return redirect(url_for('home'))

@app.route('/predict', methods=['GET', 'POST'])
@login_required
def predict():
    error_type = None
    error_message = None
    predicted_class = None
    confidence = None
    probs = None
    filename = None

    if request.method == 'POST':
        if 'image' not in request.files:
            error_type = "no_file"
            error_message = "No file uploaded"
        else:
            file = request.files['image']
            if file.filename == '':
                error_type = "no_file"
                error_message = "No file selected"
            elif not file.filename.lower().endswith(('.jpg', '.jpeg', '.png')):
                error_type = "invalid_format"
                error_message = "Only JPG, JPEG, PNG files allowed"
            else:
                filename = f"{current_user.id}_{datetime.utcnow().strftime('%Y%m%d%H%M%S')}_{file.filename}"
                filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
                file.save(filepath)
                pred, err_type, err_msg = predict_image(filepath)
                if pred is None:
                    error_type = err_type
                    error_message = err_msg
                else:
                    predicted_class = class_names[np.argmax(pred)]
                    confidence = np.max(pred) * 100
                    probs = {class_names[i]: f"{pred[0][i]*100:.2f}" for i in range(len(class_names))}
                    prediction = Prediction(user_id=current_user.id, filename=filename, predicted_class=predicted_class, confidence=confidence)
                    db.session.add(prediction)
                    db.session.commit()

    return render_template('predict.html', error_type=error_type, error_message=error_message, 
                          predicted_class=predicted_class, confidence=confidence, probs=probs, filename=filename)

if __name__ == '__main__':
    app.run(debug=True)
