import os
import random
import tensorflow as tf
import numpy as np
from PIL import Image
from flask import Flask, render_template, request, redirect, url_for, flash, send_file
from flask_sqlalchemy import SQLAlchemy
from flask_migrate import Migrate  # Added for migrations
from flask_login import LoginManager, UserMixin, login_user, login_required, logout_user, current_user
from werkzeug.security import generate_password_hash, check_password_hash
from io import BytesIO
from datetime import datetime
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart

app = Flask(__name__)
app.secret_key = 'your-secret-key'
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///database.db'
app.config['UPLOAD_FOLDER'] = 'static/uploads'
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Database
db = SQLAlchemy(app)
migrate = Migrate(app, db)  # Initialize migrations
login_manager = LoginManager(app)
login_manager.login_view = 'login'

# Load model
model = tf.keras.models.load_model('complete_model (1).keras')
IMG_SIZE = 128
class_names = ['COVID', 'NORMAL', 'PNEUMONIA']

# User and Prediction models
class User(UserMixin, db.Model):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(50), unique=True, nullable=False)
    password = db.Column(db.String(100), nullable=False)
    name = db.Column(db.String(100))
    age = db.Column(db.Integer)
    gender = db.Column(db.String(10))
    role = db.Column(db.String(10), default='user')
    last_login = db.Column(db.DateTime)  # New column

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

# Database initialization
with app.app_context():
    db.create_all()
    if not User.query.filter_by(username='admin').first():
        admin = User(username='admin', password=generate_password_hash('Admin@123'), role='admin')
        db.session.add(admin)
        db.session.commit()

# Prediction function
def predict_image(image_path):
    try:
        img = Image.open(image_path).resize((IMG_SIZE, IMG_SIZE)).convert('RGB')
        if img.size[0] < 128 or img.size[1] < 128:
            return None, "Image resolution too low (min 128x128)"
        img_array = np.array(img) / 255.0
        img_array = np.expand_dims(img_array, 0)
        pred = model.predict(img_array, verbose=0)
        return pred, None
    except Exception:
        return None, "Invalid image: Not a chest X-ray or corrupted file"

# Routes
@app.route('/')
def home():
    return render_template('home.html')

@app.route('/about')
def about():
    return render_template('about.html')

@app.route('/faq')
def faq():
    return render_template('faq.html')

@app.route('/contact')
def contact():
    return render_template('contact.html')

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        captcha_answer = int(request.form['captcha_answer'])
        captcha_num1, captcha_num2 = map(int, request.form['captcha'].split('+'))
        if captcha_answer != captcha_num1 + captcha_num2:
            flash('Incorrect captcha')
            return redirect(url_for('login'))
        user = User.query.filter_by(username=username).first()
        if user and check_password_hash(user.password, password):
            user.last_login = datetime.utcnow()
            db.session.commit()
            login_user(user)
            return redirect(url_for('predict'))
        flash('Invalid credentials')
    captcha_num1, captcha_num2 = random.randint(0, 9), random.randint(0, 9)
    return render_template('login.html', captcha=f"{captcha_num1}+{captcha_num2}")

@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        role = request.form['role']
        if role == 'guest':
            gender = request.form['gender']
            username = f"guest_{random.randint(1000, 9999)}"
            user = User(username=username, password=generate_password_hash('guestpass'), gender=gender, role='guest')
        else:  # new user
            username = request.form['username']
            password = request.form['password']
            name = request.form['name']
            age = request.form['age']
            gender = request.form['gender']
            if len(password) < 8 or not any(c.isupper() for c in password) or not any(c in '!@#$%^&*' for c in password):
                flash('Password must be 8+ chars with 1 uppercase and 1 special symbol')
                return redirect(url_for('register'))
            if User.query.filter_by(username=username).first():
                flash('Username already exists')
                return redirect(url_for('register'))
            user = User(username=username, password=generate_password_hash(password), name=name, age=int(age), gender=gender, role='user')
        db.session.add(user)
        db.session.commit()
        flash('Registration successful. Please login.')
        return redirect(url_for('login'))
    return render_template('register.html')

@app.route('/logout')
@login_required
def logout():
    logout_user()
    return redirect(url_for('home'))

@app.route('/predict', methods=['GET', 'POST'])
@login_required
def predict():
    if request.method == 'POST':
        if 'image' not in request.files:
            flash('No file uploaded')
            return redirect(url_for('predict'))
        file = request.files['image']
        if file.filename == '':
            flash('No file selected')
            return redirect(url_for('predict'))
        if file and file.filename.lower().endswith(('.jpg', '.jpeg', '.png')):
            filename = f"{current_user.id}_{datetime.utcnow().strftime('%Y%m%d%H%M%S')}_{file.filename}"
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)
            pred, error = predict_image(filepath)
            if error:
                flash(error)
                return redirect(url_for('predict'))
            predicted_class = class_names[np.argmax(pred)]
            confidence = np.max(pred) * 100
            probs = {class_names[i]: f"{pred[0][i]*100:.2f}" for i in range(len(class_names))}
            
            # Save prediction to database
            prediction = Prediction(user_id=current_user.id, filename=filename, predicted_class=predicted_class, confidence=confidence)
            db.session.add(prediction)
            db.session.commit()
            
            return render_template('predict.html', predicted_class=predicted_class, confidence=confidence, probs=probs, filename=filename)
        flash('Only JPG, JPEG, PNG allowed')
    return render_template('predict.html')

@app.route('/generate_report')
@login_required
def generate_report():
    predicted_class = request.args.get('predicted_class')
    confidence = request.args.get('confidence')
    filename = request.args.get('filename')
    report = (
        f"Prediction Report\n"
        f"Name: {current_user.name or 'Guest'}\n"
        f"Age: {current_user.age or 'N/A'}\n"
        f"Gender: {current_user.gender or 'N/A'}\n"
        f"Date: {datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S')}\n"
        f"Predicted: {predicted_class} with {confidence}% confidence\n"
        f"Image: {filename}"
    )
    return send_file(BytesIO(report.encode()), download_name=f'report_{filename}.txt', as_attachment=True)

@app.route('/notify_radiologist', methods=['POST'])
@login_required
def notify_radiologist():
    predicted_class = request.form['predicted_class']
    confidence = request.form['confidence']
    filename = request.form['filename']
    if predicted_class == 'COVID':
        msg = MIMEMultipart()
        msg['From'] = 'your-email@gmail.com'
        msg['To'] = 'radiologist@example.com'
        msg['Subject'] = 'COVID-19 Detection Alert'
        body = (
            f"Patient Report\n"
            f"Name: {current_user.name or 'Guest'}\n"
            f"Age: {current_user.age or 'N/A'}\n"
            f"Gender: {current_user.gender or 'N/A'}\n"
            f"Prediction: {predicted_class} with {confidence}% confidence\n"
            f"Image: {filename}"
        )
        msg.attach(MIMEText(body, 'plain'))
        with smtplib.SMTP('smtp.gmail.com', 587) as server:
            server.starttls()
            server.login('your-email@gmail.com', 'your-app-password')
            server.send_message(msg)
        flash('Report sent to radiologist')
    return redirect(url_for('predict'))

@app.route('/admin')
@login_required
def admin():
    if current_user.role != 'admin':
        flash('Access denied')
        return redirect(url_for('home'))
    users = User.query.all()
    predictions = Prediction.query.all()
    return render_template('admin.html', users=users, predictions=predictions)

@app.route('/admin/export')
@login_required
def export_data():
    if current_user.role != 'admin':
        flash('Access denied')
        return redirect(url_for('home'))
    import csv
    from io import StringIO
    output = StringIO()
    writer = csv.writer(output)
    writer.writerow(['ID', 'Username', 'Role', 'Last Login'])
    for user in User.query.all():
        writer.writerow([user.id, user.username, user.role, user.last_login])
    output.seek(0)
    return send_file(BytesIO(output.getvalue().encode()), download_name='users.csv', as_attachment=True)

@app.route('/admin/export_predictions')
@login_required
def export_predictions():
    if current_user.role != 'admin':
        flash('Access denied')
        return redirect(url_for('home'))
    import csv
    from io import StringIO
    output = StringIO()
    writer = csv.writer(output)
    writer.writerow(['ID', 'User ID', 'Filename', 'Predicted Class', 'Confidence', 'Timestamp'])
    for pred in Prediction.query.all():
        writer.writerow([pred.id, pred.user_id, pred.filename, pred.predicted_class, pred.confidence, pred.timestamp])
    output.seek(0)
    return send_file(BytesIO(output.getvalue().encode()), download_name='predictions.csv', as_attachment=True)

if __name__ == '__main__':
    app.run(debug=True)