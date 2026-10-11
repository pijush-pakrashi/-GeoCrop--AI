import os
import time
import json
import string
import bcrypt
from flask import Flask, redirect, render_template, url_for, request, session, jsonify
from markupsafe import Markup
from flask_sqlalchemy import SQLAlchemy
from flask_login import UserMixin, login_user, LoginManager, login_required, logout_user, current_user
from wtforms import StringField, PasswordField, SubmitField, SelectField
from wtforms.validators import InputRequired, Length, ValidationError, Email
from flask_wtf import FlaskForm
from flask_bcrypt import Bcrypt
from datetime import datetime
import requests
import numpy as np
import pandas as pd
import config
import pickle
import io
import torch
from torchvision import transforms
from PIL import Image
from utils.model import ResNet9
from utils.fertilizer import fertilizer_dic
from utils.disease import disease_dic

# -------------------------LOADING THE TRAINED MODELS -----------------------------------------------

# Loading crop recommendation model
crop_recommendation_model_path = 'models/RandomForest.pkl'
crop_recommendation_model = pickle.load(
    open(crop_recommendation_model_path, 'rb'))

# Loading plant disease classification model

disease_classes = ['Apple___Apple_scab',
                   'Apple___Black_rot',
                   'Apple___Cedar_apple_rust',
                   'Apple___healthy',
                   'Blueberry___healthy',
                   'Cherry_(including_sour)___Powdery_mildew',
                   'Cherry_(including_sour)___healthy',
                   'Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot',
                   'Corn_(maize)___Common_rust_',
                   'Corn_(maize)___Northern_Leaf_Blight',
                   'Corn_(maize)___healthy',
                   'Grape___Black_rot',
                   'Grape___Esca_(Black_Measles)',
                   'Grape___Leaf_blight_(Isariopsis_Leaf_Spot)',
                   'Grape___healthy',
                   'Orange___Haunglongbing_(Citrus_greening)',
                   'Peach___Bacterial_spot',
                   'Peach___healthy',
                   'Pepper,_bell___Bacterial_spot',
                   'Pepper,_bell___healthy',
                   'Potato___Early_blight',
                   'Potato___Late_blight',
                   'Potato___healthy',
                   'Raspberry___healthy',
                   'Soybean___healthy',
                   'Squash___Powdery_mildew',
                   'Strawberry___Leaf_scorch',
                   'Strawberry___healthy',
                   'Tomato___Bacterial_spot',
                   'Tomato___Early_blight',
                   'Tomato___Late_blight',
                   'Tomato___Leaf_Mold',
                   'Tomato___Septoria_leaf_spot',
                   'Tomato___Spider_mites Two-spotted_spider_mite',
                   'Tomato___Target_Spot',
                   'Tomato___Tomato_Yellow_Leaf_Curl_Virus',
                   'Tomato___Tomato_mosaic_virus',
                   'Tomato___healthy']

# disease prediction
disease_model_path = 'models/plant_disease_model.pth'
disease_model = ResNet9(3, len(disease_classes))
disease_model.load_state_dict(torch.load(
    disease_model_path, map_location=torch.device('cpu')))
disease_model.eval()



def weather_fetch(city_name):
    """
    Fetch and returns the temperature and humidity of a city
    :params: city_name
    :return: temperature, humidity
    """
    api_key = config.weather_api_key
    base_url = "http://api.openweathermap.org/data/2.5/weather?"

    try:
        complete_url = base_url + "appid=" + api_key + "&q=" + str(city_name)
        response = requests.get(complete_url, timeout=4)
        x = response.json()

        if str(x.get("cod")) == "200" and "main" in x:
            y = x["main"]
            temperature = round((y["temp"] - 273.15), 2)
            humidity = y["humidity"]
            return temperature, humidity
    except Exception:
        pass

    # Reliable default agricultural climate fallback if API key is rate-limited/offline
    return 26.0, 75.0

def predict_image(img, model=disease_model):
    """
    Transforms image to tensor and predicts disease label
    :params: image
    :return: prediction (string)
    """
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.ToTensor(),
    ])
    image = Image.open(io.BytesIO(img))
    img_t = transform(image)
    img_u = torch.unsqueeze(img_t, 0)

    # Get predictions from model
    yb = model(img_u)
    # Pick index with highest probability
    _, preds = torch.max(yb, dim=1)
    prediction = disease_classes[preds[0].item()]
    # Retrieve the class label
    return prediction



app = Flask(__name__)
app.config["SQLALCHEMY_DATABASE_URI"] = "sqlite:///database.db"
app.config["SECRET_KEY"] = 'thisissecretkey'
db = SQLAlchemy(app)
bcrypt = Bcrypt(app)

SETTINGS_FILE = os.path.join(app.root_path, 'site_settings.json')

def get_site_settings():
    default_settings = {
        'home_title': 'Find out the most suitable crop to grow in your farm',
        'home_subtitle': 'Precision Agriculture Using Machine Learning & IoT',
        'about_title': 'Improving Agriculture, Cultivating Crops To Maximize Farmer Prosperity.',
        'about_desc': 'We use state-of-the-art machine learning (Random Forest) and deep convolutional neural networks (PyTorch ResNet-9) to guide farmers through the entire farming lifecycle. Make informed decisions to understand your regional soil chemistry, manage soil deficits, and accurately detect plant leaf diseases with actionable remedies.',
        'details_heading': 'GeoCropAI — Technical Architecture & Overview',
        'details_desc': 'Comprehensive precision farming ecosystem integrating machine learning, deep neural networks, and IoT sensors to maximize crop productivity.',
        'contact_email': 'support@geocropai.com',
        'contact_phone': '+91 98765 43210',
        'contact_address': 'Kolkata, West Bengal, India',
        'contact_hours': 'Mon - Sat: 9:00 AM - 6:00 PM'
    }
    if os.path.exists(SETTINGS_FILE):
        try:
            with open(SETTINGS_FILE, 'r', encoding='utf-8') as f:
                data = json.load(f)
                default_settings.update(data)
        except Exception:
            pass
    return default_settings

def save_site_settings(data):
    current = get_site_settings()
    current.update(data)
    with open(SETTINGS_FILE, 'w', encoding='utf-8') as f:
        json.dump(current, f, indent=2, ensure_ascii=False)

@app.context_processor
def inject_site_settings():
    return dict(site=get_site_settings())

login_manager = LoginManager()
login_manager.init_app(app)
login_manager.login_view = "login"

@login_manager.unauthorized_handler
def unauthorized():
    # Admin routes should redirect to AdminLogin, not the user login page
    if request.path.startswith('/admin') or request.path == '/admindashboard':
        return redirect(url_for('AdminLogin'))
    return redirect(url_for('login'))


@login_manager.user_loader
def load_user(user_id):
    if session.get('is_admin'):
        return UserAdmin.query.get(int(user_id))
    return User.query.get(int(user_id))

class User(db.Model, UserMixin):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(20), nullable=False, unique=True)
    email = db.Column(db.String(100), nullable=False, unique=True)
    password = db.Column(db.String(80), nullable=False)
    gender = db.Column(db.String(10), nullable=True)  # 'boy' or 'girl'
    profile_pic = db.Column(db.String(255), nullable=True)
    date_joined = db.Column(db.DateTime, default=datetime.utcnow)

class UserAdmin(db.Model, UserMixin):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(20), nullable=False, unique=True)
    password = db.Column(db.String(80), nullable=False)

    @property
    def email(self):
        return "admin@geocropai.com"

    @property
    def gender(self):
        return "boy"

    @property
    def date_joined(self):
        return datetime(2024, 1, 1)

    @property
    def profile_pic(self):
        return None

def validate_email_syntax(form, field):
    import re
    if not field.data or not re.match(r"^[^@\s]+@[^@\s]+\.[^@\s]+$", field.data.strip()):
        raise ValidationError("Please provide a valid email address.")

class RegisterForm(FlaskForm):
    username = StringField(validators=[InputRequired(), Length(min=3, max=20)], render_kw={"placeholder": "Username", "class": "form-control"})
    email = StringField(validators=[InputRequired(), validate_email_syntax], render_kw={"placeholder": "your@email.com", "class": "form-control"})
    password = PasswordField(validators=[InputRequired(), Length(min=5, max=20)], render_kw={"placeholder": "Password", "class": "form-control"})
    gender = SelectField('Gender', choices=[('male', '👨 MALE'), ('female', '👩 FEMALE'), ('other', '🧑 OTHER')], validators=[InputRequired()])
    submit = SubmitField("Create Account")

    def validate_username(self, username):
        existing = User.query.filter_by(username=username.data).first()
        if existing:
            raise ValidationError("Username already taken. Please choose another.")

    def validate_email(self, email):
        existing = User.query.filter_by(email=email.data).first()
        if existing:
            raise ValidationError("Email already registered. Please login.")


class LoginForm(FlaskForm):
    email = StringField(validators=[InputRequired(), validate_email_syntax], render_kw={"placeholder": "your@email.com", "class": "form-control"})
    password = PasswordField(validators=[InputRequired(), Length(min=5, max=20)], render_kw={"placeholder": "Password", "class": "form-control"})
    submit = SubmitField("Login")

class AdminLoginForm(FlaskForm):
    username = StringField(validators=[InputRequired(), Length(min=5, max=20)], render_kw={"placeholder": "Admin Username", "class": "form-control"})
    password = PasswordField(validators=[InputRequired(), Length(min=5, max=20)], render_kw={"placeholder": "Password", "class": "form-control"})
    submit = SubmitField("Admin Login")



class ContactUs(db.Model):
    sno = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(200), nullable=False)
    email = db.Column(db.String(500), nullable=False)
    text = db.Column(db.String(900), nullable=False)
    reply_text = db.Column(db.String(1500), nullable=True)
    replied_at = db.Column(db.DateTime, nullable=True)
    date_created = db.Column(db.DateTime, default=datetime.utcnow)

    def __repr__(self) -> str:
        return f"{self.sno} - {self.name}"


@app.route("/")
def hello_world():
    return render_template("index.html")
    

@app.route("/aboutus")
def aboutus():
    return render_template("aboutus.html")

@app.route("/details")
def details():
    return render_template("details.html")

@app.route("/contact", methods=['GET', 'POST'])
def contact():
    if request.method=='POST':
        name = request.form['name']
        email = request.form['email']
        text = request.form['text']
        contacts = ContactUs(name=name, email=email, text=text)
        db.session.add(contacts)
        db.session.commit()
    
    # Pass user's previous messages & admin replies if logged in
    user_messages = []
    if current_user.is_authenticated and not session.get('is_admin'):
        user_messages = ContactUs.query.filter_by(email=current_user.email).order_by(ContactUs.date_created.desc()).all()
    
    return render_template("contact.html", user_messages=user_messages)

@app.route("/login", methods=['GET', 'POST'])
def login():
    form = LoginForm()
    if current_user.is_authenticated:
        return redirect(url_for('dashboard'))
    elif form.validate_on_submit():
        user = User.query.filter_by(email=form.email.data).first()
        if user:
            if bcrypt.check_password_hash(user.password, form.password.data):
                session['is_admin'] = False
                login_user(user)
                return redirect(url_for('dashboard'))
        form.email.errors.append('Invalid email or password.')
    return render_template("login.html", form=form)

@app.route('/dashboard', methods=['GET', 'POST'])
@login_required
def dashboard():
    title = 'dashboard'
    return render_template('dashboard.html', title=title)

@app.route('/logout', methods=['GET', 'POST'])
@login_required
def logout():
    session.pop('is_admin', None)
    logout_user()
    return redirect(url_for('hello_world'))

def detect_gender_from_name(name):
    if not name:
        return 'boy'
    clean = name.strip().lower()
    female_names = {
        'sangita', 'sangeeta', 'priya', 'puja', 'pooja', 'ananya', 'ankita', 'riya', 'diya',
        'sneha', 'shreya', 'moumita', 'debjani', 'sweta', 'shweta', 'tina', 'rina', 'mina',
        'jaya', 'sushmita', 'neha', 'nisha', 'kavita', 'sunita', 'anita', 'rita', 'geeta',
        'gita', 'babita', 'monika', 'rekha', 'deepa', 'dipa', 'madhu', 'madhuri', 'payel',
        'payal', 'ritika', 'ishita', 'preeti', 'priti', 'anjali', 'radha', 'laxmi', 'lakshmi',
        'simran', 'muskan', 'khushi', 'tanya', 'sonia', 'sonam', 'arati', 'aarti', 'pallavi',
        'rashmi', 'rupa', 'suman', 'tanusree', 'tanushree', 'madhumita', 'soma', 'roma',
        'supriya', 'pratima', 'archana', 'alpana', 'barnali', 'chaitali', 'doyel', 'ipsita',
        'kakali', 'keya', 'koyel', 'manasi', 'mitra', 'nupur', 'paramita', 'paulomi',
        'piyali', 'poly', 'poulami', 'ratna', 'ruma', 'sampa', 'sharmila', 'shampa', 'sikha',
        'smriti', 'sohini', 'subhra', 'suchitra', 'sujata', 'sukanya', 'sarmistha', 'susmita',
        'tumpa', 'uma', 'urmi', 'baishakhi', 'chhanda', 'chitra', 'gargi', 'indrani', 'jhuma',
        'jhumur', 'kamala', 'kalyani', 'mousumi', 'nandita', 'paromita', 'piya', 'rimpa',
        'runa', 'sayani', 'srabani', 'sudeshna', 'swagata', 'tithi'
    }
    if clean in female_names:
        return 'girl'
    female_suffixes = (
        'ita', 'ika', 'iya', 'ina', 'ani', 'ali', 'ati', 'uti', 'shree', 'sree',
        'priya', 'devi', 'rani', 'bala', 'lata', 'wati', 'nita', 'mita', 'rita',
        'dita', 'gita', 'eeta', 'purna', 'mani', 'lekha', 'nandini'
    )
    for suf in female_suffixes:
        if clean.endswith(suf):
            return 'girl'
    male_suffixes = (
        'deep', 'jeet', 'pal', 'das', 'nath', 'esh', 'ish', 'osh', 'ash', 'raj',
        'dev', 'kumar', 'singh', 'jit', 'moy', 'joy', 'sen', 'roy', 'babu'
    )
    for suf in male_suffixes:
        if clean.endswith(suf):
            return 'boy'
    if clean.endswith(('a', 'i', 'ee')):
        male_a_exceptions = {'krishna', 'shiva', 'rama', 'surya', 'chandra', 'indra', 'aditya'}
        if clean not in male_a_exceptions:
            return 'girl'
    return 'boy'

@app.route("/signup", methods=['GET', 'POST'])
def signup():
    form = RegisterForm()
    if form.validate_on_submit():
        hashed_password = bcrypt.generate_password_hash(form.password.data)
        # Use the gender the user explicitly selected at signup (no guessing!)
        selected_gender = form.gender.data  # 'boy' or 'girl' — chosen by user
        new_user = User(username=form.username.data, email=form.email.data, password=hashed_password, gender=selected_gender)
        db.session.add(new_user)
        db.session.commit()
        return redirect(url_for('login'))
    return render_template("signup.html", form=form)


@app.route('/profile')
@login_required
def profile():
    if session.get('is_admin'):
        return redirect(url_for('admindashboard'))
    return render_template('profile.html', user=current_user)

@app.route('/upload-profile-photo', methods=['POST'])
@login_required
def upload_profile_photo():
    if 'profile_photo' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400
    file = request.files['profile_photo']
    if file.filename == '':
        return jsonify({'error': 'Empty filename'}), 400
    
    ext = file.filename.rsplit('.', 1)[-1].lower() if '.' in file.filename else 'png'
    if ext not in ['jpg', 'jpeg', 'png', 'webp', 'gif']:
        ext = 'png'
        
    upload_dir = os.path.join(app.root_path, 'static', 'uploads', 'profiles')
    os.makedirs(upload_dir, exist_ok=True)
    filename = f"user_{current_user.id}_{int(time.time())}.{ext}"
    filepath = os.path.join(upload_dir, filename)
    file.save(filepath)
    
    current_user.profile_pic = filename
    db.session.commit()
    
    photo_url = url_for('static', filename=f'uploads/profiles/{filename}')
    # Always return JSON — the profile page uses fetch() and expects JSON, never a redirect
    return jsonify({'success': True, 'filename': filename, 'photo_url': photo_url})


@app.route('/forgot-password', methods=['GET', 'POST'])
def forgot_password():
    error = None
    success = None
    if request.method == 'POST':
        action = request.form.get('action')
        if action == 'verify':
            username = request.form.get('username')
            email = request.form.get('email')
            user = User.query.filter_by(username=username, email=email).first()
            if user:
                return render_template('forgot_password.html', user_id=user.id, step='reset')
            else:
                error = 'No account found with those details.'
        elif action == 'reset':
            user_id = request.form.get('user_id')
            new_password = request.form.get('new_password')
            user = User.query.get(int(user_id))
            if user and new_password and len(new_password) >= 5:
                user.password = bcrypt.generate_password_hash(new_password)
                db.session.commit()
                success = 'Password updated! Please login.'
                return render_template('forgot_password.html', step='done', success=success)
    return render_template('forgot_password.html', step='verify', error=error)


@ app.route('/crop-recommend')
@login_required
def crop_recommend():
    title = 'crop-recommend - Crop Recommendation'
    return render_template('crop.html', title=title)

@ app.route('/fertilizer')
@login_required
def fertilizer_recommendation():
    title = '- Fertilizer Suggestion'
    return render_template('fertilizer.html', title=title)

# @app.route('/disease-predict', methods=['GET', 'POST'])
# @login_required
# def disease_prediction():
#     title = '- Disease Detection'
#     return render_template('disease.html', title=title)

@app.route('/disease-predict', methods=['GET', 'POST'])
@login_required
def disease_prediction():
    title = '- Disease Detection'

    if request.method == 'POST':
        if 'file' not in request.files:
            return render_template('disease.html', title=title)
        file = request.files.get('file')
        if not file:
            return render_template('disease.html', title=title)
        try:
            img = file.read()

            prediction = predict_image(img)

            prediction = Markup(str(disease_dic[prediction]))
            return render_template('disease-result.html', prediction=prediction, title=title)
        except:
            pass
    return render_template('disease.html', title=title)


# ===============================================================================================

# RENDER PREDICTION PAGES

# render crop recommendation result page


@ app.route('/crop-predict', methods=['POST'])
def crop_prediction():
    title = '- Crop Recommendation'

    if request.method == 'POST':
        try:
            N = int(request.form.get('nitrogen', '').strip() or 90)
        except (ValueError, TypeError):
            N = 90
        try:
            P = int(request.form.get('phosphorous', '').strip() or 42)
        except (ValueError, TypeError):
            P = 42
        try:
            K = int(request.form.get('pottasium', '').strip() or 43)
        except (ValueError, TypeError):
            K = 43
        try:
            ph = float(request.form.get('ph', '').strip() or 6.5)
        except (ValueError, TypeError):
            ph = 6.5
        try:
            rainfall = float(request.form.get('rainfall', '').strip() or 200.0)
        except (ValueError, TypeError):
            rainfall = 200.0

        city = (request.form.get("city") or "").strip() or "Kolkata"

        weather_info = weather_fetch(city)
        if weather_info is not None:
            temperature, humidity = weather_info
            data = np.array([[N, P, K, temperature, humidity, ph, rainfall]])
            my_prediction = crop_recommendation_model.predict(data)
            final_prediction = my_prediction[0]

            return render_template('crop-result.html', prediction=final_prediction, title=title)
        else:
            return render_template('try_again.html', title=title)

# render fertilizer recommendation result page

@ app.route('/fertilizer-predict', methods=['POST'])
def fert_recommend():
    title = '- Fertilizer Suggestion'

    crop_name = str(request.form.get('cropname', '') or request.form.get('cropname_hidden', '') or 'rice').lower().strip()
    if crop_name in ['', 'select']:
        crop_name = 'rice'

    try:
        N = int(request.form.get('nitrogen', '').strip() or 40)
    except (ValueError, TypeError):
        N = 40
    try:
        P = int(request.form.get('phosphorous', '').strip() or 20)
    except (ValueError, TypeError):
        P = 20
    try:
        K = int(request.form.get('pottasium', '').strip() or 20)
    except (ValueError, TypeError):
        K = 20

    df = pd.read_csv('dataset/fertilizer.csv')
    matched = df[df['Crop'] == crop_name]
    if matched.empty:
        matched = df[df['Crop'] == 'rice']

    nr = matched['N'].iloc[0]
    pr = matched['P'].iloc[0]
    kr = matched['K'].iloc[0]

    n = nr - N
    p = pr - P
    k = kr - K
    temp = {abs(n): "N", abs(p): "P", abs(k): "K"}
    max_value = temp[max(temp.keys())]
    if max_value == "N":
        if n < 0:
            key = 'NHigh'
        else:
            key = "Nlow"
    elif max_value == "P":
        if p < 0:
            key = 'PHigh'
        else:
            key = "Plow"
    else:
        if k < 0:
            key = 'KHigh'
        else:
            key = "Klow"

    response = Markup(str(fertilizer_dic[key]))
    return render_template('fertilizer-result.html', recommendation=response, title=title)


@app.route("/display")
def querydisplay():
    alltodo = ContactUs.query.all()
    return render_template("display.html",alltodo=alltodo)

@app.route("/AdminLogin", methods=['GET', 'POST'])
def AdminLogin():
    form = AdminLoginForm()
    if current_user.is_authenticated and session.get('is_admin'):
        return redirect(url_for('admindashboard'))
    elif form.validate_on_submit():
        user = UserAdmin.query.filter_by(username=form.username.data).first()
        if user and bcrypt.check_password_hash(user.password, form.password.data):
            session['is_admin'] = True
            login_user(user)
            return redirect(url_for('admindashboard'))
        form.username.errors.append('Invalid administrator credentials.')
    return render_template("adminlogin.html", form=form)

@app.route("/admindashboard")
@login_required
def admindashboard():
    if not session.get('is_admin'):
        return redirect(url_for('dashboard'))
    alltodo = ContactUs.query.order_by(ContactUs.sno.desc()).all()
    alluser = User.query.order_by(User.id.desc()).all()
    return render_template("admindashboard.html", alltodo=alltodo, alluser=alluser)

@app.route("/admin/reply-query/<int:sno>", methods=['POST'])
@login_required
def admin_reply_query(sno):
    if not session.get('is_admin'):
        return redirect(url_for('hello_world'))
    item = ContactUs.query.get_or_404(sno)
    reply_msg = request.form.get('reply_message', '').strip()
    if reply_msg:
        item.reply_text = reply_msg
        item.replied_at = datetime.utcnow()
        db.session.commit()
    return redirect(url_for('admindashboard'))

@app.route("/admin/delete-query/<int:sno>", methods=['POST'])
@login_required
def admin_delete_query(sno):
    if not session.get('is_admin'):
        return redirect(url_for('hello_world'))
    item = ContactUs.query.get_or_404(sno)
    db.session.delete(item)
    db.session.commit()
    return redirect(url_for('admindashboard'))

@app.route("/admin/delete-user/<int:user_id>", methods=['POST'])
@login_required
def admin_delete_user(user_id):
    if not session.get('is_admin'):
        return redirect(url_for('hello_world'))
    u = User.query.get_or_404(user_id)
    db.session.delete(u)
    db.session.commit()
    return redirect(url_for('admindashboard'))

@app.route("/admin/update-content", methods=['POST'])
@login_required
def admin_update_content():
    if not session.get('is_admin'):
        return redirect(url_for('hello_world'))
    fields = [
        'home_title', 'home_subtitle',
        'about_title', 'about_desc',
        'details_heading', 'details_desc',
        'contact_email', 'contact_phone', 'contact_address', 'contact_hours'
    ]
    data = {}
    for f in fields:
        val = request.form.get(f)
        if val is not None:
            data[f] = val.strip()
    save_site_settings(data)
    return redirect(url_for('admindashboard'))

@app.route("/reg", methods=['GET', 'POST'])
def reg():
    # Admin registration disabled - single admin only
    return redirect(url_for('AdminLogin'))


# ============================================================
# ADMIN CREDENTIALS (SINGLE ADMIN ONLY — DO NOT MODIFY)
# Username: ADMIN-CGCA
# Password: Admin20033
# ============================================================
ADMIN_USERNAME = 'ADMIN-CGCA'
ADMIN_PASSWORD = 'Admin20033'

def init_database():
    with app.app_context():
        db.create_all()
        # Remove any old admins and enforce single ADMIN-CGCA account
        old = UserAdmin.query.filter(UserAdmin.username != ADMIN_USERNAME).all()
        for o in old:
            db.session.delete(o)
        db.session.commit()
        existing = UserAdmin.query.filter_by(username=ADMIN_USERNAME).first()
        if existing:
            # Update password to ensure it matches
            existing.password = bcrypt.generate_password_hash(ADMIN_PASSWORD)
            db.session.commit()
        else:
            hashed_pwd = bcrypt.generate_password_hash(ADMIN_PASSWORD)
            admin = UserAdmin(username=ADMIN_USERNAME, password=hashed_pwd)
            db.session.add(admin)
            db.session.commit()
        # NOTE: Gender is always set by the user at signup — NEVER auto-detect/overwrite it here.
        print("[GeoCropAI] Admin account ready: " + ADMIN_USERNAME)



if __name__ == "__main__":
    init_database()
    app.run(debug=True, port=8000, threaded=True)
