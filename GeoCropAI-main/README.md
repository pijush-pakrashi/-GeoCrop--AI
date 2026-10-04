<div align="center">

<img src="static/images/tomato-growing-geocropai.gif" width="500">

</div>

#  🌱 GeoCropAI — Precision Agriculture using Machine Learning

GeoCropAI is a machine-learning-based web application designed to assist farmers with data-driven agricultural decisions.

## DATA SOURCE 📊
- [Crop recommendation dataset ](https://www.kaggle.com/atharvaingle/crop-recommendation-dataset) (custom built dataset)
- [Fertilizer suggestion dataset](https://github.com/Gladiator07/Harvestify/blob/master/Data-processed/fertilizer.csv) (custom built dataset)
- [Disease detection dataset](https://www.kaggle.com/vipoooool/new-plant-diseases-dataset)

## MOTIVATION 💪
Farming is one of the major sectors that influences a country’s economic growth.

 -  In country like India, majority of the population is dependent on agriculture for their livelihood. Many new technologies, such as Machine Learning and Deep Learning, are being implemented into agriculture so that it is easier for farmers to grow and maximize their yield.

 -   In this project, I present a website in which the following applications are implemented; Crop recommendation, Fertilizer recommendation and Plant disease prediction, respectively.

 -  In the crop recommendation application, the user can provide the soil data from their side and the application will predict which crop should the user grow.

  -   For the fertilizer recommendation application, the user can input the soil data and the type of crop they are growing, and the application will predict what the soil lacks or has excess of and will recommend improvements.

  -   For the last application, that is the plant disease prediction application, the user can input an image of a diseased plant leaf, and the application will predict what disease it is and will also give a little background about the disease and suggestions to cure it.

  
## 👥 Contributors

### GeoCropAI Team

| Contributor |
|-------------|
| **Mandrita Dasgupta** |
| **Debdeep Ghosh** |
| **Pijush Pakrashi** |
| **Asmrit Banerjee** |

We worked collaboratively on the development, implementation, and enhancement of **GeoCropAI**, combining machine learning, deep learning, and web technologies to build a unified precision agriculture platform.

## System Architecture/Workflow
```bash
                         ┌──────────────────┐
                         │       User          │
                         └────────┬─────────┘
                                   │
                                   ▼
                     ┌─────────────────────────┐
                       │     Flask Web App       │
                     └───────────┬─────────────┘
                                   │
             ┌──────────────────┼──────────────────┐
             ▼                    ▼                    ▼
      Crop Recommendation  Fertilizer System   Disease Detection
             │                   │                   │
             ▼                  ▼                   ▼
       Soil + Weather       N/P/K Analysis      Leaf Image
             │                   │                  │
             ▼                  ▼                  ▼
       Random Forest       Rule-Based Logic      ResNet9
             │                   │                  │
             ▼                  ▼                  ▼
       Crop Prediction    Fertilizer Advice    Disease Result

```
## ML Pipeline
```bash
Soil Parameters + City
          ↓
 OpenWeatherMap API
          ↓
Temperature + Humidity
          ↓
Random Forest Model
          ↓
 Recommended Crop
```
```bash
Leaf Image
    ↓
Image Preprocessing
    ↓
   ResNet9
    ↓
38-Class Classification
    ↓
Disease Information
    ↓
Prevention / Cure Guidance
```

## Home Page of our WebApplication
![Home Page of our WebApplication](https://github.com/atharval1/precision-agriculture-using-machine-learning/blob/main/Project-docs/App-snaps/Home.png)


## How to use 💻
- Crop Recommendation system ==> enter the corresponding nutrient values of your soil, state and city. Note that, the N-P-K (Nitrogen-Phosphorous-Pottasium) values to be entered should be the ratio between them. Refer this website for more information. Note: When you enter the city name, make sure to enter mostly common city names. Remote cities/towns may not be available in the Weather API from where humidity, temperature data is fetched.

- Fertilizer suggestion system ==> Enter the nutrient contents of your soil and the crop you want to grow. The algorithm will tell which nutrient the soil has excess of or lacks. Accordingly, it will give suggestions for buying fertilizers.

- Disease Detection System ==> Upload an image of leaf of your plant. The algorithm will tell the crop type and whether it is diseased or healthy. If it is diseased, it will tell you the cause of the disease and suggest you how to prevent/cure the disease accordingly. Note that, for now it only supports few crops.

## 💻 How to Run Locally

### 1. Clone the Repository

```bash
git clone https://github.com/mandrita16/GeoCropAI.git
```
### 2. Navigate to the Project Directory
```bash
cd GeoCropAI
```

### 3. Create a Virtual Environment

Create a Python virtual environment to keep the project's dependencies isolated:
```bash
python -m venv venv
```
### 4. Activate the Virtual Environment

For Windows:
```bash
venv\Scripts\activate
```
For macOS / Linux:
```bash
source venv/bin/activate
```

### 5. Install the Required Dependencies

Install all the required Python packages using the project's requirements file:
```bash
pip install -r requirements.txt
```

### 6. Configure the Application

Set up your local config.py file with the required API credentials and configuration values.

⚠️ Important: Never upload API keys, passwords, tokens, or other sensitive information to GitHub. Keep all credentials private and use environment variables whenever possible.

### 7. Run the Application

Start the Flask application by running:
```bash
python app.py
```
Once the application starts successfully, the terminal will display the local server address. Open that URL in your web browser to access GeoCropAI.

🔄 Quick Setup

If Python and Git are already installed, the basic setup is:
```bash
git clone https://github.com/mandrita16/GeoCropAI.git
cd GeoCropAI
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
python app.py
```

