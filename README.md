# Diabetes Risk Prediction App

This repository contains a complete machine learning project designed to predict the likelihood of diabetes based on individual health metrics. It includes everything from data exploration and modeling to deploying a functional web app using Flask.

## Project Overview

The goal of this project is to build an accessible tool that predicts diabetes risk using input from users. The model is embedded in a web application where users can enter clinical information and receive a prediction in real-time.

## Project Pipeline

The following stages were followed:

1. **Data Collection** – A public medical dataset was used, focusing on diabetes diagnoses.
2. **Exploratory Data Analysis** – Patterns and correlations were identified across variables.
3. **Data Visualization** – Charts and plots were created to provide visual insights.
4. **Data Preprocessing** – Missing values, outliers, and scaling were handled appropriately.
5. **Model Training** – Multiple machine learning algorithms were tested and compared.
6. **Model Evaluation** – Performance metrics such as accuracy, recall, and confusion matrix were used.

## Technical Details

### Machine Learning Model

- **Libraries**: `scikit-learn`, `pandas`, `numpy`, `matplotlib`, `seaborn`
- **Algorithms Used**: Logistic Regression, Decision Trees, Random Forest (among others)
- **Features**:
  - Pregnancies
  - Insulin
  - Age
  - BMI (Body Mass Index)
  - Blood Pressure
  - Glucose
  - Skin Thickness
  - Diabetes Pedigree Function
- **Target**: Predicts whether the person is likely to develop diabetes (Yes or No).

### Web Application

- **Framework**: Flask
- **User Flow**:
  - User submits personal health data through a form.
  - Model processes input and returns a binary prediction (Positive/Negative).
- **Deployment Options**: Can be hosted on Heroku, Render, or any Flask-compatible service.

## Getting Started

### Prerequisites

Make sure you have the following installed:

```bash
Python 3.x
pip
```

### Installation Steps

1. Clone the repository:

```bash
git clone https://github.com/annahico/Final_Project_ML.git
cd Final_Project_ML
```

2. Install the dependencies:

```bash
pip install -r requirements.txt
```

3. Run the Flask app:

```bash
python app.py
```

4. Open your browser and go to:

```bash
http://localhost:5000
```


## Future Improvements

- Integrate more advanced models such as XGBoost or neural networks.
- Add user authentication and profile history.
- Enhance UI/UX design for better experience.
- Expand dataset with lifestyle or medical history data.

## Contributing

Feel free to contribute! Submit issues or pull requests. For major changes, open a discussion first.

## Acknowledgements

- Dataset: [Pima Indians Diabetes Database – Kaggle](https://www.kaggle.com/datasets/johndasilva/diabetes)
- Libraries and Tools:
  - [scikit-learn](https://scikit-learn.org/)
  - [Flask](https://flask.palletsprojects.com/)

---

### 👩‍💻 Creator

Made with ❤️ by **annahico**

💡 *Contributions and suggestions are welcome! Feel free to fork, open issues, or submit pull requests.*

