# Deploying a Scalable ML Pipeline with FastAPI

This project involves building and deploying a machine learning pipeline using **FastAPI**. The pipeline includes data preprocessing, model training, model inference, and evaluation, along with a FastAPI RESTful API for serving predictions.

---

## **Tools and Environments Used**
### **Programming Language & Frameworks**
- **Python 3.8+** – Core programming language used.
- **FastAPI** – For building the RESTful API.
- **scikit-learn** – Machine learning library used for data preprocessing, training, and evaluation.
- **pandas** – Data manipulation and analysis.
- **NumPy** – Numerical computing library.
- **pytest** – Unit testing framework.
- **DVC (Data Version Control)** – For versioning datasets.

### **Development & Environment Management**
- **Git & GitHub** – Version control and repository management.
- **GitHub Actions** – CI/CD pipeline to automate testing and linting.
- **pip** – Python package manager.
- **conda** – Alternative package and environment manager (used via `environment.yml`).
- **Docker** *(Optional)* – For containerizing the application.
- **Virtual Environments**:
  - **Conda environments** *(for Conda-based setup)*

### **Deployment & Infrastructure**
- **Uvicorn** – ASGI server for running FastAPI.

---

## **Project Overview**
This project uses a **census dataset** to build a machine learning model that predicts whether a person earns more than $50K based on various attributes. It includes:
1. **Data Preprocessing**
2. **Model Training & Evaluation**
3. **Performance Metrics Calculation**
4. **Model Deployment via FastAPI**
5. **Continuous Integration & Testing using GitHub Actions**
6. **Dataset Version Control with DVC**

---

## **Steps to Set Up the Project**

### **Environment Setup**
You can choose to set up the environment using either **pip** or **conda**.

#### **Option 1: Using Conda**
1. Create the environment:
    ```bash
    conda env create -f environment.yml
    ```
2. Activate the environment:
    ```bash
    conda activate fastapi
    ```

#### **Option 2: Using Pip**
1. Install dependencies using pip:
    ```bash
    python3 -m venv fastapi
    source fastapi/bin/activate
    pip install -r requirements.txt
    ```

---

### **GitHub Repository Setup**
1. Fork the repository and clone it to your local machine:
    ```bash
    git clone https://github.com/[your-github-username]/Deploying-a-Scalable-ML-Pipeline-with-FastAPI.git
    cd Deploying-a-Scalable-ML-Pipeline-with-FastAPI
    ```
2. Initialize a git repository (if not done yet):
    ```bash
    git init
    ```
3. Connect your local repository to GitHub:
    ```bash
    git remote add origin https://github.com/[your-username]/Deploying-a-Scalable-ML-Pipeline-with-FastAPI.git
    ```
4. Set up **GitHub Actions** for continuous integration, running both `pytest` for tests and `flake8` for code quality checks.

---

### **Data Management**
1. **Download the dataset (`census.csv`)** and place it in the `data/` directory.
2. **Use DVC (Data Version Control) to track the dataset:**
    ```bash
    dvc add data/census.csv
    git add data/census.csv.dvc
    git commit -m "Add census data to DVC tracking"
    ```

---

### **Data Preprocessing**
- Implemented in **`ml/data.py`**
- Handles:
  - One-hot encoding of categorical features (`OneHotEncoder`)
  - Label binarization of the target variable (`LabelBinarizer`)
  - Handling missing values (`?` replaced with `NaN`)
  - Feature scaling (if needed)

---

### **Model Development**
1. **Implemented in `ml/model.py`**
2. Functions:
   - `train_model(X_train, y_train)` – Trains the ML model using **RandomForestClassifier**.
   - `compute_model_metrics(y, preds)` – Computes **precision, recall, and F1-score**.
   - `inference(model, X)` – Runs inference on test data.
   - `save_model(model, path)` – Saves trained model.
   - `load_model(path)` – Loads a pre-trained model.

---

### **FastAPI API Creation**
1. Implemented in `main.py`
2. Endpoints:
   - **GET `/`** – Returns a welcome message.
   - **POST `/predict`** – Accepts JSON input and returns the model’s prediction.

---

### **Continuous Integration with GitHub Actions**
- **Setup in `.github/workflows/python-ci.yml`**
- **Runs on every push** to validate:
  - **`flake8`** (Python linting)
  - **`pytest`** (Unit tests)

---

### **Testing and Validation**
1. **Test file:** `test_ml.py`
2. Validates:
   - **Data Processing** (`process_data`)
   - **Model Training & Inference**
   - **Performance Metrics Calculation**

3. **Run tests manually:**
    ```bash
    pytest test_ml.py -v
    ```

---

### **Final Steps**
1. Once everything is set up, **commit all changes** and **push** them to GitHub:
    ```bash
    git add .
    git commit -m "Finalized ML pipeline and API"
    git push
    ```
2. **Ensure GitHub Actions workflow runs successfully**.
3. **Take a screenshot** of the passing workflow and save it in the `screenshots/` folder.

---

## **Folder Structure**
```
📦 Deploying-a-Scalable-ML-Pipeline-with-FastAPI
├── 📂 data/                      # Contains dataset
│   ├── census.csv                # Original dataset
│   ├── census.csv.dvc             # DVC tracking file
│   ├── clean_census.csv           # Preprocessed dataset
├── 📂 ml/                        # Machine Learning pipeline
│   ├── data.py                    # Data preprocessing
│   ├── model.py                   # Model training, inference, and evaluation
├── 📂 model/                      # Trained model artifacts
│   ├── model.pkl                   # Saved trained model
│   ├── encoder.pkl                 # OneHotEncoder
│   ├── lb.pkl                      # Label binarizer
├── 📂 screenshot/                 # Store proof of passing CI/CD tests
│   ├── continuous_integration.png  # Screenshot of passing GitHub Actions
├── train_model.py                 # Script for model training
├── main.py                        # FastAPI app
├── test_ml.py                     # Unit tests for the ML pipeline
├── local_api.py                   # Local API testing script
├── requirements.txt                # Dependencies
├── environment.yml                 # Conda environment setup
├── .github/workflows/python-ci.yml # GitHub Actions workflow for CI/CD
```

---

## **Future Improvements**
- **Dockerize the application** for containerized deployment.
- **Deploy API to AWS Lambda, Heroku, or Azure Functions**.
- **Use MLflow for experiment tracking**.
- **Automate model retraining on new data**.

---

## **Final Notes**
- **Follow best practices** for data versioning (DVC) and model persistence.
- **Use GitHub Actions** for continuous integration to ensure smooth development.
- **Ensure the API is functional** before deployment.

---

🚀 **Now you're ready to deploy your machine learning pipeline with FastAPI!** 🚀

