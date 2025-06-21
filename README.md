# 🌿 Plant Disease Detection using Deep Learning

This project is a deep learning-based solution for detecting plant diseases from images. It includes:

- A **trained CNN model** built and trained in `model.ipynb`.
- An interactive **Streamlit web app** (`main.py`) for uploading and classifying plant images.
- A **Dockerfile** to containerize the application for easy deployment.

---

## 🚀 Features

- Upload leaf images and get disease classification predictions instantly.
- Supports common image formats (JPG, JPEG, PNG).
- Uses a Keras model (`plant_disease_prediction.h5`) trained on a labeled dataset.
- Simple UI via Streamlit.
- Ready for deployment using Docker.

---

## 🧠 Model Architecture

- Model: Convolutional Neural Network (CNN)
- Input Size: 224x224
- Output: Softmax over plant disease classes
- Preprocessing: Resizing, normalization
- Training details and evaluation are included in the Jupyter Notebook (`model.ipynb`)

---

## 📸 Sample

![Model Working](https://github.com/user-attachments/assets/f7fde23d-a959-4bc7-9007-83187f7f1ddd)

---

## 📦 Project Structure

```
.
├── model.ipynb                 # Notebook for training and evaluating the model
├── app/
│      ├── Dockerfile                 # Docker container specification
│      ├── config.toml                # Docker container specification
│      ├── credentials.toml           # Docker container specification
│      ├── requirements.txt           # Requirement File
│      ├── main.py                    # Streamlit web app for image classification
│      ├── class_indices.json         # Mapping of class indices to disease names
│      └── trained_models/
│                 └── plant_disease_prediction.h5   # Trained Keras model
|
├── kaggle.json               # Kaggle Credentials
└── plantvillage dataset      # Dataset

````

---

## 🛠️ Setup Instructions

### 1. Clone the Repository

```bash
git clone https://github.com/OkRathod/Plant-Disease-Prediction-with-CNN.git
cd Plant-Disease-Prediction-with-CNN
````

### 2. Install Dependencies (Go to app folder first)

You can install required Python packages using:

```bash
pip install -r requirements.txt
```

If `requirements.txt` is not available, manually install:

```bash
pip install streamlit tensorflow pillow numpy
```

---

## 🧪 Running the App

Make sure your `trained_models/plant_disease_prediction.h5` and `class_indices.json` are available in the directory.

```bash
streamlit run main.py
```

Then open the URL shown in the terminal to access the web interface.

---

## 🐳 Docker Usage

### 1. Build the Docker Image

```bash
docker build -t plant-disease-app .
```

### 2. Run the Container

```bash
docker run -p 8501:8501 plant-disease-app
```

Then visit: [http://localhost:8501](http://localhost:8501)

---

## 🧾 Notes

* Ensure your model file (`plant_disease_prediction.h5`) is saved under `trained_models/`
* `class_indices.json` must reflect the label indices used during training.

---

## 📌 Future Improvements

* Add multi-label support for leaves with multiple diseases.
* Expand model to identify crop types and recommend treatments.
* Deploy on cloud with persistent storage and scalability.

---



