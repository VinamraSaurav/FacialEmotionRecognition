
# Facial Emotion Recognition 😃

This project implements a real-time facial emotion detection system using deep learning techniques. The model is trained to recognize emotions from facial expressions in images or video streams.

## Table of Contents 📚
- [Cloning the Repository](#cloning-the-repository)
- [Dataset](#dataset)
- [Setting Up the Virtual Environment](#setting-up-the-virtual-environment)
- [Requirements](#requirements-)
- [Folder Structure](#folder-structure)
- [Preparing the Dataset](#preparing-the-dataset)
- [Training the Model](#training-the-model)
- [Real-time Emotion Detection](#real-time-emotion-detection)
- [Haarcascade Frontal Face Detection](#haarcascade-frontal-face-detection)
- [License](#license)

## Cloning the Repository 🧑‍💻

To get started, clone the repository using the following command:

```bash
git clone https://github.com/VinamraSaurav/FacialEmotionRecognition
cd FacialEmotionRecognition
```

## Dataset 📊

The project includes the `fer2013.csv` dataset, which contains facial expressions labeled with their corresponding emotions. This dataset is crucial for training the emotion detection model.

## Setting Up the Virtual Environment 🛠️

To set up a Python virtual environment, follow these steps:

1. **Create a Virtual Environment**:
   ```bash
   python -m venv venv
   ```

2. **Activate the Virtual Environment**:
   - On Windows:
     ```bash
     venv\Scripts\activate
     ```
   - On macOS/Linux:
     ```bash
     source venv/bin/activate
     ```

3. **Install Requirements**: Install the required packages using the `requirements.txt` file:
   ```bash
   pip install -r requirements.txt
   ```

## Requirements 📦
The following packages are required to run this project:

- TensorFlow
- Keras
- OpenCV
- NumPy
- Pandas
- Matplotlib

These dependencies are specified in the requirements.txt file, and you can install them using the above command.
## Folder Structure 🗂️

The project has the following folder structure:

```
FacialEmotionRecognition/
│
├── src/
│   ├── dataset_prepare.py          # Script to prepare the dataset from CSV
│   ├── emotions.py                 # Real-time emotion detection script
│   ├── train.py                    # Script for training the model
│   └── haarcascade_frontalface_default.xml  # Haar Cascade file for face detection
│
├── data/                           # Directory containing folders for each emotion
│   ├── angry/                      # Folder for images of angry faces
│   ├── sad/                        # Folder for images of sad faces
│   ├── happy/                      # Folder for images of happy faces
│   └── ...                         # Additional emotion folders
│
├── venv/                           # Python virtual environment directory
│
├── model.h5                        # Saved trained model file
├── fer2013.csv                     # Dataset in CSV format
├── README.md                       # Project documentation file
└── requirements.txt                # List of dependencies

```

## Preparing the Dataset 🗃️

Download the `fer2013.csv` dataset from sources like [Kaggle](https://www.kaggle.com/datasets/msambare/fer2013) and save it to the root directory of this project. This file contains images of facial expressions labeled with corresponding emotions, essential for training and organizing the dataset.

Once downloaded, use the `dataset_prepare.py` script to process the CSV file, which will organize the images into folders based on emotions.

Run the following command to prepare the dataset:

```bash
python src/dataset_prepare.py
```
This script reads ```fer2013.csv``` in root directory and creates emotion-specific folders with the images organized for model training. Make sure fer2013.csv is stored in the root directory of the project before running this command.


## Training the Model 🚀

To train the model using the prepared dataset, you can use the `train.py` script. Ensure that the dataset is prepared before running this script. Run the following command:

```bash
python src/train.py
```

This will start the training process and save the trained model as `model.h5`.


## Real-time Emotion Detection 🎥

The `emotions.py` script is used for real-time emotion detection. It captures video from the webcam and predicts emotions based on the faces detected in the video stream. Run the script using:

```bash
python src/emotions.py
```

Make sure the trained model (`model.h5`) is present in the root directory.

## Haarcascade Frontal Face Detection 👀

The project uses `haarcascade_frontalface_default.xml`, a pre-trained Haar Cascade model, for detecting faces in images and video streams. This XML file is included in the project directory and is used within the `emotions.py` script to identify faces before predicting their emotions.


