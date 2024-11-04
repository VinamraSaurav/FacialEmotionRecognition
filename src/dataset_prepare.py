import pandas as pd
import os
import cv2
import numpy as np

# Set the path to the CSV file
data_dir = 'fer2013.csv'  # Make sure the CSV file is in the same directory

# Load the data
data = pd.read_csv(data_dir)

# Map integer emotion labels to their corresponding string names
emotion_labels = {
    0: 'angry',
    1: 'disgusted',
    2: 'fearful',
    3: 'happy',
    4: 'neutral',
    5: 'sad',
    6: 'surprised'
}

# Specify the output directory
output_dir = 'data'
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# Process the data
for index, row in data.iterrows():
    emotion = row['emotion']
    pixels = row['pixels'].split(' ')
    img = np.array(pixels, dtype='uint8').reshape(48, 48)
    img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)  # Convert to BGR format for saving

    # Use the string label for the emotion
    emotion_str = emotion_labels.get(emotion)
    if emotion_str is None:
        print(f"Unknown emotion label {emotion} at index {index}. Skipping...")
        continue

    # Create a directory for the emotion if it doesn't exist
    emotion_dir = os.path.join(output_dir, emotion_str)
    if not os.path.exists(emotion_dir):
        os.makedirs(emotion_dir)

    # Save the image
    cv2.imwrite(os.path.join(emotion_dir, f'{index}.png'), img)

print("Dataset preparation complete!")
