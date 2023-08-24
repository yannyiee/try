from typing import Container
import streamlit as st
from PIL import Image
import tensorflow as tf
import cv2
import numpy as np
import pandas as pd
from threading import Thread

# models
model_paths = [
    'mobilenetv2_6_Classes_model.keras',
    'densenet121_6_Classes_85.66%.keras',
    'vgg16_6_Classes_model.keras',
    'resnet50v2_6_Classes_model.keras'
]

model_names = [
    'MobileNetV2',
    'DenseNet121',
    'VGG16',
    'ResNet50V2'
]

# Load models using the custom model loading function
models = [tf.keras.models.load_model(path) for path in model_paths]

# #option page
st.markdown("<h1 style='color:red;font-size:30px;text-align: center;'>Welcome to Cat Breed Recognition</h1>", unsafe_allow_html=True)
st.write("")
st.markdown("<h2 style='color:blue;text-align: center;'>Select a method</h2>", unsafe_allow_html=True)

# button
empty_col, col1 = st.columns([4,1])
with empty_col:
    st.write("")
with col1:
    camera_clicked = st.button("Camera Capture")

st.title('Image Recognition')
st.write('Upload an image for recognition')
        # Upload image through Streamlit's file uploader
uploaded_image = st.file_uploader("Choose an image...", type=["jpg","jpeg", "png"])
if uploaded_image is not None:
    target_size = (224, 224)  # Adjust to match your model's input size

    # Load class names
    class_names = {
            0 : 'Bengal',
            1 : 'Persian',
            2 : 'Ragdoll',
            3 : 'Siamese',
            4 : 'Sphynx - Hairless Cat',
            5 : 'Tuxedo'
    }
    image = Image.open(uploaded_image)
    image = image.resize(target_size)
    image_array = np.array(image) / 255.0
    image_array = np.expand_dims(image_array,axis=0)

    st.image(image, caption='Uploaded Image', use_column_width=True)
    st.write("")

    class_occurrences = {class_name: 0 for class_name in class_names.values()}
    table_data = []

    # Display the prediction results for each class and each model
    for model_name, model in zip(model_names, models):
        row = {"Model":model_name}
        prediction = model.predict(image_array)
        #show the highest
        predicted_class_index = np.argmax(prediction)
        predicted_class_name = class_names[predicted_class_index]
        class_occurrences[predicted_class_name] += 1
        for i, class_name in class_names.items():
                confidence = prediction[0][i]
                if confidence>0:
                    row[class_name] = f"{confidence * 100:.2f}%"
                    
        table_data.append(row)
        confidence = prediction[0][predicted_class_index]    
        st.write("")  # Empty line to separate predictions of different models

    # Find class names with highest probability above 85%
    above_85_percent_classes = []
    for class_name in class_names.values():
        above_85_percent_models = sum(
            [1 for row in table_data if class_name in row and float(row[class_name][:-1]) >= 85]
        )
        if above_85_percent_models >= 2:
            above_85_percent_classes.append(class_name)

    # Find the most frequent class name across models
    most_frequent_class = max(class_occurrences, key=class_occurrences.get)

    #check which class above 85%
    if above_85_percent_classes:
        if len(above_85_percent_classes) ==2:
            st.write("\n\nThe possible hybrid cat breeds between:")
            for class_name in above_85_percent_classes:
                st.write(class_name)
        elif most_frequent_class in class_names.values():
            st.markdown(f"<h4 style='font-size: 100px;color=green;'>The most possible cat breed: {most_frequent_class}</h4>", unsafe_allow_html=True)
        else:
            st.write("No matches found")
            
    # Create a DataFrame from the table data
    df = pd.DataFrame(table_data)
    df.set_index("Model", inplace=True)

# Apply CSS styling to the table
    def alternate_row_color(row_index):
        return 'background-color: #f0f0f0' if row_index % 2 == 0 else 'background-color: white'

    def highlight_max(s):
        is_max = s == s.max()
        return ['background-color: #379683' if v else '' for v in is_max]

    # Apply custom CSS styles to change label colors and alternate row colors
    styled_df = df.style.apply(highlight_max, axis=0).applymap(alternate_row_color, subset=pd.IndexSlice[:, df.columns != 'Model'])
    styled_df.set_table_styles([{
        'selector': 'th',
        'props': [
            ('background-color', '#907163'),
            ('color', 'black')  # Change x-axis label color
        ]
    }, {
        'selector': 'td',
        'props': [
            ('background-color', '#379683'),
            ('color', 'black')  # Change y-axis label color
        ]
    }])


# Check if any prediction is above 70% confidence level
    above_70_percent = any(
        any(float(row[class_name][:-1]) >= 70 for class_name in class_names.values())
        for row in table_data
    )

    # Display the styled DataFrame as a table if any prediction is above 70%, else print 'Unknown'
    if above_70_percent:
        st.markdown("<h3 style='color:green;text-align: center;'>Prediction Probabilities Table</h3>", unsafe_allow_html=True)
        st.table(styled_df)
    else:
        st.write(f"<h4 style='color=red;font-size: 25px;'>Unknown</h4>", unsafe_allow_html=True)

# ---------------------------------------------camera------------------------------------------------------------------------------------------

if camera_clicked:
   # models
    model_paths = [
        'mobilenetv2_6_Classes_model.keras',
        'densenet121_6_Classes_85.66%.keras',
        'vgg16_6_Classes_model.keras',
        'resnet50v2_6_Classes_model.keras'
    ]

    model_names = [
        'MobileNetV2',
        'DenseNet121',
        'VGG16',
        'ResNet50V2'
    ]

    # Load models using the custom model loading function
    models = [tf.keras.models.load_model(path) for path in model_paths]

        # Preprocess the frame
    def preprocess_frame(frame, target_size):
        resized_frame = cv2.resize(frame, target_size)
        normalized_frame = resized_frame / 255.0  # Normalize pixel values
        return normalized_frame

    target_size = (224, 224)  # Adjust to match your model's input size

    # Load class names
    class_names = {
            0 : 'Bengal',
            1 : 'Persian',
            2 : 'Ragdoll',
            3 : 'Siamese',
            4 : 'Sphynx - Hairless Cat',
            5 : 'Tuxedo'
    }
    # Capture frames from webcam
    cap = cv2.VideoCapture(0)  # 0 represents the default camera

    while True:
        ret, frame = cap.read()  # Read a frame from the camera
        if not ret:
            break
        
        # Preprocess the frame
        preprocessed_frame = preprocess_frame(frame, target_size)
        preprocessed_frame = np.expand_dims(preprocessed_frame, axis=0)  # Add batch dimension
        
        # Perform inference
        for model_name, model in zip(model_names, models):
            predictions = model.predict(preprocessed_frame)
            class_index = np.argmax(predictions)
            confidence = predictions[0, class_index] * 100
            predicted_class_name = class_names[class_index]
        
        # Display the results
        if confidence >= 70.0:
            color = (0, 255, 0)  # Green
        else:
            predicted_class_name = "Unknown"
            color = (0, 0, 255)  # Red
        
        cv2.putText(frame, f"Prediction: {predicted_class_name}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
        if predicted_class_name != "Unknown":
            cv2.putText(frame, f"Confidence: {confidence:.2f}%", (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
        
        if predicted_class_name != "Unknown":
            cv2.rectangle(frame, (0, 0), (frame.shape[1], frame.shape[0]), color, 2)
        
        cv2.imshow('Webcam Cat Detection', frame)
        #press q to quit
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
