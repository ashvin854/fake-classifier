from flask import Flask, request, render_template, redirect, url_for
from tensorflow.keras.models import load_model
import numpy as np
from PIL import Image
import os

# Initialize Flask application
app = Flask(__name__)

# Load the trained model
model = load_model('product/fake_image_classifier.keras')

# Print model input shape and summary for debugging
print("Model input shape:", model.input_shape)
model.summary()

# Upload folder configuration
UPLOAD_FOLDER = 'static/uploads/'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Allowed file extensions for image upload
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}

# Check file extension
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# Home route
@app.route('/')
def home():
    return render_template('index.html')

# Prediction route
@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return redirect(request.url)

    file = request.files['file']

    if file and allowed_file(file.filename):
        # Save the uploaded image
        filename = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
        file.save(filename)

        # Open and process the image
        img = Image.open(filename)

        # Resize to match model input (128x128)
        img = img.resize((128, 128))

        # Convert image to numpy array and normalize (scale pixel values to [0, 1])
        img_array = np.array(img)
        img_array = img_array / 255.0  # Normalize image to [0, 1]

        # Add batch dimension (shape should be (1, 128, 128, 3))
        img_array = np.expand_dims(img_array, axis=0)

        # Print the shape of the image array to debug
        print("Image array shape:", img_array.shape)

        try:
            # Make prediction
            prediction = model.predict(img_array)
            print("Prediction:", prediction)  # Print raw prediction for debugging

            # Check the prediction output and decide if it's "Fake" or "Real"
            result = 'Real Product' if prediction[0][0] > 0.5 else 'Fake Product'
            print("Prediction result:", result)  # Print result for debugging

            # Render result page with prediction
            return render_template('result.html', filename=file.filename, result=result, prediction_value=prediction[0][0])

        except Exception as e:
            print("Error during prediction:", e)
            return render_template('error.html', error_message=str(e))

    return redirect(url_for('home'))

if __name__ == '__main__':
    app.run(debug=True)
