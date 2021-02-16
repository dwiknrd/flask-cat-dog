import os
from flask import Flask, request, redirect, url_for
from werkzeug.utils import secure_filename
import shutil

from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import ImageDataGenerator

app = Flask(__name__)
app.secret_key = b'secret'
BASE_PATH = os.path.dirname(os.path.abspath(__file__))
ALLOWED_EXTENSION = {'jpg', 'jpeg', 'png'}
def allowed_files(filename: str) -> bool:
    return '.' in filename and filename.rsplit('.',1)[1].lower() in ALLOWED_EXTENSION
@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        if 'file' not in request.files:
            return 'NO FILE'
        file = request.files.get('file')
        if file.filename == '' :
            return 'NO FILE SELECTED'
        if file and allowed_files(file.filename):
            shutil.rmtree(os.path.join(BASE_PATH, 'images/test/'))
            os.mkdir(os.path.join(BASE_PATH, 'images/test/'))
            filename = secure_filename(file.filename)
            file.save(os.path.join(BASE_PATH, 'images/test', filename))

            #Machine Learning Model
            new_model = load_model('cnn_model_cat_dogs.h5')
            IMG_HEIGHT = 150
            IMG_WIDTH = 150
            Image_Size = (IMG_WIDTH, IMG_HEIGHT)
            test_image_generator = ImageDataGenerator(rescale = 1./255)
            test_data_gen = test_image_generator.flow_from_directory('images',classes=['test'], target_size=Image_Size, batch_size=1, class_mode=None, shuffle=False)
            probabilities = new_model.predict(test_data_gen)
            probabilities = probabilities[0][0]

            if probabilities > 0.5:
                return {
                    "probability": round(probabilities*100, 2),
                    "class": "DOG"
                }
            else:
                return {
                    "probability": round((1-probabilities)*100, 2),
                    "class": "CAT"
                }



            return 'FILE UPLOADED SUCCESSFULLY'
    return 'HELLO WORLD NIH'
if __name__ == '__main__':
    app.run(debug=True)