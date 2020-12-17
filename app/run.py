import json
import os
import imghdr
from glob import glob

from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize

from flask import Flask
from flask import render_template, request, jsonify, redirect, url_for, abort
#from sklearn.externals import joblib
from werkzeug.utils import secure_filename
from sqlalchemy import create_engine

from dog_recognition import DogRecognizer

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 2048 * 2048
app.config['UPLOAD_EXTENSIONS'] = ['.jpg', '.png', '.gif']
app.config['UPLOAD_PATH'] = 'static/uploads'

recognizer = DogRecognizer()

def validate_image(stream):
    header = stream.read(512)  # 512 bytes should be enough for a header check
    stream.seek(0)  # reset stream pointer
    format = imghdr.what(None, header)
    if not format:
        print('Image not in format')
        return None
    return '.' + (format if format != 'jpeg' else 'jpg')


# index webpage displays cool visuals and receives user input text for model
@app.route('/')
@app.route('/index')
def index():
    
    # render web page with plotly graphs
    return render_template('master.html', ids=None, graphJSON=None)


# web page that handles user query and displays model results
@app.route('/go', methods=['POST','GET'])
def go():
    # save user input in query
    query = request.args.get('query', '') 

    uploaded_file = request.files['file']
    filename = secure_filename(uploaded_file.filename)
    if filename != '':
        file_ext = os.path.splitext(filename)[1]
        if file_ext not in app.config['UPLOAD_EXTENSIONS'] or \
                file_ext != validate_image(uploaded_file.stream):
            abort(400)
        uploaded_file.save(os.path.join(app.config['UPLOAD_PATH'], filename))

    breed, certainty, human, sample_images = recognizer.predict_breed(os.path.join(app.config['UPLOAD_PATH'], filename))

    print(sample_images)

    # This will render the go.html Please see that file. 
    return render_template(
        'go.html',
        query=query,
        breed=breed,
        certainty=certainty,
        human=human,
        received_file=filename,
        sample_images=sample_images[0]
    )

@app.route('/', methods=['POST'])
@app.route('/index', methods=['POST'])
def upload_files():
    uploaded_file = request.files['file']
    filename = secure_filename(uploaded_file.filename)
    if filename != '':
        file_ext = os.path.splitext(filename)[1]
        if file_ext not in app.config['UPLOAD_EXTENSIONS'] or \
                file_ext != validate_image(uploaded_file.stream):
            abort(400)
        uploaded_file.save(os.path.join(app.config['UPLOAD_PATH'], filename))
    return redirect(url_for('index'))

def main():
    app.run(host='0.0.0.0', port=3001, debug=True)


if __name__ == '__main__':
    main()