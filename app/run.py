import os
import imghdr
from glob import glob

from flask import Flask
from flask import render_template, request, jsonify, redirect, url_for, abort
from werkzeug.utils import secure_filename

from dog_recognition import DogRecognizer

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 2048 * 2048
app.config['UPLOAD_EXTENSIONS'] = ['.jpg', '.png', '.gif']
app.config['UPLOAD_PATH'] = 'static/uploads'

# Initialize the model
recognizer = DogRecognizer()

def validate_image(stream):
    """
    Validate the input stream to be an image.

    Args:
    stream - input stream from an image

    Returns:
    image_type - string with the file extension in the list of valid extensions
    """
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
    """
    Main method for the Flask App

    Args:
    None

    Returns:
    Master.html as a rendered template
    """
    # render web page with the main page
    return render_template('master.html', ids=None, graphJSON=None)

@app.route('/go', methods=['POST','GET'])
def go():
    """
    Go method for the Flask App, to handle the file uploaded

    Args:
    stream - input stream from an image

    Returns:
    image_type - string with the file extension in the list of valid extensions
    """
    # save user input in query, for future use
    query = request.args.get('query', '') 

    # Read the file attached in the request
    uploaded_file = request.files['file']
    filename = secure_filename(uploaded_file.filename)
    if filename != '':
        file_ext = os.path.splitext(filename)[1]
        if file_ext not in app.config['UPLOAD_EXTENSIONS'] or \
                file_ext != validate_image(uploaded_file.stream):
            abort(400)
        uploaded_file.save(os.path.join(app.config['UPLOAD_PATH'], filename))

    # Once the file has been uploaded and saved in the path, we call the predict engine and get in return the breed, certainty, if it was a human 
    # and a list of sample images
    breed, certainty, human, sample_images = recognizer.predict_breed(os.path.join(app.config['UPLOAD_PATH'], filename))

    # This will render the go.html, and are passing the prediction values, query and received file for rendering
    return render_template(
        'go.html',
        query=query,
        breed=breed,
        certainty=certainty,
        human=human,
        received_file=os.path.join(app.config['UPLOAD_PATH'], filename),
        sample_images=sample_images[0]
    )

def main():
    """
    Main application method

    Args:
    None

    Returns:
    None
    """
    app.run(host='0.0.0.0', port=3001, debug=True)


if __name__ == '__main__':
    main()