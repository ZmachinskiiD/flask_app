import os

from flask import Flask, request, render_template, session
from flask_uploads import UploadSet, configure_uploads, IMAGES

import infer_on_single_image as code_base
from un_utils import gen_opt
from un_createdb import prepare_model
from un_image_similarity import similar_images

app = Flask(__name__)
app.secret_key = os.urandom(24)
photos = UploadSet('photos', IMAGES)
app.config['UPLOADED_PHOTOS_DEST'] = 'static/temp'
configure_uploads(app, photos)
upload_filename = ""

model = prepare_model('three_view_long_share_d0.75_256_s1_google', gen_opt())

@app.route("/", methods=['GET', 'POST'])
def index():
    return render_template("main.html")

# Prediction page
@app.route("/ImgSelected/upload/", methods=['GET', 'POST'])
def evaluateNew():
    if request.method == 'POST' and 'photo' in request.files:
        filename = photos.save(request.files['photo'])
        filename = '/static/temp/'+filename
        similar_images_ = similar_images(model, filename)['0']
        similar_images_ = [similar_images_[i][0] for i in range(len(similar_images_))]
        for i in range(len(similar_images_)):
            similar_images_[i] = similar_images_[i].strip()
            similar_images_[i] = '../../' + similar_images_[i]
        print(similar_images_)
        gt = [0]*60
        prev_evaluated_images = similar_images_
        session['prev_evaluated_images'] = prev_evaluated_images
        return render_template("img_selected.html", filename = filename, evaluated=prev_evaluated_images,gt=gt, valid=0)
    return render_template("img_selected.html", filename = "")

if __name__ == "__main__":
    app.run(debug=True)