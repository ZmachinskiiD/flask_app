from flask import Flask,request, render_template, session
from flask_uploads import UploadSet, configure_uploads, IMAGES
import infer_on_single_image as code_base
import os
app = Flask(__name__)
app.secret_key = os.urandom(24)
photos = UploadSet('photos', IMAGES)
app.config['UPLOADED_PHOTOS_DEST'] = 'static/temp'
configure_uploads(app, photos)
upload_filename = ""

# Storing models to reduce load time
model_oxford = code_base.getModel() # FROM INFER_ON_SINGLE_IMAGE
model_paris = code_base.getModel(weights_file="./static/weights/paris_final.pth")

# Main page
@app.route("/", methods=['GET', 'POST'])
def index():
    
    return render_template("main.html") 

# Prediction page
@app.route("/ImgSelected/upload/", methods=['GET', 'POST'])
def evaluateNew():
    if request.method == 'POST' and 'photo'in request.files:
        filename = photos.save(request.files['photo'])
        filename = '/static/temp/'+filename

        directory_name=request.form.get('directory')
        directory_name='./static/data/'+request.form.get('directory')+'/'
        model_name='./static/weights/'+request.form.get('model')+'.pth'
        model_names=code_base.getModel(weights_file=model_name)
        print(model_name)
        similar_images = code_base.inference_on_single_labelled_image_pca_web_original(model_names,filename,directory_name)
        gt = [0]*60
        prev_evaluated_images = similar_images
        session['prev_evaluated_images'] = prev_evaluated_images
        return render_template("img_selected.html", filename = filename, evaluated=prev_evaluated_images,gt=gt, valid=0)
    return render_template("img_selected.html", filename = "")

if __name__ == "__main__":
    app.run(debug=True)