import json
import base64

from api import im2latex 
from flask import Flask, request, render_template

app = Flask(__name__, template_folder='templates')

#----------------Im2LaTeX-----------------
@app.route('/im2latex/', methods=['POST'])
def im2latex_api():
    img = request.files["image"].read()
    img_b64 = base64.b64encode(img)
    result = im2latex(img_b64) 
    return json.dumps(result, ensure_ascii=False)

#----------------Im2LaTeX Demo-----------------
@app.route('/im2latex/', methods=['GET'])
def im2latex_demo():
    return render_template("./index.html")