from flask import Flask
from flask import render_template
from flask import request, jsonify
from newfunc import *

app = Flask(__name__)


@app.route('/hnzl_calc/', methods=["POST"])
def HNZL_calc():
    if request.method == "POST":
        model = request.values.get('model')
        val = eval(request.values.get('val'))
        prob = hnzl_predict(model, val)
        return jsonify({"res":prob})

@app.route('/A_predict', methods=["GET"])
def A_predict():
    return render_template('Agroup.html')

@app.route('/B_predict', methods=["GET"])
def B_predict():
    return render_template('Bgroup.html')

@app.route('/subA1_predict', methods=["GET"])
def subA1_predict():
    return render_template('subA1group.html')

@app.route('/subA2_predict', methods=["GET"])
def subA2_predict():
    return render_template('subA2group.html')

@app.route('/index', methods=["GET"])
def front_page():
    return render_template('index.html')

# @app.route('/cancer_predict', methods=["GET"])
# def HNZL_cancer_predict():
#     return render_template('cancer_lr.html')


if __name__ == '__main__':
    app.run(
        host='0.0.0.0',
        port=8086
    )
