from flask import Flask, json, request, jsonify, redirect, abort, render_template_string
from flask.helpers import url_for
from flask.templating import render_template
from flask.wrappers import Response
from werkzeug.utils import secure_filename
import os
# http://127.0.0.1/analysis
from test import main_test

app = Flask("my-app", static_folder="static", template_folder="templates")

# 文件上传目录
app.config['UPLOAD_FOLDER'] = 'static/uploads/'
# 支持的文件格式
app.config['ALLOWED_EXTENSIONS'] = {'png', 'jpg', 'jpeg', 'gif'}  # 集合类型




@app.errorhandler(401)
def page_unauthorized(error):
    return render_template_string('<h1> Unauthorized </h1><h2>{{ error_info }}</h2>', error_info=error), 401


@app.route('/')
def index():  # 网站渲染
    return render_template('index.html')

@app.route('/index')
def home():
    # return render_template('index.html')
    return index()

@app.route('/author')
def author():  # 网站渲染
    return render_template('author.html')

@app.route('/mailAnalysis', methods=['POST', 'Get'])
def mailAnalysis():
    info = request.args.get('content')
    with open(r"THUCNews\data\test1.txt", "w", encoding='utf-8') as f:
        f.write(info)  # 自带文件关闭功能，不需要再写f.close()
    current_model = 'TextCNN'
    test_type,test_probabilty = main_test(current_model)
    #result = {'type': 0, 'msg': 'acc：' + str(test_acc)}
    result = {'type': 0, 'msg': '1、最可能的新闻类别：' + str(test_type)+"\n2、"+current_model+"模型预测得到是该类别的概率："+str(test_probabilty)}
    return jsonify(result)


@app.route('/analysis', methods=['POST', 'Get'])
def analysis():
    return render_template('analysis.html', page_title='新闻文本分类演示及测试系统')


if __name__ == '__main__':
    # app.run(debug=True)
    app.run( port=80, debug=True)
