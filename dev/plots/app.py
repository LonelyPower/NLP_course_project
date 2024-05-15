# app.py

# 在 Flask 应用程序中导入你的 main.py 文件
from flask import Flask, request,jsonify
from flask import render_template
from main import init, wordcloud, show_bar, classify, text_recm, cluster,ner
from matplotlib.font_manager import FontProperties
# from flask_cors import CORS

app = Flask(__name__)

font = r"C:\Windows\Fonts\msyh.ttc"
chinese_font = FontProperties(fname=font)
bert_path = r"H:\My Code\nlp\nlp_course_project\pretrained_weight\bert-base-chinese"
data_path = r"H:\大二\冬季\创新实训2\THUCNews\THUCNews"


# 定义一个路由来处理请求，并调用 main.py 中的函数返回结果
@app.route("/init")
def run_main_function():
    init(data_path, bert_path)
    path=cluster()
    return render_template('homepage.html')
    # return {"stat": "success"}

@app.route("/homepage")
def fun0():
    return render_template('homepage.html')

@app.route("/app/wordcloud", methods=["GET","POST"])
def fun1():
    cls = request.args.get("name")
    path = wordcloud(cls,font,chinese_font)
    # path = r"D:\JetBrains\wenjian\pythonProject\NLP_course_project-main\dev\plots\wordcloud.png"
    return {"stat": "success", "age": path}


@app.route("/app/bar", methods=["GET"])
def fun2():
    cls = request.args.get("name")
    path = show_bar(cls,chinese_font)
    # path = r"D:\JetBrains\wenjian\pythonProject\NLP_course_project-main\dev\plots\bar.png"
    return {"stat": "success", "age": path}

@app.route("/app/cluster")
def fun3():
    path=cluster()
    return {"stat": "success", "cls": path}

@app.route("/app/classify")
def fun4():
    text = request.args.get("text")
    cls=classify(text)
    return render_template("homepage.html", text=cls)

@app.route("/app/recm", methods=["GET"])
def fun5():
    text = request.args.get("text")
    num = request.args.get("num")
    cls,res = text_recm(text, num)
    tags= ner(res[0])
    return render_template("newslist.html", text=res,cls=cls,tags=tags)


if __name__ == "__main__":
    app.run(debug=True)
