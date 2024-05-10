# app.py

# 在 Flask 应用程序中导入你的 main.py 文件
from flask import Flask, request
from main import init, wordcloud, show_bar, classify, text_recm, cluster
from matplotlib.font_manager import FontProperties

app = Flask(__name__)

font = r"C:\Windows\Fonts\msyh.ttc"
chinese_font = FontProperties(fname=font)
bert_path = r"../../pretrained_weights/bert-base-chinese"
data_path = r"H:\大二\冬季\创新实训2\THUCNews\THUCNews"


# 定义一个路由来处理请求，并调用 main.py 中的函数返回结果
@app.route("/init")
def run_main_function():
    init(data_path, bert_path)
    return {"stat": "success"}


@app.route("/app/wordcloud", methods=["GET"])
def fun1():
    cls = request.args.get("name")
    path = wordcloud(cls,font,chinese_font)
    return {"stat": "success", "age": path}


@app.route("/app/bar", methods=["GET"])
def fun2():
    cls = request.args.get("name")
    path = show_bar(cls,chinese_font)
    return {"stat": "success", "age": path}

@app.route("/app/cluster")
def fun3():
    path=cluster()
    return {"stat": "success", "cls": path}

@app.route("/app/classify")
def fun4():
    text = request.args.get("text")
    cls=classify(text)
    return {"stat": "success", "cls": cls}

@app.route("/app/recm", methods=["GET"])
def fun5():
    text = request.args.get("text")
    num = request.args.get("num")
    res=text_recm(text,num)
    return {"stat": "success", "text": res}


if __name__ == "__main__":
    app.run(debug=True)
