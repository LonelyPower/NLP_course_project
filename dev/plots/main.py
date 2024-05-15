import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(os.getcwd()))))
import json
import random
import torch
import argparse
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from tqdm import tqdm
from wordcloud import WordCloud, STOPWORDS
from collections import defaultdict
sys.path.insert(0,r'../')
from utils import read_and_sample_single_folder
from transformers import BertTokenizer, BertModel
from sklearn.cluster import KMeans
from sklearn.manifold import TSNE
from sklearn.metrics import silhouette_score
sys.path.insert(0, r"../../")
from model import Bert_Naive
from hanlp_restful import HanLPClient

def init(data_path, bert_path):
    global label_dict
    global data_cls
    global rec_map
    global bert_model
    global bert_tokenizer
    global class_indices
    global text_vectors1
    global net

    bert_tokenizer = BertTokenizer.from_pretrained(bert_path)
    bert_model = BertModel.from_pretrained(bert_path)

    # # //////////////////////////////////////////////////////////////////
    try:
        with open(os.path.join("../class_indices.json"), "r") as f:
            class_indices = {v: k for k, v in json.load(f).items()}
        print(class_indices)
    except Exception as e:
        print("An error occurred:", e)
    label_dict = {v: k for k, v in class_indices.items()}
    print(label_dict)


    # # //////////////////////////////////////////////////////////////////

    # # return
    random.seed(114514)
    data_path, data_label = list(
        read_and_sample_single_folder(data_path, sample_rate=0.001)
    )  # 先采样0.001的数据进行后续测试
    data_cls = defaultdict(list)
    for path, label in tqdm(zip(data_path, data_label)):
        with open(path, "r", encoding="utf-8") as f:
            data = f.readlines()
        temp = ""
        for line in data:
            temp += line
        data_cls[label].append(temp)
    print(data_cls)

    # # //////////////////////////////////////////////////////////////////

    # # //////////////////////////////////////////////////////////////////
    rec_map = {
        bert_model.embeddings(
            bert_tokenizer(
                text=class_indices[k],
                max_length=512,
                padding="max_length",
                truncation=True,
                return_tensors="pt",
            )["input_ids"]
        ): k
        for k in data_cls.keys()
    }
    # # //////////////////////////////////////////////////////////////////

    data_cls.keys()
    text_vectors1 = []
    for cls in tqdm(data_cls.keys()):
        texts = data_cls[cls]
        cls_vectors = np.array(
            [
                np.mean(
                    bert_model.embeddings(
                        bert_tokenizer(
                            text=text,
                            max_length=512,
                            padding="max_length",
                            truncation=True,
                            return_tensors="pt",
                        )["input_ids"]
                    )
                    .squeeze(0)
                    .detach()
                    .numpy(),
                    axis=0,
                )
                for text in texts
            ]
        )
        text_vectors1.extend(cls_vectors)

    # # # //////////////////////////////////////////////////////////////////

    args = argparse.Namespace(
        max_length=512,
        num_classes=14,
        batch_size=8,
        num_epochs=100,
        lr=0.00001,
        weight_decay=0.01,
        input_size=768,
        hidden_size=256,
        num_layers=8,
        num_heads=8,
        dropout_rate=0.1,
        bidirectional=True,
    )
    net = Bert_Naive(bert=bert_model, args=args)
    # # net.load_state_dict(torch.load(r'H:\My Code\nlp\nlp_course_project-master\news_cls_recm\bert_naive_20240501174421.pth'))
    net.load_state_dict(
        torch.load(
            r"../../bert_naive_20240501174421.pth",
            map_location="cpu",
        )
    )


def wordcloud(tgt_cls,font,chinese_font):
    # tgt_cls = "体育"
    wc = WordCloud(
        background_color="white",  # 背景设置为白色
        font_path=font,  # 字体
        max_words=200,  # 最大显示的关键词数量
        stopwords=STOPWORDS,  # 使用上面导入停用词表
        max_font_size=250,  # 最大字体
        random_state=30,  # 设置随机状态数，及配色的方案数
        height=860,  # 如果使用默认图片，则可以设置高
        margin=2,  # 图片属性
        width=1000,  # 设置宽
    ).generate("".join(data_cls[label_dict[tgt_cls]]))
    plt.figure(figsize=(20, 20), dpi=300)
    plt.imshow(wc, interpolation="bilinear")
    plt.axis("off")
    plt.title(f"{tgt_cls}词云图", fontsize=100, font=chinese_font)
    # plt.show()
    plt.savefig(r"H:\My Code\nlp\nlp_course_project\dev\plots\static\img\wordcloud.png")
    print("a")
    return os.getcwd() + "\\wordcloud.png"


def show_bar(tgt_cls,chinese_font):
    # # 初始化 defaultdict 用于存储长度计数
    data_length = defaultdict(lambda: defaultdict(int))
    for label, data in data_cls.items():
        for text in data:
            data_length[label][len(text)] += 1  # 更新对应label和长度的计数器

    tgt_dict = data_length[label_dict[tgt_cls]]
    tgt_df = pd.DataFrame(tgt_dict.keys(), index=tgt_dict.values())
    fig, ax = plt.subplots(1, 1, figsize=(20, 10))
    sns.histplot(tgt_df, ax=ax, color="teal", kde=True)
    sns.kdeplot(tgt_df, color="black", ax=ax)
    plt.title(f"{tgt_cls} 文本长度分布")

    plt.title(f"{tgt_cls} 文本长度分布", font=chinese_font)
    plt.xlabel("文本长度")
    # plt.show()
    plt.savefig(r"H:\My Code\nlp\nlp_course_project\dev\plots\static\img\bar.jpg")
    return os.getcwd() + "\\bar.png"

def ner(text):
    HanLP = HanLPClient("https://hanlp.hankcs.com/api",auth="NDczNUBiYnMuaGFubHAuY29tOkVWVFM4MVBnSUNDcjhpTUE=",language="zh",)
    a=HanLP(text, tasks="ner*")["ner/msra"]
    f={}
    r=[]
    res={'PERSON':[],'ORGANIZATION':[],'LOCATION':[],'DATE':[]}
    for i in a:
        for j in i:
            if j[0] not in f:
                f[j[0]]=j[1]
    for key, value in f.items():
        r.append([key, value])
    for i in r:
        if i[1]=='PERSON':
            res['PERSON'].append(i[0])
        elif i[1]=='ORGANIZATION':
            res['ORGANIZATION'].append(i[0])
        elif i[1]=='LOCATION':
            res['LOCATION'].append(i[0])
        elif i[1]=='DATE':
            res['DATE'].append(i[0])
    print(res)
    return res
def text_recm(input_,num=10):
    # input_ = "篮球比赛"
    input_embedding = bert_model.embeddings(
        bert_tokenizer(
            text=input_,
            max_length=512,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )["input_ids"]
    )

    similarities = {}
    for recommend_vector, cls in rec_map.items():
        similarity = torch.nn.functional.cosine_similarity(
            recommend_vector[0], input_embedding[0]
        )
        similarities[cls] = torch.mean(similarity)

    sorted_recommendations = sorted(
        similarities.items(), key=lambda x: x[1].item(), reverse=True
    )
    data_cls
    top_recommendation = sorted_recommendations[0]

    # print("最相似的推荐内容:", )
    # print("相似度:", top_recommendation[1].item())
    # print("推荐内容:\n", *data_cls[top_recommendation[0]][:10])
    # print([index[0] for index in sorted_recommendations[:5]])
    return class_indices[top_recommendation[0]],list(data_cls[top_recommendation[0]][:num])


def cluster():
    plt.rcParams["font.sans-serif"] = ["SimHei"]  ###解决中文乱码
    plt.rcParams["axes.unicode_minus"] = False

    text_vectors = np.array(text_vectors1)
    # 现在可以使用K-means进行聚类
    num_clusters = len(data_cls)  # 假设聚成5类
    kmeans = KMeans(n_clusters=num_clusters, random_state=0)
    cluster_labels = kmeans.fit_predict(text_vectors)

    # 使用t-SNE进行降维可视化
    tsne = TSNE(n_components=2, random_state=0)
    text_tsne = tsne.fit_transform(text_vectors)
    color_map = [
        "salmon",
        "burlywood",
        "darkseagreen",
        "lightseagreen",
        "coral",
        "steelblue",
        "teal",
        "darkviolet",
        "crimson",
        "tomato",
        "orangered",
        "slategray",
        "slateblue",
        "darkorange",
        "gold",
        "yellowgreen",
        "olivedrab",
        "darkgoldenrod",
        "sienna",
        "peru",
        "chocolate",
        "saddlebrown",
        "firebrick",
        "maroon",
        "lightsalmon",
        "lightcoral",
        "indianred",
        "rosybrown",
        "sandybrown",
    ]

    # 绘制聚类可视化结果
    plt.figure(figsize=(10, 8))
    for i in range(num_clusters // 2):
        cluster_points = text_tsne[cluster_labels == i]
        plt.scatter(
            cluster_points[:, 0], cluster_points[:, 1], label=f"{class_indices[i]}"
        )
    plt.title("Text Clustering Visualization")
    plt.legend()
    # plt.show()
    plt.savefig("cluster1.png")

    # 绘制聚类可视化结果
    plt.figure(figsize=(10, 8))
    for i in range(num_clusters // 2, num_clusters):
        cluster_points = text_tsne[cluster_labels == i]
        plt.scatter(
            cluster_points[:, 0], cluster_points[:, 1], label=f"{class_indices[i]}"
        )
    plt.title("Text Clustering Visualization")
    plt.legend()
    # plt.show()
    plt.savefig("cluster2.png")

    # 绘制聚类可视化结果
    plt.figure(figsize=(10, 8))
    for i in range(num_clusters):
        cluster_points = text_tsne[cluster_labels == i]
        plt.scatter(
            cluster_points[:, 0],
            cluster_points[:, 1],
            label=f"{class_indices[i]}",
            c=color_map[i],
        )
    plt.title("Text Clustering Visualization")
    plt.legend()
    # plt.show()
    plt.savefig("cluster3.png")

    # 计算  轮廓系数
    silhouette_avg = silhouette_score(text_vectors, cluster_labels)
    print(f"The average silhouette_score is : {silhouette_avg}")
    
    return [os.getcwd() + "\\cluster1.png", os.getcwd() + "\\cluster2.png", os.getcwd() + "\\cluster3.png"]


def classify(text):
    tokens = bert_tokenizer(text)
    input_ids, attention_mask = torch.tensor(tokens["input_ids"]).unsqueeze(
        0
    ), torch.tensor(tokens["attention_mask"]).unsqueeze(0)
    print("文本:", text)
    print(
        "类别:",
        class_indices[
            torch.argmax(
                torch.nn.functional.softmax(net(input_ids, attention_mask), dim=-1)
            ).item()
        ],
    )
    return str(
        class_indices[
            torch.argmax(
                torch.nn.functional.softmax(net(input_ids, attention_mask), dim=-1)
            ).item()
        ]
    )

# from matplotlib.font_manager import FontProperties
# font = r"C:\Windows\Fonts\msyh.ttc"
# chinese_font = FontProperties(fname=font)
# bert_path = r"H:\My Code\nlp\nlp_course_project\pretrained_weight\bert-base-chinese"
# data_path = r"H:\大二\冬季\创新实训2\THUCNews\THUCNews"
# init(data_path, bert_path)
# show_bar("体育",chinese_font)
# wordcloud("体育",font,chinese_font)