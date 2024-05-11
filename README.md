# NLP_Course_Project

## 新闻文本分类及推荐系统

### 功能点及技术栈：

1. 基于已有语料库的词频统计与词云可视化：|||||||||| Function FINISHED
   - 技术栈：NLP库（NLTK），词云生成库（WordCloud），数据可视化库（如Matplotlib、Seaborn）

2. 文本分类： 基于BERT、LSTM、MultiHeadAttention： |||||||||| Training Finished
   - 技术栈：深度学习框架（PyTorch），预训练模型（BERT），文本处理工具（transformers库）

3. 文本推荐： 基于词向量匹配： 
   - 技术栈：词向量模型（GloVe），推荐算法（基于余弦相似度）

4. 命名实体识别及其可视化： 基于HanLP：
   - 技术栈：HanLP工具包，命名实体识别算法

5. 文本聚类及其可视化：
   - 技术栈：文本聚类算法（K-means），数据可视化库（Matplotlib、Seaborn、Plotly），文本处理工具（NLTK）

### 工作流程：

1. 进入系统之后，先行询问用户爱好，并进行相应初始推荐。

2. 设置文本分类模块，用户输入已有文本可进行相应分类推理。

3. 用户可选择是否显示当前模块的词云，以来大致感受该类别文章的特点关键词。

4. 设置文本推荐模块，用户输入已有文本可进行相应推荐。

5. 用户可在文本聚类模块中查找与自己当前文章相似的类型，并进行检索以查找相关信息。

6. 对于已有文本，用户点开之后自动进行类别描述与命名实体识别加上对应词汇标注展示。

# 需求
bert_naive_20240501174421.pth
nlp_course_project\pretrained_weight\bert-base-chinese\pytorch_model.bin
