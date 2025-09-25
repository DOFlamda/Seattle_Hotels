import re
from data import get_data
from desc import get_top_n_words
import matplotlib.pyplot as plt
import numpy as np

import nltk
from nltk.corpus import stopwords

# 确保下载了停用词数据
try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')

# 设置中文字体支持
plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号

#文本处理
def clean_text(text):
    REPLACE_BY_SPACE_RE = re.compile('[/(){}\[\]\|@,;]')
    BAD_SYMBOLS_RE = re.compile('[^0-9a-z #+_]')
    STOPWORDS = set(stopwords.words('english'))
    text = text.lower()
    text = REPLACE_BY_SPACE_RE.sub(' ', text) 
    text = BAD_SYMBOLS_RE.sub('', text) 
    text = ' '.join(word for word in text.split() if word not in STOPWORDS) 
    return text


if __name__=="__main__":
    dt=get_data()
    common_words = get_top_n_words(dt['desc'], 20)

    dt['word_count'] = dt['desc'].apply(lambda x: len(str(x).split()))

    # 使用matplotlib绘制直方图
    plt.figure(figsize=(10, 6))
    plt.hist(dt['word_count'], bins=50, color='skyblue', edgecolor='black', alpha=0.7)
    # 添加标题和标签
    plt.title('Word Count Distribution in Hotel Description', fontsize=15)
    plt.xlabel('Word Count', fontsize=12)
    plt.ylabel('Count', fontsize=12)
    # 添加网格线
    plt.grid(True, linestyle='--', alpha=0.7)
    # 调整布局
    plt.tight_layout()
    # 显示图表
    plt.show()

    
    dt['desc_clean'] = dt['desc'].apply(clean_text)
    print(dt['desc_clean'][0])