from sklearn.feature_extraction.text import CountVectorizer
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
# 设置中文字体支持
plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号
pd.options.display.max_columns = 30

def bag_of_words(dt):
    vec=CountVectorizer().fit(dt["desc"])
    bw=vec.transform(dt["desc"])
    #print(bw.toarray()) #[[0 1 0 ... 0 0 0]。。。。。
    #print(bw.shape) #(152, 3200) 152句话，3200个不重复的词

    # for word in vec.vocabulary_.items():
    # print(word)

    sum_words=bw.sum(axis=0)
    #print(sum_words)
    words_freq = [(word, sum_words[0, idx]) for word, idx in vec.vocabulary_.items()]
    words_freq = sorted(words_freq, key=lambda x: x[1], reverse=True) #降序排序
    #print(words_freq)   #结果显示，the，and，of等频率最高的词汇 并无效力

def get_top_n_words(corpus, n=None):
    #vec = CountVectorizer(stop_words='english').fit(corpus) #先把the and of等词删除
    vec = CountVectorizer().fit(corpus)
    bag_of_words = vec.transform(corpus)
    sum_words = bag_of_words.sum(axis=0)
    words_freq = [(word, sum_words[0, idx]) for word, idx in vec.vocabulary_.items()]
    words_freq =sorted(words_freq, key = lambda x: x[1], reverse=True)
    return words_freq[:n]


if __name__=="__main__":
    from data import get_data
    dt = get_data()
    bag_of_words(dt)
    common_words = get_top_n_words(dt['desc'],40)
    print(common_words)
    df1 = pd.DataFrame(common_words, columns=['desc', 'count'])
    # 打印结果
    print(df1)
    
    # 使用matplotlib绘制条形图
    plt.figure(figsize=(12, 8))
    sorted_df = df1.groupby('desc').sum()['count'].sort_values(ascending=False)
    bars = plt.bar(sorted_df.index, sorted_df.values, color='skyblue')
    
    # 添加标题和标签
    plt.title('Top 40 words in hotel description before removing stop words', fontsize=15)
    plt.xlabel('Words', fontsize=12)
    plt.ylabel('Count', fontsize=12)
    
    # 旋转x轴标签以避免重叠
    plt.xticks(rotation=45, ha='right')
    
    # 添加数值标签
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height, '%d' % int(height),
                ha='center', va='bottom')
    
    # 调整布局
    plt.tight_layout()
    
    # 显示图表
    plt.show()
