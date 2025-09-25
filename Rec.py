#python3.11 SHR-ven
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel

#导入数据
from data import get_data
dt=get_data()

#统计desc中词汇频率并清理
from desc import get_top_n_words
common_words = get_top_n_words(dt['desc'], 20)
df1 = pd.DataFrame(common_words, columns=['desc', 'count'])
dt['word_count'] = dt['desc'].apply(lambda x: len(str(x).split()))

from text import clean_text
dt['desc'] = dt['desc'].apply(lambda x: clean_text(x))

def print_description(index):
    example = dt[dt.index == index][['desc', 'name']].values[0]
    if len(example) > 0:
        print(example[0])
        print('Name:', example[1])

dt.set_index('name', inplace = True)
tf = TfidfVectorizer(analyzer='word', ngram_range=(1, 3), min_df=0.0, stop_words='english')
tfidf_matrix = tf.fit_transform(dt['desc'])
cosine_similarities = linear_kernel(tfidf_matrix, tfidf_matrix)
indices = pd.Series(dt.index)
#print(indices[:50])

def recommendations(name, cosine_similarities = cosine_similarities):
    recommended_hotels = []
    idx = indices[indices == name].index[0]
    score_series = pd.Series(cosine_similarities[idx]).sort_values(ascending = False)
    top_10_indexes = list(score_series.iloc[1:11].index)    
    for i in top_10_indexes:
        recommended_hotels.append(list(dt.index)[i])   
    return recommended_hotels
#以二者为例
print(recommendations('Hilton Seattle Airport & Conference Center'))
recommendations("The Bacon Mansion Bed and Breakfast")

#基于文本相似性的酒店推荐：
#1.导入数据，数据由酒店名、地址、描述*组成，描述是未经处理的自然文本
#2.统计停用词：记录描述中没有价值的信息-a、the等/出现频次最高的一些无价值词汇。（现实通常要人工筛选）
#3.清楚描述中的停用词
#4.将每家酒店的干净描述文本转换为一个数学向量（tfidf_matrix），不仅考虑词频，还考虑了逆文档频率（即一个词在所有文档中的常见程度）
#5.推荐函数：输入酒店名，然后在相似度矩阵中找到该酒店与数据集中所有酒店的相似度分数，将这些相似度分数按从高到低排序。
#排除与自身相似的第一项（相似度为1.0），然后返回相似度排名前10的酒店名称列表。
