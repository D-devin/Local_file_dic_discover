## build database
import os
import jieba
import logging
import requests
from gensim import corpora

from zhon.hanzi import punctuation
import re

# 配置日志
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

class ChineseCorpusBuilder:
    def __init__(self):
        self.stopwords = self.load_stopwords()
        jieba.initialize()  # 初始化jieba分词

    def load_stopwords(self):
        """加载中文停用词表"""
        stopword_url = "https://raw.githubusercontent.com/goto456/stopwords/master/baidu_stopwords.txt"
        try:
            response = requests.get(stopword_url)
            response.encoding = 'utf-8'
            return set(response.text.splitlines())
        except:
            # 本地备用停用词
            return set(['的', '了', '在', '是', '我', '有', '和', '就'])


    def clean_text(self, text):
        """文本清洗"""
        # 移除特殊字符
        text = re.sub(f"[{punctuation}\s+\.\!\/_,$%^*()?;；:-【】+\"\']+|[+——！，。？、~@#￥%……&*（）]+", " ", text)
        # 移除数字
        text = re.sub(r'\d+', '', text)
        # 转小写
        return text.lower()

    def tokenize(self, text):
        """中文分词处理"""
        text = self.traditional_to_simple(text)
        text = self.clean_text(text)
        words = jieba.lcut(text)
        return [word for word in words if word not in self.stopwords and len(word) > 1]

    def load_nlp_corpus(self):
        """加载预训练语料"""
        # 示例：使用gensim自带的维基百科语料（需要提前下载）
        from gensim.corpora import WikiCorpus
        wiki = WikiCorpus('zhwiki-latest-pages-articles.xml.bz2', lemmatize=False)
        return (self.tokenize(' '.join(article)) for article in wiki.get_texts())

    def load_local_files(self, folder_path):
        """加载本地文本文件"""
        corpus = []
        valid_extensions = ['.txt', '.csv', '.md']
        
        for root, _, files in os.walk(folder_path):
            for file in files:
                if any(file.endswith(ext) for ext in valid_extensions):
                    file_path = os.path.join(root, file)
                    try:
                        with open(file_path, 'r', encoding='utf-8') as f:
                            content = f.read()
                            corpus.append(self.tokenize(content))
                    except Exception as e:
                        print(f"文件读取失败 {file_path}: {str(e)}")
        return corpus

    def build_corpus(self, sources):
        """构建混合语料库"""
        corpus = []
        # 添加维基百科语料
        if 'wiki' in sources:
            corpus.extend(self.load_nlp_corpus())
        # 添加本地文件
        #if 'local' in sources:
            local_folder = "./chinese_docs"  # 修改为实际路径
            #corpus.extend(self.load_local_files(local_folder))
        return corpus

    def save_corpus(self, corpus, save_path):
        """保存处理后的语料"""
        with open(os.path.join(save_path, 'processed_corpus.txt'), 'w', encoding='utf-8') as f:
            for doc in corpus:
                f.write(' '.join(doc) + '\n')

# 使用示例
if __name__ == "__main__":
    builder = ChineseCorpusBuilder()
    
    # 构建语料库（指定数据源）
    mixed_corpus = builder.build_corpus(['wiki', 'local'])
    
    # 创建词典
    dictionary = corpora.Dictionary(mixed_corpus)
    dictionary.filter_extremes(no_below=5, no_above=0.5)
    
    # 生成词袋模型
    corpus_bow = [dictionary.doc2bow(doc) for doc in mixed_corpus]
    
    # 保存语料
    builder.save_corpus(mixed_corpus, './corpus_data')
    dictionary.save('chinese_corpus.dict')
    corpora.MmCorpus.serialize('chinese_corpus.mm', corpus_bow)