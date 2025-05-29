import os 
import json 
import logging
import requests
import numpy as np 
import pandas as pd
import jieba 
import re
from typing import List, Dict, Any
from docx import Document
import fitz
from bs4 import BeautifulSoup
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud
from collections import Counter
import matplotlib
from flask import Flask, request, jsonify, send_file
from flask_cors import CORS
import traceback
from datetime import datetime
import uuid
matplotlib.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS', 'DejaVu Sans']
matplotlib.rcParams['axes.unicode_minus'] = False
from sklearn.metrics.pairwise import cosine_similarity

stopwords_path  = r'server/src/data/stopwords.txt'

## 读取传回来的json文件中的路径，并且将路径中的文件读出文本，
## 
class Get_file:
    def __init__(self, json_path: str):
        self.json_path = json_path
        open_json = open(self.json_path, 'r', encoding='utf-8') 
        #读取json文件里的路径
        self.json_dir = json.load(open_json)
        self.json_dir = self.json_dir['json_path']
        open_json.close()
        self.file_path = [] #文件路径
        self.file_name = [] #文件名
        self.kind = ['']
        self.all_conent =[] #全文件文本
        self.processed_files_content: Dict[str, str] = {} # 用于存储对应文件名对应的文件内容
        self.word_dict = {}# 词频统计
        self.inverted_index = {}# 倒排索引
        self.file_word_count = {}# 文件词频统计
        self.content_list = []
        self.total_docs = 0
        self.stopwords = self._load_stopwords()
    # 读取文本
    def read_content(self, kind: str,file_path: str) -> str:
        """读取纯文本文件 (.txt) 并返回其内容。"""
        if kind == 'txt':
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                return content
            except FileNotFoundError:
                return f"错误: 文件 '{file_path}' 未找到。"
            except Exception as e:
                return f"读取TXT文件时发生错误: {e}"
        elif kind == 'json':
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                # 将解析后的Python对象转换回格式化的JSON字符串
                content = json.dumps(data, ensure_ascii=False, indent=4)
                return content
            except FileNotFoundError:
                return f"错误: 文件 '{file_path}' 未找到。"
            except json.JSONDecodeError:
                return f"错误: 文件 '{file_path}' 不是有效的JSON格式。"
            except Exception as e:
                return f"读取JSON文件时发生错误: {e}"
        elif kind == 'csv':
            try:
                # encoding参数对于read_csv是有效的
                df = pd.read_csv(file_path, encoding='utf-8')
                content = df.to_string()
                return content
            except FileNotFoundError:
                return f"错误: 文件 '{file_path}' 未找到。"
            except pd.errors.EmptyDataError:
                return f"错误: CSV文件 '{file_path}' 为空。"
            except Exception as e:
                return f"读取CSV文件时发生错误: {e}"
        elif kind == 'xlsx' or kind == 'xls':
            try:
                # pd.read_excel 的 encoding 参数通常不直接使用，它会自动处理或通过io指定
                # sheet_name=None 读取所有工作表
                excel_data = pd.read_excel(file_path, sheet_name=None)
                all_sheets_content = []
                if isinstance(excel_data, dict): # 如果有多个工作表
                    for sheet_name, df in excel_data.items():
                        sheet_content = f"--- 内容来自工作表: {sheet_name} ---\n"
                        sheet_content += df.to_string()
                        all_sheets_content.append(sheet_content)
                else: # 如果只有一个工作表，excel_data直接是DataFrame
                    all_sheets_content.append(excel_data.to_string())
                return "\n\n".join(all_sheets_content)
            except FileNotFoundError:
                return f"错误: 文件 '{file_path}' 未找到。"
            except Exception as e:
                return f"读取Excel文件时发生错误: {e}"
        else :
            return f"错误: 未知或不支持的文件类型 '{kind}'"  # 返回错误信息而不是空值
    #洁词
    def process_text(self, content: list) -> List[str]:
        process_content = []
        # 每次循环都重新读取停用词文件，效率低下
        for text in content:
            text = re.sub(r'[^\u4e00-\u9fa5a-zA-Z0-9\s]', '', text)
            text = ''.join(jieba.cut(text))
            # 应该在类初始化时读取一次停用词
            stopwords = self._load_stopwords()  # 修改为调用方法
            text = ' '.join([word for word in text.split() if word not in stopwords])
            process_content.append(text)
        return process_content

    def _load_stopwords(self) -> List[str]:
        """加载停用词"""
        try:
            with open(stopwords_path, 'r', encoding='utf-8') as f:
                return [word.strip() for word in f.readlines()]
        except FileNotFoundError:
            return ['的', '了', '在', '是', '我', '有', '和', '就', '不', '人', '都', '一', '一个']

    #统计词频
    def all_word_count(self):
        word_dict = {}
        for file_name, content in self.processed_files_content.items():
            if content.startswith("错误:"):
                continue
            process_content = self.process_text([content])
            if process_content:
                word_list = process_content[0].split()
                for word in word_list:
                    word_dict[word] = word_dict.get(word, 0) + 1
        sorted_word_dict = sorted(word_dict.items(), key=lambda x: x[1], reverse=True)
        self.word_dict = {word: count for word, count in sorted_word_dict[:100]}  # 只保留前100个高频词
        return self.word_dict

    #构建倒索排序
    def build_inverted_index(self) -> None:
        self.inverted_index = {}
        self.file_word_count = {}
        self.total_docs = 0
        for file_name, content in self.processed_files_content.items():
            if content.startswith("错误:"):
                continue
            processed_content = self.process_text([content])
            if not processed_content:
                continue
            words = processed_content[0].split()
            self.total_docs += 1  # 记录文档总数
            # 文档级词频统计
            doc_word_count = {}
            for pos, word in enumerate(words):
                if not word:
                    continue
                # 更新文档词频
                doc_word_count[word] = doc_word_count.get(word, 0) + 1
                # 初始化倒排索引项
                if word not in self.inverted_index:
                    self.inverted_index[word] = {
                        'doc_freq': 0,  # 包含该词的文档数
                        'postings': {}   # 文档ID: {频率, 位置}
                    }
                # 初始化文档项
                if file_name not in self.inverted_index[word]['postings']:
                    self.inverted_index[word]['postings'][file_name] = {
                        'freq': 0,
                        'positions': []
                    }
                    self.inverted_index[word]['doc_freq'] += 1  # 更新文档频率
                # 更新词项信息
                posting = self.inverted_index[word]['postings'][file_name]
                posting['freq'] += 1
                posting['positions'].append(pos)  # 记录位置信息
            
            self.file_word_count[file_name] = doc_word_count

        #读文件   
    def read_dir_file(self):
        for root, dirs, files in os.walk(self.json_dir):
            for file_name in files: # 使用 file_name 避免与外部可能的 file 变量冲突
                self.file_name.append(file_name)
                self.file_path.append(os.path.join(root,file_name))
        for file_path_item in self.file_path: # 遍历收集到的文件路径
                content_or_error_msg = f"错误: 未知或不支持的文件类型 '{os.path.basename(file_path_item)}'." # 默认消息
                file_name_extension = os.path.splitext(file_path_item)[1][1:]
                content_or_error_msg = self.read_content(file_name_extension, file_path_item)
                self.processed_files_content[file_path_item] = content_or_error_msg
                self.content_list.append(content_or_error_msg)
    def init_read_dir(self):
        self.read_dir_file()
        self.all_word_count()
        self.build_inverted_index()


class scearch_content:
    def __init__(self):
        """初始化搜索类"""
        self.stopwords = self._load_stopwords()   
    def _load_stopwords(self) -> List[str]:
        """加载停用词"""
        try:
            with open(stopwords_path, 'r', encoding='utf-8') as f:
                return [word.strip() for word in f.readlines()]
        except FileNotFoundError:
            return ['的', '了', '在', '是', '我', '有', '和', '就', '不', '人', '都', '一', '一个']
    def preprocess_query(self, query: str) -> List[str]:
        """预处理查询文本"""
        # 去除特殊字符，保留中文、英文、数字
        query = re.sub(r'[^\u4e00-\u9fa5a-zA-Z0-9\s]', '', query)
        # 分词
        words = list(jieba.cut(query))
        # 去停用词和空词
        words = [word.strip() for word in words 
                if word.strip() and word.strip() not in self.stopwords and len(word.strip()) > 1]
        return words
    def generate_content_preview(self, content: str, keyword: str, preview_length: int = 200) -> str:
        """生成包含关键词的内容预览"""
        if not content or not keyword:
            return "无预览内容"

        # 查找关键词位置
        keyword_pos = content.find(keyword)
        if keyword_pos == -1:
            # 如果没找到关键词，返回开头部分
            return content[:preview_length] + "..." if len(content) > preview_length else content

        # 计算预览开始位置
        start_pos = max(0, keyword_pos - preview_length // 2)
        end_pos = min(len(content), start_pos + preview_length)

        preview = content[start_pos:end_pos]

        # 添加省略号
        if start_pos > 0:
            preview = "..." + preview
        if end_pos < len(content):
            preview = preview + "..."
    
        return preview
    
    # 词频检索
    def tf_search(self, query: str, inverted_index: Dict[str, Any], processed_files_content: Dict[str, str], file_word_count: Dict[str, any]) -> Dict[str, Any]:
        """
        基于词频的检索（使用优化后的倒排索引）
        Args:
            query: 搜索查询词
        Returns:
            检索结果字典
        """
        if not query:
            return {'found': False, 'message': '查询词为空', 'results': []}

        query_words = self.preprocess_query(query)
        if not query_words:
            return {'found': False, 'message': '查询词无效', 'results': []}

        results = []
        seen_docs = set()  # 用于跟踪已处理的文档

        for word in query_words:
            if word not in inverted_index:
                continue
            
            term_info = inverted_index[word]
            postings = term_info['postings']

            sorted_docs = sorted(
                postings.items(), 
                key=lambda x: x[1]['freq'], 
                reverse=True
            )

            for file_name, data in sorted_docs:
                if file_name in seen_docs:
                    continue
                seen_docs.add(file_name)

                content = processed_files_content.get(file_name, '')
                if content.startswith("错误:"):
                    continue  
                
                preview = self.generate_content_preview(content, word)
                word_freq = data['freq']

                # 修复：计算文档总词数
                total_terms = sum(file_word_count.get(file_name, {}).values())
                if total_terms == 0:
                    total_terms = 1  # 避免除零错误

                tf_score = word_freq / total_terms

                results.append({
                    'file_name': file_name,
                    'matched_word': word,
                    'word_frequency': word_freq,
                    'tf_score': round(tf_score, 4),
                    'content_preview': preview,
                    'positions': data['positions'][:5]  # 展示前5个位置
                })

        results.sort(key=lambda x: (
            x['word_frequency'],  # 直接按词频排序
            x['tf_score']  # 其次按TF分数排序
        ), reverse=True)


        return {
            'found': len(results) > 0,
            'query': query,
            'total_results': len(results),
            'results': results,
            'message': f'找到 {len(results)} 个匹配文档' if results else '未找到匹配文档'
        }

    # tfidf词向量检索，返回对应词每个文章中的查询词向量，以及文件路径 
    def tfidf_search(self, query: str, inverted_index: Dict[str, Any], processed_files_content: Dict[str, str], file_word_count: Dict[str, any], all_content: list) -> Dict[str, any]:
        """
        基于TF-IDF的检索
        Args:
            query: 搜索查询词
            inverted_index: 倒排索引
            processed_files_content: 文件内容字典
            file_word_count: 文件词频统计
            all_content: 所有文件内容列表
        Returns:
            检索结果字典
        """
        if not query:
            return {'found': False, 'message': '查询词为空', 'results': []}

        query_words = self.preprocess_query(query)
        if not query_words:
            return {'found': False, 'message': '查询词无效', 'results': []}

        # 使用TF-IDF向量化
        try:
            vectorizer = TfidfVectorizer(max_features=50, min_df=1, max_df=0.8, ngram_range=(1, 2))

            # 准备文档列表（确保顺序与processed_files_content一致）
            doc_list = []
            file_names = []
            for file_name, content in processed_files_content.items():
                if not content.startswith("错误:"):
                    processed_content = self.preprocess_query(content)  # 使用相同的预处理
                    doc_list.append(' '.join(processed_content))
                    file_names.append(file_name)

            if not doc_list:
                return {'found': False, 'message': '没有有效文档进行TF-IDF分析', 'results': []}

            # 构建TF-IDF矩阵
            tfidf_matrix = vectorizer.fit_transform(doc_list)

            # 处理查询
            query_processed = ' '.join(query_words)
            query_vector = vectorizer.transform([query_processed])

            # 计算余弦相似度
            similarities = cosine_similarity(query_vector, tfidf_matrix).flatten()

            # 获取相似度排序的文档索引
            doc_indices = similarities.argsort()[::-1]

            results = []
            for idx in doc_indices:
                similarity_score = similarities[idx]
                if similarity_score > 0:  # 只返回有相似度的文档
                    file_name = file_names[idx]
                    content = processed_files_content[file_name]

                    # 找到匹配的查询词
                    matched_words = []
                    for word in query_words:
                        if word in content.lower():
                            matched_words.append(word)

                    preview = self.generate_content_preview(content, matched_words[0] if matched_words else query_words[0])

                    results.append({
                        'file_name': file_name,
                        'tfidf_score': round(similarity_score, 4),
                        'matched_words': matched_words,
                        'content_preview': preview,
                        'relevance': 'high' if similarity_score > 0.3 else 'medium' if similarity_score > 0.1 else 'low'
                    })

            return {
                'found': len(results) > 0,
                'query': query,
                'total_results': len(results),
                'results': results[:20],  # 限制返回前20个结果
                'message': f'TF-IDF检索找到 {len(results)} 个相关文档' if results else 'TF-IDF检索未找到相关文档'
            }

        except Exception as e:
            return {
                'found': False,
                'message': f'TF-IDF检索过程中发生错误: {str(e)}',
                'results': []
            }
    
    # 词向量检索，返回对应词每个文章中的查询词向量，以及文件路径 
    def word_vector_search(self, query: str, inverted_index: Dict[str, Any], processed_files_content: Dict[str, str], file_word_count: Dict[str, Any], all_content: List[str]) -> Dict[str, Any]:
        """
        基于词向量的检索（使用CountVectorizer）
        Args:
            query: 搜索查询词
            inverted_index: 倒排索引
            processed_files_content: 文件内容字典
            file_word_count: 文件词频统计
            all_content: 所有文件内容列表
        Returns:
            检索结果字典
        """
        if not query:
            return {'found': False, 'message': '查询词为空', 'results': []}

        query_words = self.preprocess_query(query)
        if not query_words:
            return {'found': False, 'message': '查询词无效', 'results': []}

        try:
            # 使用CountVectorizer进行词向量化
            vectorizer = CountVectorizer(max_features=50, min_df=1, max_df=0.8, ngram_range=(1, 2))

            # 准备文档列表
            doc_list = []
            file_names = []
            for file_name, content in processed_files_content.items():
                if not content.startswith("错误:"):
                    processed_content = self.preprocess_query(content)
                    doc_list.append(' '.join(processed_content))
                    file_names.append(file_name)

            if not doc_list:
                return {'found': False, 'message': '没有有效文档进行词向量分析', 'results': []}

            # 构建词向量矩阵
            word_matrix = vectorizer.fit_transform(doc_list)

            # 处理查询
            query_processed = ' '.join(query_words)
            query_vector = vectorizer.transform([query_processed])

            # 计算余弦相似度
            similarities = cosine_similarity(query_vector, word_matrix).flatten()

            # 获取特征名称（词汇表）
            feature_names = vectorizer.get_feature_names_out()

            # 获取查询向量中的非零元素（匹配的词）
            query_vector_dense = query_vector.toarray()[0]
            matched_features = [(feature_names[i], query_vector_dense[i]) 
                               for i in range(len(query_vector_dense)) 
                               if query_vector_dense[i] > 0]

            # 获取相似度排序的文档索引
            doc_indices = similarities.argsort()[::-1]

            results = []
            for idx in doc_indices:
                similarity_score = similarities[idx]
                if similarity_score > 0:  # 只返回有相似度的文档
                    file_name = file_names[idx]
                    content = processed_files_content[file_name]

                    # 获取该文档的词向量
                    doc_vector = word_matrix[idx].toarray()[0]

                    # 找到在该文档中出现的查询词及其频率
                    doc_matched_words = []
                    for word in query_words:
                        if word in feature_names:
                            word_idx = list(feature_names).index(word)
                            if doc_vector[word_idx] > 0:
                                doc_matched_words.append({
                                    'word': word,
                                    'frequency': int(doc_vector[word_idx])
                                })

                    preview = self.generate_content_preview(content, query_words[0])

                    results.append({
                        'file_name': file_name,
                        'vector_similarity': round(similarity_score, 4),
                        'matched_word_vectors': doc_matched_words,
                        'content_preview': preview,
                        'total_matched_terms': len(doc_matched_words),
                        'relevance': 'high' if similarity_score > 0.5 else 'medium' if similarity_score > 0.2 else 'low'
                    })

            return {
                'found': len(results) > 0,
                'query': query,
                'query_vector_info': {
                    'matched_features': matched_features,
                    'total_query_terms': len(matched_features)
                },
                'total_results': len(results),
                'results': results[:20],  # 限制返回前20个结果
                'message': f'词向量检索找到 {len(results)} 个相关文档' if results else '词向量检索未找到相关文档'
            }

        except Exception as e:
            return {
                'found': False,
                'message': f'词向量检索过程中发生错误: {str(e)}',
                'results': []
            }
        
class Visualization:
    def __init__(self):
        """初始化可视化类"""
        self.output_dir = 'server/src/pic/cloud'
        self.ensure_output_dir()
        # 设置中文字体 - 添加字体存在性检查
        self._setup_matplotlib_fonts()
        # 停用词
        self.stopwords = self._load_stopwords()

    def _setup_matplotlib_fonts(self):
        """设置matplotlib中文字体"""
        try:
            plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS', 'DejaVu Sans', 'sans-serif']
            plt.rcParams['axes.unicode_minus'] = False
        except Exception as e:
            print(f"字体设置警告: {e}")
            # 使用默认字体
            plt.rcParams['font.sans-serif'] = ['DejaVu Sans']

  
    def ensure_output_dir(self):
        """确保输出目录存在"""
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir, exist_ok=True)

    def _load_stopwords(self) -> List[str]:
        """加载停用词"""
        try:
            with open(stopwords_path, 'r', encoding='utf-8') as f:
                return [word.strip() for word in f.readlines()]
        except FileNotFoundError:
            return ['的', '了', '在', '是', '我', '有', '和', '就', '不', '人', '都', '一', '一个']
    
    def preprocess_text_for_vis(self, text_data) -> List[str]:
        """为可视化预处理文本数据"""
        all_words = []

        if isinstance(text_data, dict):
            # 处理字典数据（文件内容字典）
            for content in text_data.values():
                if not content.startswith("错误:"):
                    words = self.preprocess_query(content)
                    all_words.extend(words)
        elif isinstance(text_data, list):
            # 处理列表数据
            for content in text_data:
                if isinstance(content, str) and not content.startswith("错误:"):
                    words = self.preprocess_query(content)
                    all_words.extend(words)
        elif isinstance(text_data, str):
            # 处理单个字符串
            words = self.preprocess_query(text_data)
            all_words.extend(words)

        return all_words

    def preprocess_query(self, query: str) -> List[str]:
        """预处理查询文本"""
        # 去除特殊字符，保留中文、英文、数字
        query = re.sub(r'[^\u4e00-\u9fa5a-zA-Z0-9\s]', '', query)
        # 分词
        words = list(jieba.cut(query))
        # 去停用词和空词
        words = [word.strip() for word in words 
                if word.strip() and word.strip() not in self.stopwords and len(word.strip()) > 1]
        return words
    # 生成查询文章的词云图
    def generate_wordcloud(self, query: str, processed_files_content: Dict[str,str], filename: str = 'wordcloud', top_n: int = 50) -> Dict[str, Any]:
        """
        生成查询文章的词云图

        Args:
            query: 查询词
            processed_files_content: 文件内容字典
            filename: 保存文件名
            top_n: 显示前N个词

        Returns:
            包含词频统计和保存路径的字典
        """
        try:
            # 预处理查询词
            query_words = self.preprocess_query(query)
            if not query_words:
                return {
                    'success': False,
                    'message': '查询词无效',
                    'word_freq': {},
                    'save_path': None
                }

            # 收集包含查询词的文档内容
            relevant_content = []
            for file_name, content in processed_files_content.items():
                if content.startswith("错误:"):
                    continue
                
                # 检查文档是否包含查询词
                content_words = self.preprocess_query(content)
                if any(word in content_words for word in query_words):
                    relevant_content.append(content)

            if not relevant_content:
                return {
                    'success': False,
                    'message': f'没有找到包含查询词 "{query}" 的文档',
                    'word_freq': {},
                    'save_path': None
                }

            # 预处理相关文档的文本
            all_words = self.preprocess_text_for_vis(relevant_content)

            if not all_words:
                return {
                    'success': False,
                    'message': '没有有效的文本数据生成词云',
                    'word_freq': {},
                    'save_path': None
                }

            # 统计词频
            word_freq = Counter(all_words)
            top_words = dict(word_freq.most_common(top_n))

            # 生成词云
            wordcloud = WordCloud(
                width=1200,
                height=800,
                background_color='white',
                max_words=top_n,
                colormap='viridis',
                relative_scaling=0.5,
                random_state=42
            ).generate_from_frequencies(top_words)

            # 保存词云图
            save_path = os.path.join(self.output_dir, f'{filename}_{query}.png')

            plt.figure(figsize=(15, 10))
            plt.imshow(wordcloud, interpolation='bilinear')
            plt.axis('off')
            plt.title(f'查询词 "{query}" 相关文档词云图 - 前{top_n}个高频词', fontsize=16, pad=20)
            plt.tight_layout()
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            plt.close()

            return {
                'success': True,
                'message': f'词云图已生成并保存到 {save_path}',
                'word_freq': top_words,
                'save_path': save_path,
                'total_words': len(all_words),
                'unique_words': len(word_freq),
                'relevant_docs': len(relevant_content)
            }

        except Exception as e:
            return {
                'success': False,
                'message': f'生成词云图时发生错误: {str(e)}',
                'word_freq': {},
                'save_path': None
            }
    # 生成查询文章的词向量图
    def generate_keyword_ranking(self, query: str, processed_files_content: Dict[str,str], file_word_count: Dict[str,int], filename: str = 'keyword_ranking', top_n: int = 10) -> Dict[str, Any]:
        """
        生成查询文章的关键词排行图

        Args:
            query: 查询词
            processed_files_content: 文件内容字典
            file_word_count: 文件词频统计
            filename: 保存文件名
            top_n: 显示前N个关键词

        Returns:
            包含关键词列表和保存路径的字典
        """
        try:
            # 预处理查询词
            query_words = self.preprocess_query(query)
            if not query_words:
                return {
                    'success': False,
                    'message': '查询词无效',
                    'keywords': [],
                    'save_path': None
                }

            # 收集包含查询词的文档的词频统计
            relevant_word_count = {}
            relevant_files = []

            for file_name, content in processed_files_content.items():
                if content.startswith("错误:"):
                    continue
                
                # 检查文档是否包含查询词
                content_words = self.preprocess_query(content)
                if any(word in content_words for word in query_words):
                    relevant_files.append(file_name)

                    # 合并该文档的词频统计
                    if file_name in file_word_count:
                        for word, count in file_word_count[file_name].items():
                            relevant_word_count[word] = relevant_word_count.get(word, 0) + count

            if not relevant_word_count:
                return {
                    'success': False,
                    'message': f'没有找到包含查询词 "{query}" 的文档词频数据',
                    'keywords': [],
                    'save_path': None
                }

            # 获取前N个关键词
            top_keywords = Counter(relevant_word_count).most_common(top_n)

            if not top_keywords:
                return {
                    'success': False,
                    'message': '没有足够的关键词生成排行',
                    'keywords': [],
                    'save_path': None
                }

            # 准备数据
            keywords = [item[0] for item in top_keywords]
            frequencies = [item[1] for item in top_keywords]

            # 生成排行图
            plt.figure(figsize=(12, 8))

            # 创建水平条形图
            bars = plt.barh(range(len(keywords)), frequencies, color='skyblue', alpha=0.8)

            # 设置标签
            plt.yticks(range(len(keywords)), keywords)
            plt.xlabel('词频', fontsize=12)
            plt.ylabel('关键词', fontsize=12)
            plt.title(f'查询词 "{query}" 相关文档关键词排行榜 - 前{top_n}名', fontsize=14, pad=20)

            # 在条形图上添加数值标签
            for i, (bar, freq) in enumerate(zip(bars, frequencies)):
                plt.text(bar.get_width() + max(frequencies) * 0.01, 
                        bar.get_y() + bar.get_height()/2, 
                        str(freq), 
                        ha='left', va='center', fontsize=10)

            # 反转y轴，使频率最高的在顶部
            plt.gca().invert_yaxis()

            # 调整布局
            plt.tight_layout()

            # 保存图片
            save_path = os.path.join(self.output_dir, f'{filename}_{query}.png')
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            plt.close()

            # 返回关键词列表
            keyword_list = [{'keyword': kw, 'frequency': freq, 'rank': i+1} 
                           for i, (kw, freq) in enumerate(top_keywords)]

            return {
                'success': True,
                'message': f'关键词排行图已生成并保存到 {save_path}',
                'keywords': keyword_list,
                'save_path': save_path,
                'relevant_files': relevant_files,
                'total_analyzed_files': len(relevant_files)
            }

        except Exception as e:
            return {
                'success': False,
                'message': f'生成关键词排行图时发生错误: {str(e)}',
                'keywords': [],
                'save_path': None
            }

    # 生成整个文件夹的文件的种类统计图
    def analyze_file_extensions(self, file_name: list, generate_chart: bool = True, filename: str = 'file_extensions') -> Dict[str, Any]:

        try:
            if not file_name:
                return {
                    'success': False,
                    'message': '文件名列表为空',
                    'extension_stats': [],
                    'save_path': None
                }

            # 提取文件后缀
            file_extensions = []
            for name in file_name:
                if isinstance(name, str):
                    ext = os.path.splitext(name)[1].lower()
                    if ext:
                        file_extensions.append(ext[1:])  # 去掉点号
                    else:
                        file_extensions.append('无后缀')

            if not file_extensions:
                return {
                    'success': False,
                    'message': '没有有效的文件数据进行分析',
                    'extension_stats': [],
                    'save_path': None
                }

            # 统计后缀频率
            ext_counter = Counter(file_extensions)
            extension_stats = [{'extension': ext, 'count': count, 'percentage': count/len(file_extensions)*100} 
                             for ext, count in ext_counter.most_common()]

            save_path = None

            if generate_chart and extension_stats:
                # 生成饼图
                plt.figure(figsize=(10, 8))

                extensions = [item['extension'] for item in extension_stats]
                counts = [item['count'] for item in extension_stats]

                # 设置颜色
                colors = plt.cm.Set3(np.linspace(0, 1, len(extensions)))

                # 创建饼图
                wedges, texts, autotexts = plt.pie(counts, labels=extensions, autopct='%1.1f%%', 
                                                  colors=colors, startangle=90)

                # 设置标题
                plt.title(f'文件类型分布统计 (总计: {len(file_extensions)} 个文件)', fontsize=14, pad=20)

                # 添加图例
                plt.legend(wedges, [f'{ext} ({count})' for ext, count in zip(extensions, counts)],
                          title="文件类型",
                          loc="center left",
                          bbox_to_anchor=(1, 0, 0.5, 1))

                plt.axis('equal')

                # 保存图片
                save_path = os.path.join(self.output_dir, f'{filename}.png')
                plt.savefig(save_path, dpi=300, bbox_inches='tight')
                plt.close()

            return {
                'success': True,
                'message': f'文件后缀分析完成' + (f'，图表已保存到 {save_path}' if save_path else ''),
                'extension_stats': extension_stats,
                'save_path': save_path,
                'total_files': len(file_extensions),
                'unique_extensions': len(ext_counter)
            }

        except Exception as e:
            return {
                'success': False,
                'message': f'分析文件后缀时发生错误: {str(e)}',
                'extension_stats': [],
                'save_path': None
            }

    # 统计全文件夹文件的词频前10
    def create_word_frequency_chart(self, word_dict: Dict[str, any], chart_type: str = 'bar', top_n: int = 20, filename: str = 'word_frequency') -> Dict[str, Any]:
        """
        统计全文件夹文件的词频前N名并生成图表

        Args:
            word_dict: 全文词频统计字典
            chart_type: 图表类型 ('bar', 'line', 'pie')
            top_n: 显示前N个词
            filename: 保存文件名

        Returns:
            图表生成结果
        """
        try:
            if not word_dict:
                return {
                    'success': False,
                    'message': '词频字典为空',
                    'save_path': None
                }

            # 获取前N个高频词
            if isinstance(word_dict, dict):
                # 如果已经是排序好的字典，直接取前N个
                top_words_items = list(word_dict.items())[:top_n]
            else:
                return {
                    'success': False,
                    'message': '词频数据格式不正确',
                    'save_path': None
                }

            if not top_words_items:
                return {
                    'success': False,
                    'message': '没有足够的词汇生成图表',
                    'save_path': None
                }

            words = [item[0] for item in top_words_items]
            frequencies = [item[1] for item in top_words_items]

            plt.figure(figsize=(12, 8))

            if chart_type == 'bar':
                bars = plt.bar(range(len(words)), frequencies, color='lightblue', alpha=0.8)
                plt.xticks(range(len(words)), words, rotation=45, ha='right')
                plt.ylabel('词频', fontsize=12)
                plt.xlabel('词汇', fontsize=12)

                # 在柱状图上添加数值标签
                for bar, freq in zip(bars, frequencies):
                    plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(frequencies) * 0.01,
                            str(freq), ha='center', va='bottom', fontsize=9)

            elif chart_type == 'line':
                plt.plot(range(len(words)), frequencies, marker='o', linewidth=2, markersize=6, color='blue')
                plt.xticks(range(len(words)), words, rotation=45, ha='right')
                plt.ylabel('词频', fontsize=12)
                plt.xlabel('词汇', fontsize=12)
                plt.grid(True, alpha=0.3)

                # 在数据点上添加数值标签
                for i, freq in enumerate(frequencies):
                    plt.text(i, freq + max(frequencies) * 0.02, str(freq), 
                            ha='center', va='bottom', fontsize=9)

            elif chart_type == 'pie':
                plt.pie(frequencies, labels=words, autopct='%1.1f%%', startangle=90)
                plt.axis('equal')

            plt.title(f'全文件夹词频统计 - {chart_type.upper()}图 (前{top_n}个)', fontsize=14, pad=20)
            plt.tight_layout()

            # 保存图片
            save_path = os.path.join(self.output_dir, f'{filename}_{chart_type}.png')
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            plt.close()

            return {
                'success': True,
                'message': f'{chart_type.upper()}图已生成并保存到 {save_path}',
                'save_path': save_path,
                'chart_type': chart_type,
                'top_words': top_words_items,
                'total_unique_words': len(word_dict),
                'analyzed_words': len(top_words_items)
            }

        except Exception as e:
            return {
                'success': False,
                'message': f'生成{chart_type}图时发生错误: {str(e)}',
                'save_path': None
            }

class APIHandler:
    def __init__(self):
        """初始化API"""
        self.app = Flask(__name__)
        CORS(self.app)
        # init 功能模块 
        self.file_reader = None# 实现GET_file 
        self.search_engine = scearch_content()#搜索策略实现 
        self.visual_processor = Visualization()# 可视化模块实现 

        self.cache = {}
        self._setup_routes()

        # logging 
        logging.basicConfig(level = logging.INFO)
        self.logger = logging.getLogger(__name__)
    # set_Api_routes
    def _setup_routes(self):
        # 基础功能路由
        self.app.add_url_rule('/upload_config', 'upload_config', self.upload_config, methods=['POST'])
        self.app.add_url_rule('/health', 'health_check', self.health_check, methods=['GET'])

        # 搜索功能路由
        self.app.add_url_rule('/tf_search', 'tf_search', self.tf_search, methods=['POST'])
        self.app.add_url_rule('/tfidf_search', 'tfidf_search', self.tfidf_search, methods=['POST'])
        self.app.add_url_rule('/word_vector_search', 'word_vector_search', self.word_vector_search, methods=['POST'])

        # 可视化功能路由
        self.app.add_url_rule('/generate_wordcloud', 'generate_wordcloud', self.generate_wordcloud, methods=['POST'])
        self.app.add_url_rule('/generate_keyword_ranking', 'generate_keyword_ranking', self.generate_keyword_ranking, methods=['POST'])
        self.app.add_url_rule('/analyze_file_extensions', 'analyze_file_extensions', self.analyze_file_extensions, methods=['POST'])
        self.app.add_url_rule('/create_word_frequency_chart', 'create_word_frequency_chart', self.create_word_frequency_chart, methods=['POST'])

        # 统计信息路由
        self.app.add_url_rule('/get_word_statistics', 'get_word_statistics', self.get_word_statistics, methods=['POST'])
        self.app.add_url_rule('/get_file_statistics', 'get_file_statistics', self.get_file_statistics, methods=['POST'])

        # 会话管理路由
        self.app.add_url_rule('/clear_session', 'clear_session', self.clear_session, methods=['POST'])
        self.app.add_url_rule('/get_session_info', 'get_session_info', self.get_session_info, methods=['GET'])

        # 文件下载路由
        self.app.add_url_rule('/download/<filename>', 'download_file', self.download_file, methods=['GET'])

    def _create_response(self, success: bool,  data: Any = None, message: str = "", error: str = ""):
        response = {
            'success': success,
            'timetamp': datetime.now().isoformat(),
            'message': message
        }
        if  success:
            response['data'] = data 
        else: 
            response['error'] = error 
        return jsonify(response)
    # handle_error
    def _handle_error(self, error: Exception, operation: str) :
        error_msg = f"{operation}时发生错误：{str(error)}"
        self.logger.error(error_msg)
        return self._create_response(False, error=error_msg)
    
    # reading_dir API
    def upload_config(self):
        """ 上传配置文件，并且初始化处理 """
        try: 
            data = request.get_json()
            if not data:
                return self._create_response(False, error="请求数据为空")

            config_path = data.get('config_path')
            if not config_path:
                return self._create_response(False, error="配置文件路径不能为空")

            # 添加文件存在性检查
            if not os.path.exists(config_path):
                return self._create_response(False, error="配置文件不存在")

            self.file_reader = Get_file(config_path)
            self.file_reader.init_read_dir()

            # 生成会话ID
            session_id = str(uuid.uuid4())
            self.cache[session_id] = {
                'file_reader': self.file_reader,
                'timestamp': datetime.now()
            }
            return self._create_response(True, data={'session_id': session_id}, message="配置文件上传成功")
        except Exception as e:
            return self._handle_error(e, "上传配置文件")


    #search API
    def tf_search(self):
        """"词频搜索"""
        try:
            data = request.get_json()
            session_id = data.get('session_id')
            query = data.get('query', '')
            if not session_id or session_id not in self.cache:
                return self._create_response(False, error="无效的会话ID")
            
            file_processor = self.cache[session_id]['file_reader']
            inverted_index = file_processor.inverted_index
            processed_files_content = file_processor.processed_files_content
            file_word_count = file_processor.file_word_count
            results = self.search_engine.tf_search(query, inverted_index, processed_files_content, file_word_count)
            return self._create_response(True, data=results, message="TF检索成功")
        except Exception as e:
            return self._handle_error(e, "TF检索")
 
    def tfidf_search(self):
        """"TF-IDF搜索"""
        try:
            data = request.get_json()
            session_id = data.get('session_id')
            query = data.get('query', '')
            if not session_id or session_id not in self.cache: 
                return self._create_response(False, error="无效的会话ID")

            file_processor = self.cache[session_id]['file_reader']
            inverted_index = file_processor.inverted_index
            processed_files_content = file_processor.processed_files_content
            file_word_count = file_processor.file_word_count
            all_content = file_processor.content_list  # 修复：使用 content_list 而不是 all_content
            results = self.search_engine.tfidf_search(query, inverted_index, processed_files_content, file_word_count, all_content)   
            return self._create_response(True, data=results, message="TF-IDF检索成功")
        except Exception as e:
            return self._handle_error(e, "TF-IDF检索")

    def word_vector_search(self):
        """"词向量搜索"""
        try:
            data = request.get_json()
            session_id = data.get('session_id')
            query = data.get('query', '')
            if not session_id or session_id not in self.cache: 
                return self._create_response(False, error="无效的会话ID")

            file_processor = self.cache[session_id]['file_reader']
            inverted_index = file_processor.inverted_index
            processed_files_content = file_processor.processed_files_content
            file_word_count = file_processor.file_word_count
            all_content = file_processor.content_list  # 修复：使用 content_list 而不是 all_content
            results = self.search_engine.word_vector_search(query, inverted_index, processed_files_content, file_word_count, all_content)   
            return self._create_response(True, data=results, message="词向量检索成功")
        except Exception as e:
            return self._handle_error(e, "词向量检索")

    # visualize API
    def generate_wordcloud(self):
        """生成词云图"""
        try:
            data = request.get_json()
            session_id = data.get('session_id')
            query = data.get('query', '')
            if not session_id or session_id not in self.cache: 
                return self._create_response(False, error="无效的会话ID")

            file_processor = self.cache[session_id]['file_reader']
            processed_files_content = file_processor.processed_files_content
            result = self.visual_processor.generate_wordcloud(query, processed_files_content)
            return self._create_response(True, data=result, message="词云图生成成功")
        except Exception as e:
            return self._handle_error(e, "生成词云图")
    def generate_keyword_ranking(self):
        """生成关键词排名图"""
        try:
            data = request.get_json()
            session_id = data.get('session_id')
            query = data.get('query', '')
            if not session_id or session_id not in self.cache: 
                return self._create_response(False, error="无效的会话ID")

            file_processor = self.cache[session_id]['file_reader']
            processed_files_content = file_processor.processed_files_content
            file_word_count = file_processor.file_word_count
            result = self.visual_processor.generate_keyword_ranking(query, processed_files_content, file_word_count)
            return self._create_response(True, data=result, message="关键词排名图生成成功")
        except Exception as e:
            return self._handle_error(e, "生成关键词排名图")
    def analyze_file_extensions(self):
        """分析文件后缀并生成统计图"""
        try:
            data = request.get_json()
            session_id = data.get('session_id')
            if not session_id or session_id not in self.cache: 
                return self._create_response(False, error="无效的会话ID")

            file_processor = self.cache[session_id]['file_reader']
            file_names = file_processor.file_name  # 使用文件处理器中的文件名列表
            result = self.visual_processor.analyze_file_extensions(file_names)
            return self._create_response(True, data=result, message="文件后缀分析成功")
        except Exception as e:
            return self._handle_error(e, "分析文件后缀")

    def create_word_frequency_chart(self):
        """生成词频图"""
        try:
            data = request.get_json()
            session_id = data.get('session_id')
            chart_type = data.get('chart_type', 'bar')  # 添加图表类型参数
            top_n = data.get('top_n', 20)  # 添加显示数量参数
            if not session_id or session_id not in self.cache: 
                return self._create_response(False, error="无效的会话ID")

            file_processor = self.cache[session_id]['file_reader']
            word_dict = file_processor.word_dict
            result = self.visual_processor.create_word_frequency_chart(word_dict, chart_type, top_n)
            return self._create_response(True, data=result, message="词频图生成成功")
        except Exception as e:
            return self._handle_error(e, "生成词频图")

    # 统计信息API 
    def get_word_statistics(self):
        """获取词频统计信息"""
        try:
            data = request.get_json()
            session_id = data.get('session_id')
            if not session_id or session_id not in self.cache: 
                return self._create_response(False, error="无效的会话ID")   
            file_processor = self.cache[session_id]['file_reader']
            word_dict = file_processor.word_dict
            total_words = len(word_dict)
            unique_words = len(set(word_dict.keys()))
            return self._create_response(True, data={
                'total_words': total_words,
                'unique_words': unique_words,
                'word_statistics': word_dict
            }, message="词频统计信息获取成功")
        except Exception as e:
            return self._handle_error(e, "获取词频统计信息")  
    def get_file_statistics(self):
        """获取文件统计信息"""
        try:
            data = request.get_json()
            session_id = data.get('session_id')
            if not session_id or session_id not in self.cache: 
                return self._create_response(False, error="无效的会话ID")   
            file_processor = self.cache[session_id]['file_reader']
            file_count =  file_processor.total_docs
            return self._create_response(True, data={
                'file_count': file_count,
                'file_statistics': {
                    'file_count': file_count,
                }
            }, message="文件统计信息获取成功")
        except Exception as e:
            return self._handle_error(e, "获取文件统计信息")
    def download_file(self,filename):
        """下载文件"""
        try:
            file_path = os.path.join('server/src/pic/cloud', filename)
            
            if not os.path.exists(file_path):
                return self._create_response(False, error="文件不存在")
            
            return send_file(file_path, as_attachment=True)
            
        except Exception as e:
            return self._handle_error(e, "下载文件")
    def health_check(self):
        """健康检查"""
        return self._create_response(True, 
                                   data={'status': 'healthy', 'version': '1.0.0'}, 
                                   message="服务运行正常")
    def clear_session(self):
        """清理指定会话"""
        try:
            data = request.get_json()
            session_id = data.get('session_id')

            if not session_id:
                return self._create_response(False, error="会话ID不能为空")

            if session_id in self.cache:
                del self.cache[session_id]
                return self._create_response(True, message="会话清理成功")
            else:
                return self._create_response(False, error="会话不存在")

        except Exception as e:
            return self._handle_error(e, "清理会话")

    def get_session_info(self):
        """获取会话信息"""
        try:
            cache_info = self.get_cache_info()
            return self._create_response(True, data=cache_info, message="获取会话信息成功")
        except Exception as e:
            return self._handle_error(e, "获取会话信息")

    def clear_cache(self, session_id: str = None):
        """清理缓存"""
        try:
            if session_id:
                if session_id in self.cache:
                    del self.cache[session_id]
                    return True
            else:
                self.cache.clear()
                return True
            return False
        except Exception as e:
            self.logger.error(f"清理缓存时发生错误: {str(e)}")
            return False

    def get_cache_info(self):
        """获取缓存信息"""
        return {
            'active_sessions': len(self.cache),
            'session_ids': list(self.cache.keys()),
            'cache_size': len(self.cache)
        }

    # 启动服务器
    def run_server(self, host='localhost', port=5000, debug=False):
        """启动Flask服务器"""
        self.logger.info(f"启动服务器: http://{host}:{port}")
        self.app.run(host=host, port=port, debug=debug)
    


# 服务器启动和配置类
class ServerManager:
    """
    服务器管理类
    负责配置和启动整个后端服务
    """
    
    def __init__(self, config_file: str = None):
        """初始化服务器管理器"""
        self.api_handler = APIHandler()
        self.config = self._load_config(config_file) if config_file else self._default_config()
        
    def _load_config(self, config_file: str) -> Dict[str, Any]:
        """加载配置文件"""
        try:
            with open(config_file, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception as e:
            print(f"加载配置文件失败: {e}，使用默认配置")
            return self._default_config()
    
    def _default_config(self) -> Dict[str, Any]:
        """默认配置"""
        return {
            'server': {
                'host': 'localhost',
                'port': 5000,
                'debug': False
            },
            'cors': {
                'origins': ['http://localhost:8080', 'http://localhost:3000'],
                'methods': ['GET', 'POST', 'PUT', 'DELETE'],
                'allow_headers': ['Content-Type', 'Authorization']
            },
            'upload': {
                'max_file_size': 16 * 1024 * 1024,  # 16MB
                'allowed_extensions': ['txt', 'json', 'csv', 'xlsx', 'xls', 'pdf', 'docx']
            },
            'cache': {
                'max_sessions': 100,
                'session_timeout': 3600  # 1小时
            }
        }
    
    def setup_cors(self):
        """配置CORS"""
        cors_config = self.config.get('cors', {})
        CORS(self.api_handler.app, 
             origins=cors_config.get('origins', ['*']),
             methods=cors_config.get('methods', ['GET', 'POST']),
             allow_headers=cors_config.get('allow_headers', ['Content-Type']))
    
    def add_middleware(self):
        """添加中间件"""
        
        @self.api_handler.app.before_request
        def before_request():
            """请求前处理"""
            # 记录请求日志
            self.api_handler.logger.info(f"收到请求: {request.method} {request.path}")
            
            # 检查文件大小限制
            if request.content_length:
                max_size = self.config.get('upload', {}).get('max_file_size', 16 * 1024 * 1024)
                if request.content_length > max_size:
                    return jsonify({'error': '文件大小超过限制'}), 413
        
        @self.api_handler.app.after_request
        def after_request(response):
            """请求后处理"""
            # 添加响应头
            response.headers['X-Content-Type-Options'] = 'nosniff'
            response.headers['X-Frame-Options'] = 'DENY'
            response.headers['X-XSS-Protection'] = '1; mode=block'
            return response
        
        @self.api_handler.app.errorhandler(404)
        def not_found(error):
            """404错误处理"""
            return jsonify({'error': '接口不存在', 'code': 404}), 404
        
        @self.api_handler.app.errorhandler(500)
        def internal_error(error):
            """500错误处理"""
            return jsonify({'error': '服务器内部错误', 'code': 500}), 500
    
    def start_server(self):
        """启动服务器"""
        # 配置CORS
        self.setup_cors()
        
        # 添加中间件
        self.add_middleware()
        
        # 获取服务器配置
        server_config = self.config.get('server', {})
        host = server_config.get('host', 'localhost')
        port = server_config.get('port', 5000)
        debug = server_config.get('debug', False)
        
        print(f"=== 文本信息检索系统后端服务 ===")
        print(f"服务地址: http://{host}:{port}")
        print(f"API文档: http://{host}:{port}/api/health")
        print(f"调试模式: {'开启' if debug else '关闭'}")
        print("=" * 40)
        
        # 启动服务器
        self.api_handler.run_server(host=host, port=port, debug=debug)


# 使用示例和启动脚本
def create_app(config_file: str = None):
    """创建应用实例"""
    server_manager = ServerManager(config_file)
    return server_manager.api_handler.app

def main():
    """主函数 - 启动服务器"""
    import argparse
    
    parser = argparse.ArgumentParser(description='文本信息检索系统后端服务')
    parser.add_argument('--config', type=str, help='配置文件路径')
    parser.add_argument('--host', type=str, default='localhost', help='服务器地址')
    parser.add_argument('--port', type=int, default=5000, help='服务器端口')
    parser.add_argument('--debug', action='store_true', help='开启调试模式')
    
    args = parser.parse_args()
    
    # 创建服务器管理器
    server_manager = ServerManager(args.config)
    
    # 如果有命令行参数，覆盖配置
    if args.host != 'localhost':
        server_manager.config['server']['host'] = args.host
    if args.port != 5000:
        server_manager.config['server']['port'] = args.port
    if args.debug:
        server_manager.config['server']['debug'] = True
    
    # 启动服务器
    server_manager.start_server()

if __name__ == '__main__':
    main()

                                       

        
