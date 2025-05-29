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
        self.inverted_index = {}#  倒排索引
        self.file_word_count = {}#  文件词频统计
        self.content_list = []
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
            return 
    #读文件    
    def read_dir_file(self) -> Dict[str, str]:
        for root, dirs, files in os.walk(self.json_dir):
            for file_name in files: # 使用 file_name 避免与外部可能的 file 变量冲突
                self.file_path.append(os.path.join(root,file_name))

        for file_path_item in self.file_path: # 遍历收集到的文件路径
            content_or_error_msg = f"错误: 未知或不支持的文件类型 '{os.path.basename(file_path_item)}'." # 默认消息
            file_name_extension = os.path.splitext(file_path_item)[1][1:]
            content_or_error_msg = self.read_content(file_name_extension, file_path_item)
            self.processed_files_content[file_path_item] = content_or_error_msg
            self.content_list.append(content_or_error_msg)
    #洁词
    def process_text(self, content:list)-> List[str]:
        process_content = []
        stopword_path = stopwords_path
        for text in content:
            text = re.sub(r'[^\u4e00-\u9fa5a-zA-Z0-9\s]', '', text)  # 删除非中文字符}')
            text = ''.join(jieba.cut(text))
            stopwords = open(stopword_path, 'r', encoding='utf-8').readlines()
            stopwords = [word.strip() for word in stopwords]
            text = ' '.join([word for word in text.split() if word not in stopwords])
            process_content.append(text)
        return process_content
    
    #统计词频
    def word_count(self) -> Dict[str, int]:
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
        return dict(sorted_word_dict)
    
    #构建倒索排序
    def build_index(self) -> None:
        self.inverted_index = {}
        self.file_word_count = {}
        for file_name, content in self.processed_files_content.items():
            if content.startswith("错误:"):
                continue
            process_content = self.process_text([content])
            if not process_content:
                continue
            words = process_content[0].split()
            word_count = {}

            for word in words:
                if word:
                    word_count[word] = word_count.get(word, 0) + 1
                    if word not in self.inverted_index:
                        self.inverted_index[word] = []
                    if file_name not in self.inverted_index[word]:
                        self.inverted_index[word].append(file_name)

            self.file_word_count[file_name] = word_count

    def TD_IDF(self, content:str):
        vector = TfidfVectorizer(max_features= 50, min_df=2, max_df=0.8, ngram_range=(1, 2), sublinear_tf=True)
        X = vector.fit_transform(content)
        tfidf_df = pd.DataFrame(X.toarray(), columns=vector.get_feature_names_out())
        return X,tfidf_df
    
    def word_vetctor(self, content:str):
        vectorizer = CountVectorizer(max_features=50, min_df=2, max_df=0.8, ngram_range=(1, 2))
        X = vectorizer.fit_transform(content)
        word_vector_df = pd.DataFrame(X.toarray(), columns=vectorizer.get_feature_names_out())
        return X, word_vector_df    
                
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
    
    def simple_search(self, query: str, content_list: List[str], file_names: List[str] = None) -> Dict[str, Any]:
        """
        简单文本搜索
        
        Args:
            query: 搜索查询词
            content_list: 文本内容列表
            file_names: 对应的文件名列表（可选）
        
        Returns:
            搜索结果字典
        """
        if not content_list:
            return {'found': False, 'message': '内容列表为空', 'results': []}
        
        query_words = self.preprocess_query(query)
        if not query_words:
            return {'found': False, 'message': '查询词为空或无效', 'results': []}
        
        results = []
        
        for i, content in enumerate(content_list):
            if not content or content.startswith("错误:"):
                continue
                
            # 预处理内容
            content_words = self.preprocess_query(content)
            content_text = ' '.join(content_words)
            
            # 计算匹配度
            matches = 0
            matched_words = []
            
            for word in query_words:
                if word in content_words:
                    matches += content_words.count(word)
                    matched_words.append(word)
            
            if matches > 0:
                # 计算相关度分数
                relevance = matches / len(content_words) if content_words else 0
                
                result = {
                    'index': i,
                    'file_name': file_names[i] if file_names and i < len(file_names) else f"文档_{i}",
                    'matches': matches,
                    'matched_words': list(set(matched_words)),
                    'relevance_score': relevance,
                    'content_preview': content[:200] + '...' if len(content) > 200 else content
                }
                results.append(result)
        
        # 按相关度排序
        results.sort(key=lambda x: x['relevance_score'], reverse=True)
        
        return {
            'found': len(results) > 0,
            'query': query,
            'total_results': len(results),
            'results': results,
            'message': f'找到 {len(results)} 个匹配结果' if results else '未找到匹配结果'
        }
    
    def boolean_search(self, query: str, content_dict: Dict[str, str], operation: str = 'AND') -> Dict[str, Any]:
        """
        布尔搜索
        
        Args:
            query: 搜索查询，支持多个词用空格分隔
            content_dict: 文件名到内容的字典
            operation: 布尔操作 ('AND', 'OR')
        
        Returns:
            搜索结果字典
        """
        if not content_dict:
            return {'found': False, 'message': '内容字典为空', 'results': []}
        
        query_words = self.preprocess_query(query)
        if not query_words:
            return {'found': False, 'message': '查询词为空或无效', 'results': []}
        
        results = []
        
        for file_name, content in content_dict.items():
            if not content or content.startswith("错误:"):
                continue
            
            content_words = self.preprocess_query(content)
            content_set = set(content_words)
            query_set = set(query_words)
            
            # 布尔逻辑判断
            if operation.upper() == 'AND':
                # 所有查询词都必须存在
                if query_set.issubset(content_set):
                    matched_words = list(query_set)
                    match_found = True
                else:
                    match_found = False
                    matched_words = []
            elif operation.upper() == 'OR':
                # 至少一个查询词存在
                matched_words = list(query_set.intersection(content_set))
                match_found = len(matched_words) > 0
            else:
                continue
            
            if match_found:
                # 计算匹配度
                total_matches = sum(content_words.count(word) for word in matched_words)
                relevance = total_matches / len(content_words) if content_words else 0
                
                result = {
                    'file_name': file_name,
                    'matched_words': matched_words,
                    'total_matches': total_matches,
                    'relevance_score': relevance,
                    'content_preview': content[:200] + '...' if len(content) > 200 else content
                }
                results.append(result)
        
        # 按相关度排序
        results.sort(key=lambda x: x['relevance_score'], reverse=True)
        
        return {
            'found': len(results) > 0,
            'query': query,
            'operation': operation,
            'total_results': len(results),
            'results': results,
            'message': f'使用 {operation} 操作找到 {len(results)} 个匹配结果' if results else f'使用 {operation} 操作未找到匹配结果'
        }
    
    def fuzzy_search(self, query: str, content_list: List[str], file_names: List[str] = None, threshold: float = 0.1) -> Dict[str, Any]:
        """
        模糊搜索
        
        Args:
            query: 搜索查询词
            content_list: 文本内容列表
            file_names: 对应的文件名列表（可选）
            threshold: 相关度阈值
        
        Returns:
            搜索结果字典
        """
        if not content_list:
            return {'found': False, 'message': '内容列表为空', 'results': []}
        
        query_words = self.preprocess_query(query)
        if not query_words:
            return {'found': False, 'message': '查询词为空或无效', 'results': []}
        
        results = []
        
        for i, content in enumerate(content_list):
            if not content or content.startswith("错误:"):
                continue
            
            content_words = self.preprocess_query(content)
            
            # 计算模糊匹配
            fuzzy_matches = 0
            matched_words = []
            
            for query_word in query_words:
                for content_word in content_words:
                    # 简单的子串匹配
                    if query_word in content_word or content_word in query_word:
                        fuzzy_matches += 1
                        matched_words.append(content_word)
                        break
            
            if fuzzy_matches > 0:
                relevance = fuzzy_matches / len(query_words) if query_words else 0
                
                if relevance >= threshold:
                    result = {
                        'index': i,
                        'file_name': file_names[i] if file_names and i < len(file_names) else f"文档_{i}",
                        'fuzzy_matches': fuzzy_matches,
                        'matched_words': list(set(matched_words)),
                        'relevance_score': relevance,
                        'content_preview': content[:200] + '...' if len(content) > 200 else content
                    }
                    results.append(result)
        
        # 按相关度排序
        results.sort(key=lambda x: x['relevance_score'], reverse=True)
        
        return {
            'found': len(results) > 0,
            'query': query,
            'threshold': threshold,
            'total_results': len(results),
            'results': results,
            'message': f'模糊搜索找到 {len(results)} 个匹配结果' if results else '模糊搜索未找到匹配结果'
        }
    
    def keyword_search(self, keywords: List[str], content_dict: Dict[str, str]) -> Dict[str, Any]:
        """
        关键词搜索
        
        Args:
            keywords: 关键词列表
            content_dict: 文件名到内容的字典
        
        Returns:
            搜索结果字典
        """
        if not keywords or not content_dict:
            return {'found': False, 'message': '关键词或内容为空', 'results': []}
        
        # 预处理关键词
        processed_keywords = []
        for keyword in keywords:
            processed_keywords.extend(self.preprocess_query(keyword))
        
        if not processed_keywords:
            return {'found': False, 'message': '关键词无效', 'results': []}
        
        results = []
        
        for file_name, content in content_dict.items():
            if not content or content.startswith("错误:"):
                continue
            
            content_words = self.preprocess_query(content)
            content_set = set(content_words)
            
            # 计算关键词匹配
            matched_keywords = []
            total_matches = 0
            
            for keyword in processed_keywords:
                if keyword in content_set:
                    matched_keywords.append(keyword)
                    total_matches += content_words.count(keyword)
            
            if matched_keywords:
                relevance = len(matched_keywords) / len(processed_keywords)
                
                result = {
                    'file_name': file_name,
                    'matched_keywords': matched_keywords,
                    'total_matches': total_matches,
                    'keyword_coverage': relevance,
                    'content_preview': content[:200] + '...' if len(content) > 200 else content
                }
                results.append(result)
        
        # 按关键词覆盖率排序
        results.sort(key=lambda x: x['keyword_coverage'], reverse=True)
        
        return {
            'found': len(results) > 0,
            'keywords': keywords,
            'total_results': len(results),
            'results': results,
            'message': f'关键词搜索找到 {len(results)} 个匹配结果' if results else '关键词搜索未找到匹配结果'
        }
    def search_with_ranking(self, query: str, content_dict: Dict[str, str], method: str = 'tfidf') -> Dict[str, Any]:
        """
        带排序的高级搜索
        
        Args:
            query: 搜索查询
            content_dict: 文件名到内容的字典
            method: 排序方法 ('tfidf', 'frequency')
        
        Returns:
            搜索结果字典
        """
        if not content_dict:
            return {'found': False, 'message': '内容字典为空', 'results': []}
        
        query_words = self.preprocess_query(query)
        if not query_words:
            return {'found': False, 'message': '查询词无效', 'results': []}
        
        # 过滤有效内容
        valid_contents = {k: v for k, v in content_dict.items() 
                         if v and not v.startswith("错误:")}
        
        if not valid_contents:
            return {'found': False, 'message': '没有有效内容', 'results': []}
        
        if method == 'tfidf':
            return self._tfidf_ranking(query_words, valid_contents)
        elif method == 'frequency':
            return self._frequency_ranking(query_words, valid_contents)
        else:
            return {'found': False, 'message': f'不支持的排序方法: {method}', 'results': []}
    
    def _tfidf_ranking(self, query_words: List[str], content_dict: Dict[str, str]) -> Dict[str, Any]:
        """使用TF-IDF进行排序"""
        try:
            # 准备文档
            documents = []
            file_names = []
            
            for file_name, content in content_dict.items():
                processed_content = ' '.join(self.preprocess_query(content))
                if processed_content:
                    documents.append(processed_content)
                    file_names.append(file_name)
            
            if not documents:
                return {'found': False, 'message': '没有有效文档', 'results': []}
            
            # 构建TF-IDF向量
            vectorizer = TfidfVectorizer(max_features=1000, ngram_range=(1, 2))
            tfidf_matrix = vectorizer.fit_transform(documents)
            feature_names = vectorizer.get_feature_names_out()
            
            # 计算查询向量
            query_text = ' '.join(query_words)
            query_vector = vectorizer.transform([query_text])
            
            # 计算相似度
            
            similarities = cosine_similarity(query_vector, tfidf_matrix).flatten()
            
            results = []
            for i, (file_name, similarity) in enumerate(zip(file_names, similarities)):
                if similarity > 0:
                    result = {
                        'file_name': file_name,
                        'similarity_score': float(similarity),
                        'content_preview': content_dict[file_name][:200] + '...' 
                                         if len(content_dict[file_name]) > 200 
                                         else content_dict[file_name]
                    }
                    results.append(result)
            
            # 按相似度排序
            results.sort(key=lambda x: x['similarity_score'], reverse=True)
            
            return {
                'found': len(results) > 0,
                'method': 'tfidf',
                'query': ' '.join(query_words),
                'total_results': len(results),
                'results': results,
                'message': f'TF-IDF搜索找到 {len(results)} 个匹配结果' if results else 'TF-IDF搜索未找到匹配结果'
            }
            
        except Exception as e:
            return {'found': False, 'message': f'TF-IDF搜索出错: {str(e)}', 'results': []}
    
    def _frequency_ranking(self, query_words: List[str], content_dict: Dict[str, str]) -> Dict[str, Any]:
        """使用词频进行排序"""
        results = []
        
        for file_name, content in content_dict.items():
            content_words = self.preprocess_query(content)
            
            # 计算查询词在文档中的总频次
            total_frequency = 0
            matched_words = []
            
            for word in query_words:
                freq = content_words.count(word)
                if freq > 0:
                    total_frequency += freq
                    matched_words.append(word)
            
            if total_frequency > 0:
                # 计算归一化频率分数
                frequency_score = total_frequency / len(content_words) if content_words else 0
                
                result = {
                    'file_name': file_name,
                    'frequency_score': frequency_score,
                    'total_frequency': total_frequency,
                    'matched_words': matched_words,
                    'content_preview': content[:200] + '...' if len(content) > 200 else content
                }
                results.append(result)
        
        # 按频率分数排序
        results.sort(key=lambda x: x['frequency_score'], reverse=True)
        
        return {
            'found': len(results) > 0,
            'method': 'frequency',
            'query': ' '.join(query_words),
            'total_results': len(results),
            'results': results,
            'message': f'频率搜索找到 {len(results)} 个匹配结果' if results else '频率搜索未找到匹配结果'
        }   

class Visualization:
    def __init__(self):
        """初始化可视化类"""
        self.output_dir = 'server/src/pic/cloud'
        self.ensure_output_dir()
        # 设置中文字体
        plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS', 'DejaVu Sans']
        plt.rcParams['axes.unicode_minus'] = False
        # 停用词
        self.stopwords = self._load_stopwords()

    
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
    
    def generate_wordcloud(self, text_data: Get_file.process_text(), filename: str = 'wordcloud', top_n: int = 50) -> Dict[str, Any]:
        """
        生成词云图
        
        Args:
            text_data: 文本数据（列表或字典）
            filename: 保存文件名
            top_n: 显示前N个词
        
        Returns:
            包含词频统计和保存路径的字典
        """
        try:
            # 预处理文本
            all_words = self.preprocess_text_for_vis(text_data)
            
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
            
            if not top_words:
                return {
                    'success': False,
                    'message': '没有足够的词汇生成词云',
                    'word_freq': {},
                    'save_path': None
                }
            
            # 生成词云
            wordcloud = WordCloud(
                width=1200,
                height=800,
                background_color='white',
                max_words=top_n,
                font_path='simhei.ttf',  # 如果有中文字体文件
                colormap='viridis',
                relative_scaling=0.5,
                random_state=42
            ).generate_from_frequencies(top_words)
            
            # 保存词云图
            save_path = os.path.join(self.output_dir, f'{filename}.png')
            
            plt.figure(figsize=(15, 10))
            plt.imshow(wordcloud, interpolation='bilinear')
            plt.axis('off')
            plt.title(f'词云图 - 前{top_n}个高频词', fontsize=16, pad=20)
            plt.tight_layout()
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            plt.close()
            
            return {
                'success': True,
                'message': f'词云图已生成并保存到 {save_path}',
                'word_freq': top_words,
                'save_path': save_path,
                'total_words': len(all_words),
                'unique_words': len(word_freq)
            }
            
        except Exception as e:
            return {
                'success': False,
                'message': f'生成词云图时发生错误: {str(e)}',
                'word_freq': {},
                'save_path': None
            }
    
    def generate_keyword_ranking(self, text_data, filename: str = 'keyword_ranking', top_n: int = 10) -> Dict[str, Any]:
        """
        生成关键词排行图
        
        Args:
            text_data: 文本数据（列表或字典）
            filename: 保存文件名
            top_n: 显示前N个关键词
        
        Returns:
            包含关键词列表和保存路径的字典
        """
        try:
            # 预处理文本
            all_words = self.preprocess_text_for_vis(text_data)
            
            if not all_words:
                return {
                    'success': False,
                    'message': '没有有效的文本数据生成关键词排行',
                    'keywords': [],
                    'save_path': None
                }
            
            # 统计词频
            word_freq = Counter(all_words)
            top_keywords = word_freq.most_common(top_n)
            
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
            plt.title(f'前{top_n}关键词排行榜', fontsize=14, pad=20)
            
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
            save_path = os.path.join(self.output_dir, f'{filename}.png')
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
                'total_analyzed_words': len(all_words)
            }
            
        except Exception as e:
            return {
                'success': False,
                'message': f'生成关键词排行图时发生错误: {str(e)}',
                'keywords': [],
                'save_path': None
            }
    
    def analyze_file_extensions(self, file_data, generate_chart: bool = True, filename: str = 'file_extensions') -> Dict[str, Any]:
        """
        分析文件后缀分类
        
        Args:
            file_data: 文件数据（列表或字典，包含文件路径或文件名）
            generate_chart: 是否生成图表
            filename: 保存文件名
        
        Returns:
            包含文件分类统计和图表路径的字典
        """
        try:
            file_extensions = []
            
            if isinstance(file_data, list):
                # 处理文件路径列表
                for item in file_data:
                    if isinstance(item, str):
                        ext = os.path.splitext(item)[1].lower()
                        if ext:
                            file_extensions.append(ext[1:])  # 去掉点号
                        else:
                            file_extensions.append('无后缀')
            
            elif isinstance(file_data, dict):
                # 处理字典数据（键为文件路径）
                for key in file_data.keys():
                    if isinstance(key, str):
                        ext = os.path.splitext(key)[1].lower()
                        if ext:
                            file_extensions.append(ext[1:])
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
                plt.title(f'文件类型分布 (总计: {len(file_extensions)} 个文件)', fontsize=14, pad=20)
                
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
    
    def generate_comprehensive_report(self, text_data, file_data=None, report_name: str = 'comprehensive_report') -> Dict[str, Any]:
        """
        生成综合可视化报告
        
        Args:
            text_data: 文本数据
            file_data: 文件数据（可选）
            report_name: 报告名称
        
        Returns:
            综合报告结果
        """
        try:
            results = {
                'report_name': report_name,
                'timestamp': pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S'),
                'wordcloud': None,
                'keyword_ranking': None,
                'file_analysis': None
            }
            
            # 生成词云图
            wordcloud_result = self.generate_wordcloud(text_data, f'{report_name}_wordcloud', 50)
            results['wordcloud'] = wordcloud_result
            
            # 生成关键词排行
            ranking_result = self.generate_keyword_ranking(text_data, f'{report_name}_ranking', 10)
            results['keyword_ranking'] = ranking_result
            
            # 如果提供了文件数据，进行文件分析
            if file_data:
                file_result = self.analyze_file_extensions(file_data, True, f'{report_name}_files')
                results['file_analysis'] = file_result
            
            # 生成汇总信息
            summary = {
                'wordcloud_generated': wordcloud_result.get('success', False),
                'ranking_generated': ranking_result.get('success', False),
                'file_analysis_generated': file_result.get('success', False) if file_data else False,
                'total_unique_words': wordcloud_result.get('unique_words', 0),
                'total_words_analyzed': wordcloud_result.get('total_words', 0),
                'top_keywords_count': len(ranking_result.get('keywords', [])),
                'file_types_found': file_result.get('unique_extensions', 0) if file_data else 0,
                'total_files_analyzed': file_result.get('total_files', 0) if file_data else 0
            }
            
            results['summary'] = summary
            results['success'] = True
            results['message'] = '综合可视化报告生成完成'
            
            return results
            
        except Exception as e:
            return {
                'success': False,
                'message': f'生成综合报告时发生错误: {str(e)}',
                'report_name': report_name,
                'timestamp': pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')
            }
    
    def create_word_frequency_chart(self, text_data, chart_type: str = 'bar', top_n: int = 20, filename: str = 'word_frequency') -> Dict[str, Any]:
        """
        创建词频图表
        
        Args:
            text_data: 文本数据
            chart_type: 图表类型 ('bar', 'line', 'pie')
            top_n: 显示前N个词
            filename: 保存文件名
        
        Returns:
            图表生成结果
        """
        try:
            # 预处理文本
            all_words = self.preprocess_text_for_vis(text_data)
            
            if not all_words:
                return {
                    'success': False,
                    'message': '没有有效的文本数据生成图表',
                    'save_path': None
                }
            
            # 统计词频
            word_freq = Counter(all_words)
            top_words = word_freq.most_common(top_n)
            
            if not top_words:
                return {
                    'success': False,
                    'message': '没有足够的词汇生成图表',
                    'save_path': None
                }
            
            words = [item[0] for item in top_words]
            frequencies = [item[1] for item in top_words]
            
            plt.figure(figsize=(12, 8))
            
            if chart_type == 'bar':
                plt.bar(range(len(words)), frequencies, color='lightblue', alpha=0.8)
                plt.xticks(range(len(words)), words, rotation=45, ha='right')
                plt.ylabel('词频')
                plt.xlabel('词汇')
                
            elif chart_type == 'line':
                plt.plot(range(len(words)), frequencies, marker='o', linewidth=2, markersize=6)
                plt.xticks(range(len(words)), words, rotation=45, ha='right')
                plt.ylabel('词频')
                plt.xlabel('词汇')
                plt.grid(True, alpha=0.3)
                
            elif chart_type == 'pie':
                plt.pie(frequencies, labels=words, autopct='%1.1f%%', startangle=90)
                plt.axis('equal')
            
            plt.title(f'词频分析 - {chart_type.upper()}图 (前{top_n}个)', fontsize=14, pad=20)
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
                'words_analyzed': len(all_words),
                'top_words': top_words
            }
            
        except Exception as e:
            return {
                'success': False,
                'message': f'生成{chart_type}图时发生错误: {str(e)}',
                'save_path': None
            }
    
    def get_visualization_summary(self, text_data, file_data=None) -> Dict[str, Any]:
        """
        获取可视化数据摘要（不生成图片）
        
        Args:
            text_data: 文本数据
            file_data: 文件数据（可选）
        
        Returns:
            数据摘要
        """
        try:
            summary = {
                'text_analysis': {},
                'file_analysis': {},
                'timestamp': pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')
            }
            
            # 文本分析
            if text_data:
                all_words = self.preprocess_text_for_vis(text_data)
                if all_words:
                    word_freq = Counter(all_words)
                    
                    summary['text_analysis'] = {
                        'total_words': len(all_words),
                        'unique_words': len(word_freq),
                        'top_10_words': word_freq.most_common(10),
                        'average_word_length': sum(len(word) for word in all_words) / len(all_words),
                        'longest_word': max(all_words, key=len) if all_words else '',
                        'word_diversity': len(word_freq) / len(all_words) if all_words else 0
                    }
            
            # 文件分析
            if file_data:
                file_extensions = []
                
                if isinstance(file_data, list):
                    for item in file_data:
                        if isinstance(item, str):
                            ext = os.path.splitext(item)[1].lower()
                            file_extensions.append(ext[1:] if ext else '无后缀')
                elif isinstance(file_data, dict):
                    for key in file_data.keys():
                        if isinstance(key, str):
                            ext = os.path.splitext(key)[1].lower()
                            file_extensions.append(ext[1:] if ext else '无后缀')
                
                if file_extensions:
                    ext_counter = Counter(file_extensions)
                    summary['file_analysis'] = {
                        'total_files': len(file_extensions),
                        'unique_extensions': len(ext_counter),
                        'extension_distribution': dict(ext_counter.most_common()),
                        'most_common_extension': ext_counter.most_common(1)[0] if ext_counter else None
                    }
            
            return {
                'success': True,
                'summary': summary,
                'message': '数据摘要生成完成'
            }
            
        except Exception as e:
            return {
                'success': False,
                'message': f'生成数据摘要时发生错误: {str(e)}',
                'summary': {}
            }
    
    def batch_generate_visualizations(self, datasets: List[Dict], output_prefix: str = 'batch') -> Dict[str, Any]:
        """
        批量生成可视化图表
        
        Args:
            datasets: 数据集列表，每个元素包含 {'name': str, 'text_data': data, 'file_data': data}
            output_prefix: 输出文件前缀
        
        Returns:
            批量处理结果
        """
        try:
            results = {
                'batch_name': output_prefix,
                'timestamp': pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S'),
                'processed_datasets': [],
                'success_count': 0,
                'error_count': 0
            }
            
            for i, dataset in enumerate(datasets):
                dataset_name = dataset.get('name', f'dataset_{i+1}')
                text_data = dataset.get('text_data')
                file_data = dataset.get('file_data')
                
                try:
                    # 为每个数据集生成可视化
                    dataset_result = self.generate_comprehensive_report(
                        text_data, 
                        file_data, 
                        f'{output_prefix}_{dataset_name}'
                    )
                    
                    dataset_result['dataset_name'] = dataset_name
                    results['processed_datasets'].append(dataset_result)
                    
                    if dataset_result.get('success', False):
                        results['success_count'] += 1
                    else:
                        results['error_count'] += 1
                        
                except Exception as e:
                    error_result = {
                        'dataset_name': dataset_name,
                        'success': False,
                        'message': f'处理数据集 {dataset_name} 时发生错误: {str(e)}'
                    }
                    results['processed_datasets'].append(error_result)
                    results['error_count'] += 1
            
            results['success'] = results['success_count'] > 0
            results['message'] = f'批量处理完成：成功 {results["success_count"]} 个，失败 {results["error_count"]} 个'
            
            return results
            
        except Exception as e:
            return {
                'success': False,
                'message': f'批量生成可视化时发生错误: {str(e)}',
                'batch_name': output_prefix,
                'timestamp': pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')
            }


class APIHandler:
    """
    前后端交互处理类
    负责处理Vue前端发送的请求并返回相应的数据
    """
    
    def __init__(self):
        """初始化API处理器"""
        self.app = Flask(__name__)
        CORS(self.app)  # 允许跨域请求
        
        # 初始化各个功能模块
        self.file_processor = None
        self.search_engine = scearch_content()
        self.visualizer = Visualization()
        
        # 存储处理结果的缓存
        self.cache = {}
        
        # 设置路由
        self._setup_routes()
        
        # 配置日志
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
    
    def _setup_routes(self):
        """设置API路由"""
        
        # 文件处理相关路由
        self.app.route('/api/upload-config', methods=['POST'])(self.upload_config)
        self.app.route('/api/process-files', methods=['POST'])(self.process_files)
        self.app.route('/api/get-file-list', methods=['GET'])(self.get_file_list)
        
        # 搜索相关路由
        self.app.route('/api/search/simple', methods=['POST'])(self.simple_search)
        self.app.route('/api/search/boolean', methods=['POST'])(self.boolean_search)
        self.app.route('/api/search/fuzzy', methods=['POST'])(self.fuzzy_search)
        self.app.route('/api/search/keyword', methods=['POST'])(self.keyword_search)
        self.app.route('/api/search/advanced', methods=['POST'])(self.advanced_search)
        
        # 可视化相关路由
        self.app.route('/api/visualization/wordcloud', methods=['POST'])(self.generate_wordcloud)
        self.app.route('/api/visualization/ranking', methods=['POST'])(self.generate_ranking)
        self.app.route('/api/visualization/file-analysis', methods=['POST'])(self.analyze_files)
        self.app.route('/api/visualization/comprehensive', methods=['POST'])(self.comprehensive_report)
        
        # 数据统计路由
        self.app.route('/api/statistics/word-count', methods=['GET'])(self.get_word_statistics)
        self.app.route('/api/statistics/file-stats', methods=['GET'])(self.get_file_statistics)
        
        # 文件下载路由
        self.app.route('/api/download/<path:filename>', methods=['GET'])(self.download_file)
        
        # 健康检查路由
        self.app.route('/api/health', methods=['GET'])(self.health_check)
    
    def _create_response(self, success: bool, data: Any = None, message: str = "", error: str = ""):
        """创建标准化的API响应"""
        response = {
            'success': success,
            'timestamp': datetime.now().isoformat(),
            'message': message
        }
        
        if success:
            response['data'] = data
        else:
            response['error'] = error
            
        return jsonify(response)
    
    def _handle_error(self, e: Exception, operation: str):
        """统一错误处理"""
        error_msg = f"{operation}时发生错误: {str(e)}"
        self.logger.error(f"{error_msg}\n{traceback.format_exc()}")
        return self._create_response(False, error=error_msg)
    
    # 文件处理相关API
    def upload_config(self):
        """上传配置文件并初始化文件处理器"""
        try:
            data = request.get_json()
            config_path = data.get('config_path')
            
            if not config_path:
                return self._create_response(False, error="配置文件路径不能为空")
            
            # 初始化文件处理器
            self.file_processor = Get_file(config_path)
            
            # 生成唯一的会话ID
            session_id = str(uuid.uuid4())
            self.cache[session_id] = {'file_processor': self.file_processor}
            
            return self._create_response(True, 
                                       data={'session_id': session_id}, 
                                       message="配置文件上传成功")
            
        except Exception as e:
            return self._handle_error(e, "上传配置文件")
    
    def process_files(self):
        """处理文件并构建索引"""
        try:
            data = request.get_json()
            session_id = data.get('session_id')
            
            if not session_id or session_id not in self.cache:
                return self._create_response(False, error="无效的会话ID")
            
            file_processor = self.cache[session_id]['file_processor']
            
            # 读取文件
            file_processor.read_dir_file()
            
            # 构建索引
            file_processor.build_index()
            
            # 统计信息
            stats = {
                'total_files': len(file_processor.processed_files_content),
                'valid_files': len([f for f in file_processor.processed_files_content.values() 
                                  if not f.startswith("错误:")]),
                'total_words': len(file_processor.inverted_index),
                'file_types': {}
            }
            
            # 统计文件类型
            for file_path in file_processor.file_path:
                ext = os.path.splitext(file_path)[1][1:].lower()
                stats['file_types'][ext] = stats['file_types'].get(ext, 0) + 1
            
            return self._create_response(True, 
                                       data=stats, 
                                       message="文件处理完成")
            
        except Exception as e:
            return self._handle_error(e, "处理文件")
    
    def get_file_list(self):
        """获取文件列表"""
        try:
            session_id = request.args.get('session_id')
            
            if not session_id or session_id not in self.cache:
                return self._create_response(False, error="无效的会话ID")
            
            file_processor = self.cache[session_id]['file_processor']
            
            file_list = []
            for file_path, content in file_processor.processed_files_content.items():
                file_info = {
                    'path': file_path,
                    'name': os.path.basename(file_path),
                    'extension': os.path.splitext(file_path)[1][1:],
                    'status': 'success' if not content.startswith("错误:") else 'error',
                    'size': len(content) if not content.startswith("错误:") else 0,
                    'preview': content[:100] + '...' if len(content) > 100 else content
                }
                file_list.append(file_info)
            
            return self._create_response(True, 
                                       data={'files': file_list}, 
                                       message="文件列表获取成功")
            
        except Exception as e:
            return self._handle_error(e, "获取文件列表")
    
    # 搜索相关API
    def simple_search(self):
        """简单搜索"""
        try:
            data = request.get_json()
            session_id = data.get('session_id')
            query = data.get('query', '')
            
            if not session_id or session_id not in self.cache:
                return self._create_response(False, error="无效的会话ID")
            
            file_processor = self.cache[session_id]['file_processor']
            content_list = list(file_processor.processed_files_content.values())
            file_names = list(file_processor.processed_files_content.keys())
            
            result = self.search_engine.simple_search(query, content_list, file_names)
            
            return self._create_response(True, 
                                       data=result, 
                                       message="简单搜索完成")
            
        except Exception as e:
            return self._handle_error(e, "简单搜索")
    
    def boolean_search(self):
        """布尔搜索"""
        try:
            data = request.get_json()
            session_id = data.get('session_id')
            query = data.get('query', '')
            operation = data.get('operation', 'AND')
            
            if not session_id or session_id not in self.cache:
                return self._create_response(False, error="无效的会话ID")
            
            file_processor = self.cache[session_id]['file_processor']
            content_dict = file_processor.processed_files_content
            
            result = self.search_engine.boolean_search(query, content_dict, operation)
            
            return self._create_response(True, 
                                       data=result, 
                                       message="布尔搜索完成")
            
        except Exception as e:
            return self._handle_error(e, "布尔搜索")
    
    def fuzzy_search(self):
        """模糊搜索"""
        try:
            data = request.get_json()
            session_id = data.get('session_id')
            query = data.get('query', '')
            threshold = data.get('threshold', 0.1)
            
            if not session_id or session_id not in self.cache:
                return self._create_response(False, error="无效的会话ID")
            
            file_processor = self.cache[session_id]['file_processor']
            content_list = list(file_processor.processed_files_content.values())
            file_names = list(file_processor.processed_files_content.keys())
            
            result = self.search_engine.fuzzy_search(query, content_list, file_names, threshold)
            
            return self._create_response(True, 
                                       data=result, 
                                       message="模糊搜索完成")
            
        except Exception as e:
            return self._handle_error(e, "模糊搜索")
    
    def keyword_search(self):
        """关键词搜索"""
        try:
            data = request.get_json()
            session_id = data.get('session_id')
            keywords = data.get('keywords', [])
            
            if not session_id or session_id not in self.cache:
                return self._create_response(False, error="无效的会话ID")
            
            file_processor = self.cache[session_id]['file_processor']
            content_dict = file_processor.processed_files_content
            
            result = self.search_engine.keyword_search(keywords, content_dict)
            
            return self._create_response(True, 
                                       data=result, 
                                       message="关键词搜索完成")
            
        except Exception as e:
            return self._handle_error(e, "关键词搜索")
    
    def advanced_search(self):
        """高级搜索"""
        try:
            data = request.get_json()
            session_id = data.get('session_id')
            query = data.get('query', '')
            method = data.get('method', 'tfidf')
            
            if not session_id or session_id not in self.cache:
                return self._create_response(False, error="无效的会话ID")
            
            file_processor = self.cache[session_id]['file_processor']
            content_dict = file_processor.processed_files_content
            
            result = self.search_engine.search_with_ranking(query, content_dict, method)
            
            return self._create_response(True, 
                                       data=result, 
                                       message="高级搜索完成")
            
        except Exception as e:
            return self._handle_error(e, "高级搜索")
    
    # 可视化相关API
    def generate_wordcloud(self):
        """生成词云图"""
        try:
            data = request.get_json()
            session_id = data.get('session_id')
            filename = data.get('filename', 'wordcloud')
            top_n = data.get('top_n', 50)
            
            if not session_id or session_id not in self.cache:
                return self._create_response(False, error="无效的会话ID")
            
            file_processor = self.cache[session_id]['file_processor']
            text_data = list(file_processor.processed_files_content.values())
            
            result = self.visualizer.generate_wordcloud(text_data, filename, top_n)
            
            return self._create_response(True, 
                                       data=result, 
                                       message="词云图生成完成")
            
        except Exception as e:
            return self._handle_error(e, "生成词云图")
    
    def generate_ranking(self):
        """生成关键词排行图"""
        try:
            data = request.get_json()
            session_id = data.get('session_id')
            filename = data.get('filename', 'ranking')
            top_n = data.get('top_n', 10)
            
            if not session_id or session_id not in self.cache:
                return self._create_response(False, error="无效的会话ID")
            
            file_processor = self.cache[session_id]['file_processor']
            text_data = list(file_processor.processed_files_content.values())
            
            result = self.visualizer.generate_keyword_ranking(text_data, filename, top_n)
            return self._create_response(True, 
                                       data=result, 
                                       message="关键词排行图生成完成")
            
        except Exception as e:
            return self._handle_error(e, "生成关键词排行图")
    
    def analyze_files(self):
        """文件类型分析"""
        try:
            data = request.get_json()
            session_id = data.get('session_id')
            filename = data.get('filename', 'file_analysis')
            generate_chart = data.get('generate_chart', True)
            
            if not session_id or session_id not in self.cache:
                return self._create_response(False, error="无效的会话ID")
            
            file_processor = self.cache[session_id]['file_processor']
            file_data = list(file_processor.processed_files_content.keys())
            
            result = self.visualizer.analyze_file_extensions(file_data, generate_chart, filename)
            
            return self._create_response(True, 
                                       data=result, 
                                       message="文件分析完成")
            
        except Exception as e:
            return self._handle_error(e, "文件分析")
    
    def comprehensive_report(self):
        """生成综合报告"""
        try:
            data = request.get_json()
            session_id = data.get('session_id')
            report_name = data.get('report_name', 'comprehensive_report')
            
            if not session_id or session_id not in self.cache:
                return self._create_response(False, error="无效的会话ID")
            
            file_processor = self.cache[session_id]['file_processor']
            text_data = list(file_processor.processed_files_content.values())
            file_data = list(file_processor.processed_files_content.keys())
            result = self.visualizer.generate_comprehensive_report(text_data, file_data, report_name)
            
            return self._create_response(True, 
                                       data=result, 
                                       message="综合报告生成完成")
            
        except Exception as e:
            return self._handle_error(e, "生成综合报告")
    
    # 统计信息API
    def get_word_statistics(self):
        """获取词频统计"""
        try:
            session_id = request.args.get('session_id')
            top_n = int(request.args.get('top_n', 20))
            
            if not session_id or session_id not in self.cache:
                return self._create_response(False, error="无效的会话ID")
            
            file_processor = self.cache[session_id]['file_processor']
            word_stats = file_processor.word_count()
            
            # 获取前N个高频词
            top_words = dict(list(word_stats.items())[:top_n])
            
            stats = {
                'total_unique_words': len(word_stats),
                'top_words': top_words,
                'word_distribution': {
                    'single_char': len([w for w in word_stats.keys() if len(w) == 1]),
                    'two_char': len([w for w in word_stats.keys() if len(w) == 2]),
                    'multi_char': len([w for w in word_stats.keys() if len(w) > 2])
                }
            }
            
            return self._create_response(True, 
                                       data=stats, 
                                       message="词频统计获取成功")
            
        except Exception as e:
            return self._handle_error(e, "获取词频统计")
    
    def get_file_statistics(self):
        """获取文件统计信息"""
        try:
            session_id = request.args.get('session_id')
            
            if not session_id or session_id not in self.cache:
                return self._create_response(False, error="无效的会话ID")
            
            file_processor = self.cache[session_id]['file_processor']
            
            # 统计文件信息
            total_files = len(file_processor.processed_files_content)
            valid_files = len([f for f in file_processor.processed_files_content.values() 
                             if not f.startswith("错误:")])
            error_files = total_files - valid_files
            
            # 文件类型统计
            file_types = {}
            file_sizes = []
            
            for file_path, content in file_processor.processed_files_content.items():
                ext = os.path.splitext(file_path)[1][1:].lower() or '无后缀'
                file_types[ext] = file_types.get(ext, 0) + 1
                
                if not content.startswith("错误:"):
                    file_sizes.append(len(content))
            
            stats = {
                'total_files': total_files,
                'valid_files': valid_files,
                'error_files': error_files,
                'success_rate': (valid_files / total_files * 100) if total_files > 0 else 0,
                'file_types': file_types,
                'size_stats': {
                    'total_size': sum(file_sizes),
                    'average_size': sum(file_sizes) / len(file_sizes) if file_sizes else 0,
                    'max_size': max(file_sizes) if file_sizes else 0,
                    'min_size': min(file_sizes) if file_sizes else 0
                }
            }
            
            return self._create_response(True, 
                                       data=stats, 
                                       message="文件统计信息获取成功")
            
        except Exception as e:
            return self._handle_error(e, "获取文件统计信息")
    
    # 文件下载API
    def download_file(self, filename):
        """下载生成的图片文件"""
        try:
            file_path = os.path.join('server/src/pic/cloud', filename)
            
            if not os.path.exists(file_path):
                return self._create_response(False, error="文件不存在")
            
            return send_file(file_path, as_attachment=True)
            
        except Exception as e:
            return self._handle_error(e, "下载文件")
    
    # 健康检查API
    def health_check(self):
        """健康检查"""
        return self._create_response(True, 
                                   data={'status': 'healthy', 'version': '1.0.0'}, 
                                   message="服务运行正常")
    
    # 缓存管理方法
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
            'session_ids': list(self.cache.keys())
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

                                       

        
