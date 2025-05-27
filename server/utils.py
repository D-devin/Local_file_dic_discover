import os 
import json 
import logging
import requests
import numpy as np 
import pandas as pd
import jieba 
import re
from typing import List, Dict, Any
from doxter import docx
import fitz
from bs4 import BeautifulSoup
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer

## 读取传回来的json文件中的路径，并且将路径中的文件读出文本，
## 
class Get_file:
    def __init__(self, json_path: str):
        self.json_path = json_path
        open_json = open(self.json_path, 'r', endcoding = 'utf-8')
        #读取json文件里的路径
        self.json_dir = json.load(open_json)
        self.json_dir = self.json_dir['json_path']
        open_json.close()
        self.file_path = []
        self.file_name = []
        self.kind = ['']
        self.all_conent =[]
        self.processed_files_content: Dict[str, str] = {} # 用于存储结果
        self.word_dict = {}
         
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
        
    def read_dir_file(self) -> Dict[str, str]:
        for root, dirs, files in os.walk(self.json_dir):
            for file_name in files: # 使用 file_name 避免与外部可能的 file 变量冲突
                self.file_path.append(os.path.join(root,file_name))

        for file_path_item in self.file_path: # 遍历收集到的文件路径
            content_or_error_msg = f"错误: 未知或不支持的文件类型 '{os.path.basename(file_path_item)}'." # 默认消息
            file_name_extension = os.path.splitext(file_path_item)[1][1:]
            content_or_error_msg = self.reading_txt(file_path_item,file_path_item)
          
            self.processed_files_content[file_path_item] = content_or_error_msg
            self.content_list.append(content_or_error_msg)

    def process_text(self, content:list)-> List[str]:
        process_content = []
        stopword_path = 'server/src/stopwords.txt'
        for text in content:
            text = re.sub(r'{^\u4e00 -\u9fa5]+', '', text)  # 删除非中文字符}')
            text = ''.join(jieba.cut(text))
            stopwords = open(stopword_path, 'r', encoding='utf-8').readlines()
            stopwords = [word.strip() for word in stopwords]
            text = ' '.join([word for word in text.split() if word not in stopwords])
            process_content.append(text)
        return process_content
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
    
    def word_count(self, content: str):
        word_list = content.split()
        for word in word_list:
            self.word_dict[word] = self.word_dict.get(word, 0) + 1 
        sorted_word_dict = sorted(self.word_dict.items(), key = lambda X: X[1],reverse = True)
        return sorted_word_dict
    
    def build_index(self):
        

class scearch_content:
    pass

        
