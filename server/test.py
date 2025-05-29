import os
import json
import tempfile
import unittest
from unittest.mock import patch, mock_open
import sys
import pandas as pd

# 添加server目录到路径
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from server.utils import Get_file, scearch_content, Visualization

class TestGetFile(unittest.TestCase):
    """测试Get_file类的基本功能"""
    
    def setUp(self):
        """设置测试环境"""
        # 创建临时目录和文件
        self.test_dir = tempfile.mkdtemp()
        self.config_file = os.path.join(self.test_dir, 'config.json')
        self.data_dir = os.path.join(self.test_dir, 'data')
        os.makedirs(self.data_dir, exist_ok=True)
        
        # 创建配置文件
        config_data = {'json_path': self.data_dir}
        with open(self.config_file, 'w', encoding='utf-8') as f:
            json.dump(config_data, f)
        
        # 创建测试文件
        self.create_test_files()
    
    def create_test_files(self):
        """创建测试用的各种格式文件"""
        # 创建txt文件
        txt_file = os.path.join(self.data_dir, 'test.txt')
        with open(txt_file, 'w', encoding='utf-8') as f:
            f.write('这是一个测试文本文件。包含中文和English内容。')
        
        # 创建json文件
        json_file = os.path.join(self.data_dir, 'test.json')
        test_json = {'name': '测试', 'content': '这是json测试内容'}
        with open(json_file, 'w', encoding='utf-8') as f:
            json.dump(test_json, f, ensure_ascii=False)
        
        # 创建csv文件
        csv_file = os.path.join(self.data_dir, 'test.csv')
        df = pd.DataFrame({'列1': ['数据1', '数据2'], '列2': ['内容1', '内容2']})
        df.to_csv(csv_file, index=False, encoding='utf-8')
    
    def test_init(self):
        """测试初始化功能"""
        get_file = Get_file(self.config_file)
        self.assertEqual(get_file.json_dir, self.data_dir)
        self.assertIsInstance(get_file.file_path, list)
        self.assertIsInstance(get_file.processed_files_content, dict)
    
    def test_read_txt_content(self):
        """测试读取txt文件"""
        get_file = Get_file(self.config_file)
        txt_path = os.path.join(self.data_dir, 'test.txt')
        content = get_file.read_content('txt', txt_path)
        self.assertIn('测试文本文件', content)
        self.assertNotIn('错误:', content)
    
    def test_read_json_content(self):
        """测试读取json文件"""
        get_file = Get_file(self.config_file)
        json_path = os.path.join(self.data_dir, 'test.json')
        content = get_file.read_content('json', json_path)
        self.assertIn('测试', content)
        self.assertNotIn('错误:', content)
    
    def test_read_csv_content(self):
        """测试读取csv文件"""
        get_file = Get_file(self.config_file)
        csv_path = os.path.join(self.data_dir, 'test.csv')
        content = get_file.read_content('csv', csv_path)
        self.assertIn('数据1', content)
        self.assertNotIn('错误:', content)
    
    def test_read_dir_file(self):
        """测试读取目录中的所有文件"""
        get_file = Get_file(self.config_file)
        get_file.read_dir_file()
        
        # 检查是否读取了所有文件
        self.assertGreater(len(get_file.processed_files_content), 0)
        self.assertGreater(len(get_file.content_list), 0)
    
    def test_process_text(self):
        """测试文本预处理功能"""
        get_file = Get_file(self.config_file)
        test_content = ['这是一个测试文本，包含标点符号！@#$%^&*()']
        processed = get_file.process_text(test_content)
        
        self.assertIsInstance(processed, list)
        self.assertGreater(len(processed), 0)
    
    def test_word_count(self):
        """测试词频统计功能"""
        get_file = Get_file(self.config_file)
        get_file.read_dir_file()
        word_dict = get_file.word_count()
        
        self.assertIsInstance(word_dict, dict)
        # 检查是否按频率降序排列
        if len(word_dict) > 1:
            values = list(word_dict.values())
            self.assertGreaterEqual(values[0], values[-1])
    
    def test_build_index(self):
        """测试构建倒排索引"""
        get_file = Get_file(self.config_file)
        get_file.read_dir_file()
        get_file.build_index()
        
        self.assertIsInstance(get_file.inverted_index, dict)
        self.assertIsInstance(get_file.file_word_count, dict)
    
    def tearDown(self):
        """清理测试环境"""
        import shutil
        shutil.rmtree(self.test_dir)


class TestSearchContent(unittest.TestCase):
    """测试搜索功能类"""
    
    def setUp(self):
        """设置测试环境"""
        self.search_engine = scearch_content()
        self.test_content = [
            '这是第一个测试文档，包含Python编程内容',
            '这是第二个文档，讨论机器学习和人工智能',
            '第三个文档介绍数据分析和可视化技术'
        ]
        self.test_file_names = ['doc1.txt', 'doc2.txt', 'doc3.txt']
        self.test_content_dict = dict(zip(self.test_file_names, self.test_content))
    
    def test_preprocess_query(self):
        """测试查询预处理"""
        query = '测试查询！@#$%^&*()'
        processed = self.search_engine.preprocess_query(query)
        
        self.assertIsInstance(processed, list)
        self.assertNotIn('！', ' '.join(processed))
    
    def test_simple_search(self):
        """测试简单搜索"""
        result = self.search_engine.simple_search('Python', self.test_content, self.test_file_names)
        
        self.assertIsInstance(result, dict)
        self.assertIn('found', result)
        self.assertIn('results', result)
        
        if result['found']:
            self.assertGreater(len(result['results']), 0)
    
    def test_boolean_search_and(self):
        """测试布尔搜索AND操作"""
        result = self.search_engine.boolean_search('测试 文档', self.test_content_dict, 'AND')
        
        self.assertIsInstance(result, dict)
        self.assertEqual(result['operation'], 'AND')
        self.assertIn('results', result)
    
    def test_boolean_search_or(self):
        """测试布尔搜索OR操作"""
        result = self.search_engine.boolean_search('Python 机器学习', self.test_content_dict, 'OR')
        
        self.assertIsInstance(result, dict)
        self.assertEqual(result['operation'], 'OR')
        self.assertIn('results', result)
    
    def test_fuzzy_search(self):
        """测试模糊搜索"""
        result = self.search_engine.fuzzy_search('程序', self.test_content, self.test_file_names, 0.1)
        
        self.assertIsInstance(result, dict)
        self.assertIn('threshold', result)
        self.assertIn('results', result)
    
    def test_keyword_search(self):
        """测试关键词搜索"""
        keywords = ['Python', '机器学习', '数据']
        result = self.search_engine.keyword_search(keywords, self.test_content_dict)
        
        self.assertIsInstance(result, dict)
        self.assertIn('keywords', result)
        self.assertIn('results', result)
    
    @patch('server.utils.TfidfVectorizer')
    def test_search_with_ranking_tfidf(self, mock_vectorizer):
        """测试TF-IDF排序搜索"""
        # 模拟TfidfVectorizer
        mock_vectorizer.return_value.fit_transform.return_value = [[0.5, 0.3], [0.2, 0.8]]
        mock_vectorizer.return_value.transform.return_value = [[0.4, 0.6]]
        mock_vectorizer.return_value.get_feature_names_out.return_value = ['word1', 'word2']
        
        result = self.search_engine.search_with_ranking('测试', self.test_content_dict, 'tfidf')
        
        self.assertIsInstance(result, dict)
        self.assertEqual(result.get('method'), 'tfidf')
    
    def test_search_with_ranking_frequency(self):
        """测试频率排序搜索"""
        result = self.search_engine.search_with_ranking('测试', self.test_content_dict, 'frequency')
        
        self.assertIsInstance(result, dict)
        self.assertEqual(result.get('method'), 'frequency')


class TestVisualization(unittest.TestCase):
    """测试可视化功能类"""
    
    def setUp(self):
        """设置测试环境"""
        self.visualizer = Visualization()
        self.test_text_data = [
            '这是测试文本数据，包含各种词汇',
            '另一个测试文档，用于可视化分析',
            '第三个文档包含更多测试内容'
        ]
        self.test_file_data = ['test1.txt', 'test2.py', 'test3.json', 'test4.csv']
    
    def test_preprocess_text_for_vis(self):
        """测试可视化文本预处理"""
        processed = self.visualizer.preprocess_text_for_vis(self.test_text_data)
        
        self.assertIsInstance(processed, list)
        self.assertGreater(len(processed), 0)
    
    @patch('matplotlib.pyplot.savefig')
    @patch('matplotlib.pyplot.figure')
    def test_generate_wordcloud(self, mock_figure, mock_savefig):
        """测试词云生成"""
        result = self.visualizer.generate_wordcloud(self.test_text_data, 'test_wordcloud', 10)
        
        self.assertIsInstance(result, dict)
        self.assertIn('success', result)
        self.assertIn('word_freq', result)
    
    @patch('matplotlib.pyplot.savefig')
    @patch('matplotlib.pyplot.figure')
    def test_generate_keyword_ranking(self, mock_figure, mock_savefig):
        """测试关键词排行图生成"""
        result = self.visualizer.generate_keyword_ranking(self.test_text_data, 'test_ranking', 5)
        
        self.assertIsInstance(result, dict)
        self.assertIn('success', result)
        self.assertIn('keywords', result)
    
    def test_analyze_file_extensions(self):
        """测试文件扩展名分析"""
        result = self.visualizer.analyze_file_extensions(self.test_file_data, False)
        
        self.assertIsInstance(result, dict)
        self.assertIn('success', result)
        self.assertIn('extension_stats', result)
        
        if result['success']:
            self.assertGreater(len(result['extension_stats']), 0)
    
    def test_get_visualization_summary(self):
        """测试可视化数据摘要"""
        result = self.visualizer.get_visualization_summary(self.test_text_data, self.test_file_data)
        
        self.assertIsInstance(result, dict)
        self.assertIn('success', result)
        self.assertIn('summary', result)
        
        if result['success']:
            summary = result['summary']
            self.assertIn('text_analysis', summary)
            self.assertIn('file_analysis', summary)


class TestIntegration(unittest.TestCase):
    """集成测试"""
    
    def setUp(self):
        """设置集成测试环境"""
        # 创建临时测试环境
        self.test_dir = tempfile.mkdtemp()
        self.config_file = os.path.join(self.test_dir, 'config.json')
        self.data_dir = os.path.join(self.test_dir, 'data')
        os.makedirs(self.data_dir, exist_ok=True)
        
        # 创建配置文件
        config_data = {'json_path': self.data_dir}
        with open(self.config_file, 'w', encoding='utf-8') as f:
            json.dump(config_data, f)
        
        # 创建测试文件
        test_files = {
            'doc1.txt': '人工智能是计算机科学的一个分支，致力于创建智能机器',
            'doc2.txt': '机器学习是人工智能的子集，使用算法从数据中学习',
            'doc3.txt': '深度学习是机器学习的一种方法，使用神经网络'
        }
        
        for filename, content in test_files.items():
            filepath = os.path.join(self.data_dir, filename)
            with open(filepath, 
