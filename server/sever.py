import os 
import json 
import logging
import requests
from typing import List, Dict, Any
from flask import Flask, request, jsonify, send_file
from flask_cors import CORS
from datetime import datetime
import uuid
from utils import Get_file, scearch_content, Visualization
class APIHandler:
    def __init__(self):
        """初始化API"""
        self.app = Flask(__name__)
        CORS(self.app)
        # init 功能模块 
        self.file_reader = None# 实现GET_file 
        self.search_engine = scearch_content()#搜索策略实现 
        self.visual_processor = Visualization()# 可视化模块实现 
        self._setup_routes()
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
        self.app.add_url_rule('/process_folder', 'process_folder', self.process_folder, methods=['POST'])
        # 搜索功能路由
        self.app.add_url_rule('/textrank_search', 'textrank_search', self.textrank_search, methods=['POST'])
        self.app.add_url_rule('/tfidf_search', 'tfidf_search', self.tfidf_search, methods=['POST'])
        self.app.add_url_rule('/lsi_search', 'lsi_search', self.lsi_search, methods=['POST'])

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
    def process_folder(self):
        """直接处理文件夹路径"""
        try:
            data = request.get_json()
            if not data:
                return self._create_response(False, error="请求数据为空")
            
            folder_path = data.get('folder_path')
            if not folder_path:
                return self._create_response(False, error="文件夹路径不能为空")
            
            # 检查文件夹是否存在
            if not os.path.isdir(folder_path):
                return self._create_response(False, error=f"文件夹不存在: {folder_path}")
            
            # 使用文件夹路径初始化处理器
            self.file_reader = Get_file(folder_path, is_json=False)
            self.file_reader.init_read_dir()
            
            # 生成会话ID
            session_id = str(uuid.uuid4())
            self.cache[session_id] = {
                'file_reader': self.file_reader,
                'timestamp': datetime.now()
            }
            
            return self._create_response(True, 
                                        data={'session_id': session_id}, 
                                        message="文件夹处理成功")
        except Exception as e:
            return self._handle_error(e, "处理文件夹")
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
    def textrank_search(self):
        """TextRank搜索"""
        try:
            data = request.get_json()
            session_id = data.get('session_id')
            query = data.get('query', '')
            window_size = data.get('window_size', 3)  # 可选参数：共现窗口大小
            damping = data.get('damping', 0.85)  # 可选参数：阻尼系数

            if not session_id or session_id not in self.cache:
                return self._create_response(False, error="无效的会话ID")

            file_processor = self.cache[session_id]['file_reader']
            inverted_index = file_processor.inverted_index
            processed_files_content = file_processor.processed_files_content
            file_word_count = file_processor.file_word_count

            results = self.search_engine.textrank_search(
                query, inverted_index, processed_files_content, 
                file_word_count, window_size, damping
            )

            return self._create_response(True, data=results, message="TextRank检索成功")
        except Exception as e:
            return self._handle_error(e, "TextRank检索")
 
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

    def lsi_search(self):
        """LSI潜在语义索引搜索"""
        try:
            data = request.get_json()
            session_id = data.get('session_id')
            query = data.get('query', '')
            n_components = data.get('n_components', 100)  # 可选参数：LSI维度数
            
            if not session_id or session_id not in self.cache:
                return self._create_response(False, error="无效的会话ID")
    
            file_processor = self.cache[session_id]['file_reader']
            inverted_index = file_processor.inverted_index
            processed_files_content = file_processor.processed_files_content
            file_word_count = file_processor.file_word_count
            all_content = file_processor.content_list
    
            results = self.search_engine.lsi_search(
                query, inverted_index, processed_files_content, 
                file_word_count, all_content, n_components
            )
    
            return self._create_response(True, data=results, message="LSI检索成功")
        except Exception as e:
            return self._handle_error(e, "LSI检索")

    # visualize API
    def generate_wordcloud(self):
        """生成词云图"""
        try:
            data = request.get_json()
            session_id = data.get('session_id')
            article_name = data.get('query', '')  # 修改为接收文章名
            if not session_id or session_id not in self.cache: 
                return self._create_response(False, error="无效的会话ID")

            file_processor = self.cache[session_id]['file_reader']
            processed_files_content = file_processor.processed_files_content
            result = self.visual_processor.generate_wordcloud(article_name, processed_files_content)
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
    def download_file(self, filename):
        """下载文件"""
        try:
            # 使用绝对路径和os.path.join确保路径正确
            base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), 'src', 'pic'))
            
            # 定义所有可能的子目录
            subdirs = ['cloud', 'rank', 'extensions', 'frequency', '']
            
            for subdir in subdirs:
                if subdir:
                    file_path = os.path.join(base_dir, subdir, filename)
                else:
                    file_path = os.path.join(base_dir, filename)
                
                if os.path.exists(file_path):
                    # 创建响应并设置正确的头部
                    response = send_file(
                        os.path.abspath(file_path), 
                        as_attachment=False, 
                        mimetype='image/png'
                    )
                    
                    # 设置CORS头部
                    response.headers['Access-Control-Allow-Origin'] = '*'
                    response.headers['Access-Control-Allow-Methods'] = 'GET'
                    response.headers['Access-Control-Allow-Headers'] = 'Content-Type'
                    response.headers['Cache-Control'] = 'no-cache, no-store, must-revalidate'
                    response.headers['Pragma'] = 'no-cache'
                    response.headers['Expires'] = '0'
                    
                    return response
            
            return self._create_response(False, error=f"文件不存在: {filename}")
            
        except Exception as e:
            self.logger.error(f"下载文件时发生错误: {str(e)}")
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
        # 检查端口是否可用
        def is_port_available(port):
            import socket
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                try:
                    s.bind((host, port))
                    return True
                except OSError:
                    return False
        
        # 如果默认端口不可用，尝试5001-5010范围
        if port == 5000 and not is_port_available(port):
            for new_port in range(5001, 5010):
                if is_port_available(new_port):
                    port = new_port
                    break
            else:
                self.logger.error("端口5000-5010全部被占用，请手动指定端口")
                port = 0  # 让系统自动分配端口
        
        self.logger.info(f"启动服务器: http://{host}:{port}")
        try:
            self.app.run(host=host, port=port, debug=debug)
        except OSError as e:
            if "address already in use" in str(e).lower():
                self.logger.error(f"端口 {port} 已被占用，请尝试其他端口")
                self.logger.info("提示：可执行以下命令查找占用进程：")
                self.logger.info("Windows: netstat -ano | findstr :5000")
                self.logger.info("Linux/Mac: lsof -i :5000")
            else:
                self.logger.error(f"服务器启动失败: {str(e)}")


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
    def _check_dependencies(self):
        """检查必要的依赖包"""
        required_packages = ['networkx', 'scipy']
        missing_packages = []

        for package in required_packages:
            try:
                __import__(package)
            except ImportError:
                missing_packages.append(package)

        if missing_packages:
            print(f"警告: 缺少以下依赖包: {', '.join(missing_packages)}")
            print("请运行: pip install " + " ".join(missing_packages))


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
