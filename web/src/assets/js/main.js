export default {
  name: 'App',
  data() {
    return {
      // 基础状态
      healthStatus: false,
      sessionId: null,
      loading: false,
      loadingMessage: '',
      message: null,
      activeTab: 'search',
      selectedFolder: null,
      folderPathInput: '', // 新增：文件夹路径输入

      // 标签页配置
      tabs: [
        { id: 'search', name: '搜索', icon: '🔍' },
        { id: 'visualization', name: '可视化', icon: '📊' },
        { id: 'statistics', name: '统计', icon: '📈' }
      ],

      // 搜索相关
      searchQuery: '',
      searchType: 'tf',
      searchResults: null,

      // 可视化相关
      wordcloudQuery: '',
      keywordQuery: '',
      chartType: 'bar',
      wordcloudResult: null,
      keywordResult: null,
      fileTypeResult: null,
      wordFreqResult: null,

      // 统计信息
      wordStats: null,
      fileStats: null,

      // API配置
      apiBaseUrl: 'http://localhost:5000'
    }
  },

  mounted() {
    this.checkHealth();
  },

  methods: {
    // 显示消息
    showMessage(text, type = 'info') {
      this.message = { text, type };
      setTimeout(() => {
        this.clearMessage();
      }, 5000);
    },
    
    // 清除消息
    clearMessage() {
      this.message = null;
    },
    
    // 设置加载状态
    setLoading(isLoading, message = '') {
      this.loading = isLoading;
      this.loadingMessage = message;
    },
    
    // 健康检查
    async checkHealth() {
      try {
        const response = await axios.get(`${this.apiBaseUrl}/health`);
        this.healthStatus = response.data.success;
      } catch (error) {
        this.healthStatus = false;
        this.showMessage('服务连接失败', 'error');
      }
    },
    
    // 处理文件夹路径
    async handleFolderPath() {
      if (!this.folderPathInput) {
        this.showMessage('请输入文件夹路径', 'error');
        return;
      }
      
      this.setLoading(true, '正在处理文件夹内容...');
      
      try {
        const folderPath = this.folderPathInput;
        this.selectedFolder = folderPath;
        // 发送文件夹路径到后端
        const response = await axios.post(`${this.apiBaseUrl}/process_folder`, {
          folder_path: folderPath
        });
        
        if (response.data.success) {
          this.sessionId = response.data.data.session_id;
          this.showMessage('文件夹处理成功，会话已创建');
        } else {
          this.showMessage(`处理失败: ${response.data.error}`, 'error');
        }
      } catch (error) {
        this.showMessage('文件夹处理失败: ' + error.message, 'error');
      } finally {
        this.setLoading(false);
      }
    },
    
    // 执行搜索
    async performSearch() {
      if (!this.searchQuery) return;
      
      this.setLoading(true, '正在搜索...');
      
      try {
        let endpoint;
        switch (this.searchType) {
          case 'tf':
            endpoint = '/tf_search';
            break;
          case 'tfidf':
            endpoint = '/tfidf_search';
            break;
          case 'word_vector':
            endpoint = '/word_vector_search';
            break;
          default:
            endpoint = '/tf_search';
        }
        
        const response = await axios.post(`${this.apiBaseUrl}${endpoint}`, {
          session_id: this.sessionId,
          query: this.searchQuery
        });
        
        if (response.data.success) {
          this.searchResults = response.data.data;
          this.showMessage(`找到 ${this.searchResults.total_results} 个结果`);
        } else {
          this.showMessage(`搜索失败: ${response.data.error}`, 'error');
          this.searchResults = null;
        }
      } catch (error) {
        this.showMessage('搜索请求失败: ' + error.message, 'error');
        this.searchResults = null;
      } finally {
        this.setLoading(false);
      }
    },
    
    // 生成词云
    async generateWordcloud() {
      if (!this.wordcloudQuery) return;
      
      this.setLoading(true, '正在生成词云...');
      
      try {
        const response = await axios.post(`${this.apiBaseUrl}/generate_wordcloud`, {
          session_id: this.sessionId,
          query: this.wordcloudQuery
        });
        
        if (response.data.success) {
          this.wordcloudResult = response.data.data;
          this.showMessage('词云图生成成功');
        } else {
          this.showMessage(`生成失败: ${response.data.error}`, 'error');
          this.wordcloudResult = null;
        }
      } catch (error) {
        this.showMessage('词云生成失败: ' + error.message, 'error');
        this.wordcloudResult = null;
      } finally {
        this.setLoading(false);
      }
    },
    
    // 生成关键词排行
    async generateKeywordRanking() {
      if (!this.keywordQuery) return;
      
      this.setLoading(true, '正在生成关键词排行...');
      
      try {
        const response = await axios.post(`${this.apiBaseUrl}/generate_keyword_ranking`, {
          session_id: this.sessionId,
          query: this.keywordQuery
        });
        
        if (response.data.success) {
          this.keywordResult = response.data.data;
          this.showMessage('关键词排行生成成功');
        } else {
          this.showMessage(`生成失败: ${response.data.error}`, 'error');
          this.keywordResult = null;
        }
      } catch (error) {
        this.showMessage('关键词排行生成失败: ' + error.message, 'error');
        this.keywordResult = null;
      } finally {
        this.setLoading(false);
      }
    },
    
    // 分析文件类型
    async analyzeFileTypes() {
      this.setLoading(true, '正在分析文件类型...');
      
      try {
        const response = await axios.post(`${this.apiBaseUrl}/analyze_file_extensions`, {
          session_id: this.sessionId
        });
        
        if (response.data.success) {
          this.fileTypeResult = response.data.data;
          this.showMessage('文件类型分析完成');
        } else {
          this.showMessage(`分析失败: ${response.data.error}`, 'error');
          this.fileTypeResult = null;
        }
      } catch (error) {
        this.showMessage('文件类型分析失败: ' + error.message, 'error');
        this.fileTypeResult = null;
      } finally {
        this.setLoading(false);
      }
    },
    
    // 生成词频图表
    async generateWordFrequencyChart() {
      this.setLoading(true, '正在生成词频图表...');
      
      try {
        const response = await axios.post(`${this.apiBaseUrl}/create_word_frequency_chart`, {
          session_id: this.sessionId,
          chart_type: this.chartType
        });
        
        if (response.data.success) {
          this.wordFreqResult = response.data.data;
          this.showMessage('词频图表生成成功');
        } else {
          this.showMessage(`生成失败: ${response.data.error}`, 'error');
          this.wordFreqResult = null;
        }
      } catch (error) {
        this.showMessage('词频图表生成失败: ' + error.message, 'error');
        this.wordFreqResult = null;
      } finally {
        this.setLoading(false);
      }
    },
    
    // 获取词汇统计
    async getWordStatistics() {
      this.setLoading(true, '正在获取词汇统计...');
      
      try {
        const response = await axios.post(`${this.apiBaseUrl}/get_word_statistics`, {
          session_id: this.sessionId
        });
        
        if (response.data.success) {
          this.wordStats = response.data.data;
          this.showMessage('词汇统计获取成功');
        } else {
          this.showMessage(`获取失败: ${response.data.error}`, 'error');
          this.wordStats = null;
        }
      } catch (error) {
        this.showMessage('词汇统计获取失败: ' + error.message, 'error');
        this.wordStats = null;
      } finally {
        this.setLoading(false);
      }
    },
    
    // 获取文件统计
    async getFileStatistics() {
      this.setLoading(true, '正在获取文件统计...');
      
      try {
        const response = await axios.post(`${this.apiBaseUrl}/get_file_statistics`, {
          session_id: this.sessionId
        });
        
        if (response.data.success) {
          this.fileStats = response.data.data;
          this.showMessage('文件统计获取成功');
        } else {
          this.showMessage(`获取失败: ${response.data.error}`, 'error');
          this.fileStats = null;
        }
      } catch (error) {
        this.showMessage('文件统计获取失败: ' + error.message, 'error');
        this.fileStats = null;
      } finally {
        this.setLoading(false);
      }
    },
    
    // 清理会话
    async clearSession() {
      this.setLoading(true, '正在清理会话...');
      
      try {
        const response = await axios.post(`${this.apiBaseUrl}/clear_session`, {
          session_id: this.sessionId
        });
        
        if (response.data.success) {
          this.sessionId = null;
          this.searchResults = null;
          this.wordcloudResult = null;
          this.keywordResult = null;
          this.fileTypeResult = null;
          this.wordFreqResult = null;
          this.wordStats = null;
          this.fileStats = null;
          this.selectedFolder = null;
          this.folderPathInput = '';
          this.showMessage('会话已清理');
        } else {
          this.showMessage(`清理失败: ${response.data.error}`, 'error');
        }
      } catch (error) {
        this.showMessage('会话清理失败: ' + error.message, 'error');
      } finally {
        this.setLoading(false);
      }
    },
    
    // 格式化分数显示
    formatScore(result, searchType) {
      switch (searchType) {
        case 'tf':
          return `词频: ${result.word_frequency} (TF: ${result.tf_score})`;
        case 'tfidf':
          return `TF-IDF: ${result.tfidf_score}`;
        case 'word_vector':
          return `向量相似度: ${result.vector_similarity}`;
        default:
          return '';
      }
    },
    
    // 从路径中提取文件名
    getFileName(fullPath) {
      return fullPath.split('/').pop() || fullPath;
    },
    
    // 获取图片URL
    getImageUrl(path) {
      if (!path) return '';
      const filename = path.split('/').pop();
      return `${this.apiBaseUrl}/download/${filename}`;
    }
  }
}