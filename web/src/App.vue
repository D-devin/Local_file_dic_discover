<template>
  <div id="app">
    <!-- 顶部导航栏 -->
    <nav class="navbar">
      <div class="nav-brand">
        <h1>📚 文档智能检索系统</h1>
      </div>
      <div class="nav-status">
        <div class="status-badge" :class="{ online: healthStatus, offline: !healthStatus }">
          <span class="status-dot"></span>
          {{ healthStatus ? '服务在线' : '服务离线' }}
        </div>
        <button @click="checkHealth" class="btn-refresh">🔄</button>
      </div>
    </nav>

    <!-- 主容器 -->
    <div class="main-container">
      <!-- 配置上传区域 -->
      <div v-if="!sessionId" class="upload-section">
        <div class="upload-card">
          <div class="upload-icon">📁</div>
          <h2>开始使用</h2>
          <p>请选择包含文档的文件夹</p>
          <input 
            type="file" 
            ref="fileInput" 
            @change="handleFolderUpload" 
            webkitdirectory
            multiple
            style="display: none"
          >
          <button @click="$refs.fileInput.click()" class="btn-upload">
            选择文件夹
          </button>
          <div class="folder-info" v-if="selectedFolder">
            <p>已选择文件夹: <strong>{{ selectedFolder }}</strong></p>
            <p>包含文件数: <strong>{{ fileCount }}</strong></p>
          </div>
          <div class="folder-instructions">
            <details>
              <summary>使用说明</summary>
              <ul>
                <li>选择包含文档的文件夹（支持所有文本格式）</li>
                <li>系统会自动处理文件夹中的所有文档</li>
                <li>处理完成后即可使用搜索和分析功能</li>
              </ul>
            </details>
          </div>
        </div>
      </div>

      <!-- 主功能区域 -->
      <div v-if="sessionId" class="workspace">
        <!-- 会话信息栏 -->
        <div class="session-bar">
          <span class="session-info">
            🔗 会话ID: {{ sessionId.substring(0, 8) }}...
          </span>
          <button @click="clearSession" class="btn-clear">清理会话</button>
        </div>

        <!-- 功能标签页 -->
        <div class="tabs">
          <button 
            v-for="tab in tabs" 
            :key="tab.id"
            @click="activeTab = tab.id"
            :class="['tab-btn', { active: activeTab === tab.id }]"
          >
            {{ tab.icon }} {{ tab.name }}
          </button>
        </div>

        <!-- 搜索功能 -->
        <div v-show="activeTab === 'search'" class="tab-content">
          <div class="search-panel">
            <div class="search-header">
              <h3>🔍 智能搜索</h3>
              <select v-model="searchType" class="search-type-select">
                <option value="tf">TF 词频搜索</option>
                <option value="tfidf">TF-IDF 搜索</option>
                <option value="word_vector">词向量搜索</option>
              </select>
            </div>
            
            <div class="search-input-group">
              <input 
                v-model="searchQuery" 
                type="text" 
                placeholder="输入搜索关键词..."
                class="search-input"
                @keyup.enter="performSearch"
              >
              <button @click="performSearch" :disabled="!searchQuery || loading" class="btn-search">
                {{ loading ? '搜索中...' : '搜索' }}
              </button>
            </div>

            <!-- 搜索结果 -->
            <div v-if="searchResults" class="search-results">
              <div class="results-header">
                <h4>搜索结果 ({{ searchResults.total_results || 0 }} 条)</h4>
              </div>
              
              <div v-if="searchResults.results && searchResults.results.length > 0" class="results-list">
                <div v-for="(result, index) in searchResults.results" :key="index" class="result-card">
                  <div class="result-header">
                    <span class="file-name">📄 {{ getFileName(result.file_name) }}</span>
                    <span class="result-score">
                      {{ formatScore(result, searchType) }}
                    </span>
                  </div>
                  <div class="result-preview">{{ result.content_preview }}</div>
                  <div v-if="result.matched_words" class="matched-words">
                    <span class="tag" v-for="word in result.matched_words" :key="word">{{ word }}</span>
                  </div>
                  <div v-if="result.positions" class="positions-info">
                    位置: {{ result.positions.join(', ') }}
                  </div>
                </div>
              </div>
              
              <div v-else class="no-results">
                <div class="no-results-icon">🔍</div>
                <p>未找到相关结果</p>
              </div>
            </div>
          </div>
        </div>

        <!-- 可视化功能 -->
        <div v-show="activeTab === 'visualization'" class="tab-content">
          <div class="viz-grid">
            <!-- 词云图 -->
            <div class="viz-card">
              <div class="viz-header">
                <h4>☁️ 词云图</h4>
              </div>
              <div class="viz-controls">
                <input 
                  v-model="wordcloudQuery" 
                  type="text" 
                  placeholder="输入关键词生成词云"
                  class="viz-input"
                >
                <button @click="generateWordcloud" :disabled="!wordcloudQuery || loading" class="btn-viz">
                  生成
                </button>
              </div>
              <div v-if="wordcloudResult?.save_path" class="viz-display">
                <img :src="getImageUrl(wordcloudResult.save_path)" alt="词云图" class="viz-image">
                <div class="viz-stats">
                  <span>相关文档: {{ wordcloudResult.relevant_docs }}</span>
                  <span>总词汇: {{ wordcloudResult.total_words }}</span>
                </div>
              </div>
              <div v-else-if="wordcloudResult" class="viz-message">
                {{ wordcloudResult.message }}
              </div>
            </div>

            <!-- 关键词排行 -->
            <div class="viz-card">
              <div class="viz-header">
                <h4>📊 关键词排行</h4>
              </div>
              <div class="viz-controls">
                <input 
                  v-model="keywordQuery" 
                  type="text" 
                  placeholder="输入关键词生成排行"
                  class="viz-input"
                >
                <button @click="generateKeywordRanking" :disabled="!keywordQuery || loading" class="btn-viz">
                  生成
                </button>
              </div>
              <div v-if="keywordResult?.save_path" class="viz-display">
                <img :src="getImageUrl(keywordResult.save_path)" alt="关键词排行" class="viz-image">
                <div class="viz-stats">
                  <span>分析文件: {{ keywordResult.total_analyzed_files }}</span>
                </div>
              </div>
              <div v-else-if="keywordResult" class="viz-message">
                {{ keywordResult.message }}
              </div>
            </div>

            <!-- 文件类型分析 -->
            <div class="viz-card">
              <div class="viz-header">
                <h4>📈 文件类型分析</h4>
              </div>
              <div class="viz-controls">
                <button @click="analyzeFileTypes" :disabled="loading" class="btn-viz">
                  分析文件类型
                </button>
              </div>
              <div v-if="fileTypeResult?.save_path" class="viz-display">
                <img :src="getImageUrl(fileTypeResult.save_path)" alt="文件类型分析" class="viz-image">
                <div class="viz-stats">
                  <span>总文件: {{ fileTypeResult.total_files }}</span>
                  <span>文件类型: {{ fileTypeResult.unique_extensions }}</span>
                </div>
              </div>
              <div v-else-if="fileTypeResult" class="viz-message">
                {{ fileTypeResult.message }}
              </div>
            </div>

            <!-- 词频统计 -->
            <div class="viz-card">
              <div class="viz-header">
                <h4>📉 词频统计</h4>
              </div>
              <div class="viz-controls">
                <select v-model="chartType" class="chart-select">
                  <option value="bar">柱状图</option>
                  <option value="line">折线图</option>
                  <option value="pie">饼图</option>
                </select>
                <button @click="generateWordFrequencyChart" :disabled="loading" class="btn-viz">
                  生成图表
                </button>
              </div>
              <div v-if="wordFreqResult?.save_path" class="viz-display">
                <img :src="getImageUrl(wordFreqResult.save_path)" alt="词频统计" class="viz-image">
                <div class="viz-stats">
                  <span>分析词汇: {{ wordFreqResult.analyzed_words }}</span>
                </div>
              </div>
              <div v-else-if="wordFreqResult" class="viz-message">
                {{ wordFreqResult.message }}
              </div>
            </div>
          </div>
        </div>

        <!-- 统计信息 -->
        <div v-show="activeTab === 'statistics'" class="tab-content">
          <div class="stats-grid">
            <div class="stat-card">
              <div class="stat-icon">📝</div>
              <h4>词汇统计</h4>
              <div v-if="wordStats" class="stat-data">
                <div class="stat-item">
                  <span class="stat-label">总词汇数</span>
                  <span class="stat-value">{{ wordStats.total_words }}</span>
                </div>
                <div class="stat-item">
                  <span class="stat-label">唯一词汇</span>
                  <span class="stat-value">{{ wordStats.unique_words }}</span>
                </div>
                <div class="stat-item">
                  <span class="stat-label">高频词汇</span>
                  <ul class="top-words">
                    <li v-for="(count, word, index) in wordStats.top_words" :key="index">
                      {{ word }}: {{ count }}
                    </li>
                  </ul>
                </div>
              </div>
              <button @click="getWordStatistics" :disabled="loading" class="btn-stat">
                获取统计
              </button>
            </div>

            <div class="stat-card">
              <div class="stat-icon">📁</div>
              <h4>文件统计</h4>
              <div v-if="fileStats" class="stat-data">
                <div class="stat-item">
                  <span class="stat-label">文件总数</span>
                  <span class="stat-value">{{ fileStats.file_count }}</span>
                </div>
                <div class="stat-item">
                  <span class="stat-label">文件类型</span>
                  <ul class="file-types">
                    <li v-for="(count, type) in fileStats.file_types" :key="type">
                      {{ type }}: {{ count }}
                    </li>
                  </ul>
                </div>
              </div>
              <button @click="getFileStatistics" :disabled="loading" class="btn-stat">
                获取统计
              </button>
            </div>
          </div>
        </div>
      </div>
    </div>

    <!-- 全局加载遮罩 -->
    <div v-if="loading" class="loading-overlay">
      <div class="loading-spinner"></div>
      <p>{{ loadingMessage }}</p>
    </div>

    <!-- 消息提示 -->
    <transition name="message">
      <div v-if="message" :class="['message-toast', message.type]">
        <span>{{ message.text }}</span>
        <button @click="clearMessage" class="message-close">×</button>
      </div>
    </transition>
  </div>
</template>

<script>
import axios from 'axios';

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
      fileCount: 0,

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
    
    // 文件夹上传处理
    async handleFolderUpload(event) {
      const files = event.target.files;
      if (!files || files.length === 0) return;
      
      this.setLoading(true, '正在上传和处理文件夹内容...');
      
      try {
        // 获取文件夹名称（从第一个文件的相对路径中提取）
        const folderPath = files[0].webkitRelativePath.split('/')[0];
        this.selectedFolder = folderPath;
        this.fileCount = files.length;
        
        // 创建FormData对象并添加所有文件
        const formData = new FormData();
        for (let i = 0; i < files.length; i++) {
          formData.append('files', files[i], files[i].webkitRelativePath);
        }
        
        // 发送到后端API
        const response = await axios.post(`${this.apiBaseUrl}/upload_folder`, formData, {
          headers: {
            'Content-Type': 'multipart/form-data'
          }
        });
        
        if (response.data.success) {
          this.sessionId = response.data.data.session_id;
          this.showMessage('文件夹上传成功，会话已创建');
        } else {
          this.showMessage(`上传失败: ${response.data.error}`, 'error');
        }
      } catch (error) {
        this.showMessage('文件夹上传失败: ' + error.message, 'error');
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
          this.fileCount = 0;
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
</script>

<style>
/* 基础样式 */
* {
  margin: 0;
  padding: 0;
  box-sizing: border-box;
  font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
}

body {
  background-color: #f5f7fa;
  color: #333;
  line-height: 1.6;
}

#app {
  display: flex;
  flex-direction: column;
  min-height: 100vh;
}

/* 导航栏样式 */
.navbar {
  display: flex;
  justify-content: space-between;
  align-items: center;
  padding: 1rem 2rem;
  background: linear-gradient(135deg, #1e5799, #207cca);
  color: white;
  box-shadow: 0 2px 10px rgba(0, 0, 0, 0.1);
}

.nav-brand h1 {
  font-size: 1.5rem;
  font-weight: 600;
}

.nav-status {
  display: flex;
  align-items: center;
  gap: 10px;
}

.status-badge {
  display: flex;
  align-items: center;
  padding: 5px 12px;
  border-radius: 20px;
  font-size: 0.9rem;
  font-weight: 500;
  background-color: rgba(255, 255, 255, 0.2);
}

.status-dot {
  display: inline-block;
  width: 8px;
  height: 8px;
  border-radius: 50%;
  margin-right: 6px;
}

.status-badge.online .status-dot {
  background-color: #4caf50;
}

.status-badge.offline .status-dot {
  background-color: #f44336;
}

.btn-refresh {
  background: none;
  border: none;
  color: white;
  font-size: 1.2rem;
  cursor: pointer;
  padding: 5px;
  border-radius: 50%;
  transition: background 0.3s;
}

.btn-refresh:hover {
  background: rgba(255, 255, 255, 0.2);
}

/* 主容器样式 */
.main-container {
  flex: 1;
  padding: 2rem;
  max-width: 1400px;
  margin: 0 auto;
  width: 100%;
}

.upload-section {
  display: flex;
  justify-content: center;
  align-items: center;
  height: calc(100vh - 150px);
}

.upload-card {
  background: white;
  border-radius: 12px;
  box-shadow: 0 5px 15px rgba(0, 0, 0, 0.1);
  padding: 2.5rem;
  text-align: center;
  max-width: 500px;
  width: 100%;
}

.upload-icon {
  font-size: 4rem;
  margin-bottom: 1.5rem;
}

.upload-card h2 {
  margin-bottom: 1rem;
  color: #2c3e50;
}

.upload-card p {
  margin-bottom: 1.5rem;
  color: #7f8c8d;
}

.btn-upload {
  background: #3498db;
  color: white;
  border: none;
  padding: 12px 25px;
  border-radius: 6px;
  font-size: 1rem;
  font-weight: 600;
  cursor: pointer;
  transition: background 0.3s;
  margin-bottom: 1.5rem;
}

.btn-upload:hover {
  background: #2980b9;
}

.folder-info {
  background: #f8f9fa;
  border-radius: 8px;
  padding: 1rem;
  text-align: left;
  margin: 1rem 0;
  font-size: 0.95rem;
}

.folder-info p {
  margin-bottom: 0.5rem;
}

.folder-instructions {
  background: #f8f9fa;
  border-radius: 8px;
  padding: 1rem;
  text-align: left;
  margin-top: 1rem;
  font-size: 0.9rem;
}

.folder-instructions summary {
  font-weight: 600;
  cursor: pointer;
  margin-bottom: 0.5rem;
}

.folder-instructions ul {
  padding-left: 1.5rem;
  margin-top: 0.5rem;
}

.folder-instructions li {
  margin-bottom: 0.5rem;
}

/* 工作区样式 */
.workspace {
  background: white;
  border-radius: 12px;
  box-shadow: 0 5px 15px rgba(0, 0, 0, 0.05);
  overflow: hidden;
}

.session-bar {
  display: flex;
  justify-content: space-between;
  align-items: center;
  padding: 1rem 1.5rem;
  background: #f8f9fa;
  border-bottom: 1px solid #eaeaea;
}

.session-info {
  font-size: 0.95rem;
  color: #7f8c8d;
}

.btn-clear {
  background: #e74c3c;
  color: white;
  border: none;
  padding: 6px 12px;
  border-radius: 4px;
  font-size: 0.85rem;
  cursor: pointer;
  transition: background 0.3s;
}

.btn-clear:hover {
  background: #c0392b;
}

/* 标签页样式 */
.tabs {
  display: flex;
  border-bottom: 1px solid #eaeaea;
  padding: 0 1.5rem;
}

.tab-btn {
  padding: 12px 20px;
  background: none;
  border: none;
  cursor: pointer;
  font-size: 1rem;
  font-weight: 500;
  color: #7f8c8d;
  position: relative;
  transition: all 0.3s;
}

.tab-btn:hover {
  color: #3498db;
}

.tab-btn.active {
  color: #3498db;
  font-weight: 600;
}

.tab-btn.active::after {
  content: '';
  position: absolute;
  bottom: -1px;
  left: 0;
  width: 100%;
  height: 3px;
  background: #3498db;
  border-radius: 3px 3px 0 0;
}

/* 搜索面板样式 */
.tab-content {
  padding: 1.5rem;
}

.search-panel {
  background: #f8fafc;
  border-radius: 8px;
  padding: 1.5rem;
}

.search-header {
  display: flex;
  justify-content: space-between;
  align-items: center;
  margin-bottom: 1.5rem;
}

.search-header h3 {
  font-size: 1.3rem;
  color: #2c3e50;
}

.search-type-select {
  padding: 8px 12px;
  border-radius: 6px;
  border: 1px solid #ddd;
  background: white;
  font-size: 0.95rem;
}

.search-input-group {
  display: flex;
  gap: 10px;
  margin-bottom: 1.5rem;
}

.search-input {
  flex: 1;
  padding: 12px 15px;
  border: 1px solid #ddd;
  border-radius: 6px;
  font-size: 1rem;
  transition: border 0.3s;
}

.search-input:focus {
  border-color: #3498db;
  outline: none;
  box-shadow: 0 0 0 3px rgba(52, 152, 219, 0.2);
}

.btn-search {
  background: #3498db;
  color: white;
  border: none;
  padding: 0 25px;
  border-radius: 6px;
  font-size: 1rem;
  font-weight: 600;
  cursor: pointer;
  transition: background 0.3s;
}

.btn-search:disabled {
  background: #bdc3c7;
  cursor: not-allowed;
}

.btn-search:hover:not(:disabled) {
  background: #2980b9;
}

/* 搜索结果样式 */
.results-header {
  margin-bottom: 1.2rem;
}

.results-header h4 {
  font-size: 1.1rem;
  color: #2c3e50;
}

.results-list {
  display: grid;
  gap: 1rem;
}

.result-card {
  background: white;
  border-radius: 8px;
  box-shadow: 0 2px 8px rgba(0, 0, 0, 0.05);
  padding: 1.2rem;
  transition: transform 0.2s, box-shadow 0.2s;
}

.result-card:hover {
  transform: translateY(-2px);
  box-shadow: 0 5px 15px rgba(0, 0, 0, 0.08);
}

.result-header {
  display: flex;
  justify-content: space-between;
  align-items: center;
  margin-bottom: 0.8rem;
}

.file-name {
  font-weight: 600;
  color: #2c3e50;
}

.result-score {
  background: #e1f5fe;
  color: #0288d1;
  padding: 4px 8px;
  border-radius: 4px;
  font-size: 0.85rem;
  font-weight: 500;
}

.result-preview {
  color: #555;
  margin-bottom: 0.8rem;
  line-height: 1.5;
}

.matched-words {
  display: flex;
  flex-wrap: wrap;
  gap: 6px;
  margin-bottom: 0.5rem;
}

.tag {
  background: #e8f4fc;
  color: #3498db;
  padding: 3px 8px;
  border-radius: 4px;
  font-size: 0.85rem;
}

.positions-info {
  font-size: 0.85rem;
  color: #7f8c8d;
}

.no-results {
  text-align: center;
  padding: 2rem;
}

.no-results-icon {
  font-size: 3rem;
  margin-bottom: 1rem;
  color: #bdc3c7;
}

.no-results p {
  color: #7f8c8d;
}

/* 可视化网格样式 */
.viz-grid {
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(350px, 1fr));
  gap: 1.5rem;
}

.viz-card {
  background: white;
  border-radius: 8px;
  box-shadow: 0 2px 10px rgba(0, 0, 0, 0.05);
  padding: 1.5rem;
  display: flex;
  flex-direction: column;
}

.viz-header {
  margin-bottom: 1.2rem;
}

.viz-header h4 {
  font-size: 1.2rem;
  color: #2c3e50;
}

.viz-controls {
  display: flex;
  gap: 10px;
  margin-bottom: 1.2rem;
}

.viz-input {
  flex: 1;
  padding: 10px 12px;
  border: 1px solid #ddd;
  border-radius: 6px;
  font-size: 0.95rem;
}

.btn-viz {
  background: #3498db;
  color: white;
  border: none;
  padding: 0 20px;
  border-radius: 6px;
  font-size: 0.95rem;
  font-weight: 500;
  cursor: pointer;
  transition: background 0.3s;
}

.btn-viz:hover {
  background: #2980b9;
}

.btn-viz:disabled {
  background: #bdc3c7;
  cursor: not-allowed;
}

.chart-select {
  padding: 8px 12px;
  border-radius: 6px;
  border: 1px solid #ddd;
  background: white;
  font-size: 0.95rem;
}

.viz-display {
  flex: 1;
  display: flex;
  flex-direction: column;
}

.viz-image {
  width: 100%;
  height: auto;
  border-radius: 6px;
  margin-bottom: 0.8rem;
}

.viz-stats {
  display: flex;
  justify-content: space-between;
  font-size: 0.85rem;
  color: #7f8c8d;
}

.viz-message {
  padding: 1rem;
  background: #f8f9fa;
  border-radius: 6px;
  color: #7f8c8d;
  font-size: 0.95rem;
  text-align: center;
}

/* 统计信息样式 */
.stats-grid {
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(350px, 1fr));
  gap: 1.5rem;
}

.stat-card {
  background: white;
  border-radius: 8px;
  box-shadow: 0 2px 10px rgba(0, 0, 0, 0.05);
  padding: 1.5rem;
  text-align: center;
}

.stat-icon {
  font-size: 3rem;
  margin-bottom: 1rem;
  color: #3498db;
}

.stat-card h4 {
  font-size: 1.2rem;
  margin-bottom: 1.2rem;
  color: #2c3e50;
}

.stat-data {
  margin-bottom: 1.5rem;
}

.stat-item {
  display: flex;
  justify-content: space-between;
  margin-bottom: 0.8rem;
  padding-bottom: 0.8rem;
  border-bottom: 1px solid #eee;
}

.stat-label {
  font-weight: 500;
  color: #7f8c8d;
}

.stat-value {
  font-weight: 600;
  color: #2c3e50;
}

.top-words, .file-types {
  text-align: left;
  margin-top: 0.5rem;
  padding-left: 1.5rem;
}

.top-words li, .file-types li {
  margin-bottom: 0.3rem;
}

.btn-stat {
  background: #3498db;
  color: white;
  border: none;
  padding: 8px 20px;
  border-radius: 6px;
  font-size: 0.95rem;
  cursor: pointer;
  transition: background 0.3s;
}

.btn-stat:hover {
  background: #2980b9;
}

.btn-stat:disabled {
  background: #bdc3c7;
  cursor: not-allowed;
}

/* 加载遮罩 */
.loading-overlay {
  position: fixed;
  top: 0;
  left: 0;
  right: 0;
  bottom: 0;
  background: rgba(255, 255, 255, 0.8);
  display: flex;
  flex-direction: column;
  justify-content: center;
  align-items: center;
  z-index: 1000;
}

.loading-spinner {
  width: 50px;
  height: 50px;
  border: 5px solid #f3f3f3;
  border-top: 5px solid #3498db;
  border-radius: 50%;
  animation: spin 1s linear infinite;
  margin-bottom: 1rem;
}

@keyframes spin {
  0% { transform: rotate(0deg); }
  100% { transform: rotate(360deg); }
}

/* 消息提示 */
.message-toast {
  position: fixed;
  bottom: 20px;
  right: 20px;
  padding: 15px 20px;
  border-radius: 8px;
  color: white;
  display: flex;
  align-items: center;
  box-shadow: 0 4px 12px rgba(0, 0, 0, 0.15);
  z-index: 1000;
  animation: fadeIn 0.3s;
}

.message-toast.info {
  background: #3498db;
}

.message-toast.error {
  background: #e74c3c;
}

.message-toast.success {
  background: #2ecc71;
}

@keyframes fadeIn {
  from { opacity: 0; transform: translateY(20px); }
  to { opacity: 1; transform: translateY(0); }
}

.message-close {
  background: none;
  border: none;
  color: white;
  font-size: 1.2rem;
  margin-left: 15px;
  cursor: pointer;
}

/* 响应式调整 */
@media (max-width: 768px) {
  .navbar {
    flex-direction: column;
    padding: 1rem;
  }
  
  .nav-status {
    margin-top: 1rem;
  }
  
  .tabs {
    overflow-x: auto;
    padding: 0 1rem;
  }
  
  .tab-btn {
    padding: 10px 15px;
    font-size: 0.9rem;
  }
  
  .main-container {
    padding: 1rem;
  }
  
  .viz-grid, .stats-grid {
    grid-template-columns: 1fr;
  }
}
</style>