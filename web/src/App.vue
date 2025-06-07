<template>

  <div id="app">
    <!-- 顶部导航栏 -->
     <NavBar :health-status="healthStatus" @check-health="checkHealth" />

    <!-- 主容器 -->
        <div class="main-container">
      <!-- 配置上传区域 -->
      <FolderConfig 
        v-if="!sessionId"
        :loading="loading"
        :selected-folder="selectedFolder"
        @folder-submit="handleFolderSubmit"
      />


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
          <SearchPanel 
            :loading="loading"
            :search-results="searchResults"
            @search="handleSearch"
          />
        </div>

        <!-- 可视化功能 -->
        <div v-show="activeTab === 'visualization'" class="tab-content">
          <VisualizationPanel 
            :loading="loading"
            :wordcloud-result="wordcloudResult"
            :keyword-result="keywordResult"
            :file-type-result="fileTypeResult"
            :word-freq-result="wordFreqResult"
            @generate-wordcloud="generateWordcloud"
            @generate-keyword-ranking="generateKeywordRanking"
            @analyze-file-types="analyzeFileTypes"
            @generate-word-frequency-chart="generateWordFrequencyChart"
          />
        </div>


        <!-- 统计信息 -->
        <div v-show="activeTab === 'statistics'" class="tab-content">
          <StatisticsPanel 
            :loading="loading"
            :word-stats="wordStats"
            :file-stats="fileStats"
            @get-word-statistics="getWordStatistics"
            @get-file-statistics="getFileStatistics"
          />
        </div>

      </div>
    </div>

    <!-- 全局加载遮罩 -->
 <!-- 全局加载遮罩 -->
    <LoadingOverlay :show="loading" :message="loadingMessage" />

    <!-- 消息提示 -->
    <MessageToast :message="message" @close="clearMessage" />
  </div>
</template>

<script>
import axios from 'axios';
import LoadingOverlay from './components/LoadingOverlay.vue';
import MessageToast from './components/MessageToast.vue';
import NavBar from './components/NavBar.vue';
import FolderConfig from './components/FolderConfig.vue';
import SearchPanel from './components/SearchPanel.vue';
import VisualizationPanel from './components/VisualizationPanel.vue';
import StatisticsPanel from './components/StatisticsPanel.vue';
export default {
  name: 'App',
  components: {
    LoadingOverlay,
    MessageToast,
    NavBar,
    FolderConfig,
    SearchPanel,
    VisualizationPanel,
    StatisticsPanel
  },


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

      // 标签页配置
      tabs: [
        { id: 'search', name: '搜索', icon: '🔍' },
        { id: 'visualization', name: '可视化', icon: '📊' },
        { id: 'statistics', name: '统计', icon: '📈' }
      ],

      // 搜索相关
      searchResults: null,

      // 可视化相关
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
        getImageUrl(path) {
        if (!path) return '';
        
        const filename = path.split('/').pop().split('\\').pop();
        // 添加时间戳避免缓存问题
        const timestamp = new Date().getTime();
        const url = `http://localhost:5000/download/${filename}?t=${timestamp}`;
        
        this.currentImageUrl = url;
        this.imageLoading = true;
        this.imageError = false;
        
        return url;
      },

      onImageLoad() {
        console.log('图片加载成功');
        this.imageLoading = false;
        this.imageError = false;
      },

      onImageError(event) {
        console.error('图片加载失败:', event.target.src);
        this.imageLoading = false;
        this.imageError = true;
      },

      retryImageLoad() {
        if (this.wordcloudResult?.save_path) {
          this.imageLoading = true;
          this.imageError = false;
          // 强制刷新图片
          const img = this.$el.querySelector('.viz-image');
          if (img) {
            img.src = this.getImageUrl(this.wordcloudResult.save_path);
          }
        }
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
    
    // 执行搜索
    async handleSearch(searchData) {
    const { query, type } = searchData;
      
    this.setLoading(true, '正在搜索...');
      
    try {
      let endpoint;
      switch (type) {
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
        query: query
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
    // 处理文件夹路径提交
    async handleFolderSubmit(data) {
      if (data.error) {
        this.showMessage(data.error, 'error');
        return;
      }

      this.setLoading(true, '正在处理文件夹内容...');

      try {
        const folderPath = data.folderPath;
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



      // 生成词云
      async generateWordcloud(query) {
        this.setLoading(true, '正在生成词云...');
      
        try {
          const response = await axios.post(`${this.apiBaseUrl}/generate_wordcloud`, {
            session_id: this.sessionId,
            query: query
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
async generateKeywordRanking(query) {
  this.setLoading(true, '正在生成关键词排行...');
  
  try {
    const response = await axios.post(`${this.apiBaseUrl}/generate_keyword_ranking`, {
      session_id: this.sessionId,
      query: query
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
async generateWordFrequencyChart(chartType) {
  this.setLoading(true, '正在生成词频图表...');
  
  try {
    const response = await axios.post(`${this.apiBaseUrl}/create_word_frequency_chart`, {
      session_id: this.sessionId,
      chart_type: chartType
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

    
    
    
  }
}
</script>

<style scoped>
    @import './assets/css/main.css';
    
</style>