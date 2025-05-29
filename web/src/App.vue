<template>
  <div id="app">
    <header class="app-header">
      <h1>文档搜索与分析系统</h1>
      <div class="health-status">
        <span :class="['status-indicator', healthStatus ? 'online' : 'offline']"></span>
        {{ healthStatus ? '服务正常' : '服务离线' }}
        <button @click="checkHealth" class="refresh-btn">刷新</button>
      </div>
    </header>

    <nav class="app-nav">
      <button 
        v-for="tab in tabs" 
        :key="tab.key"
        :class="['nav-btn', { active: activeTab === tab.key }]"
        @click="activeTab = tab.key"
      >
        {{ tab.label }}
      </button>
    </nav>

    <main class="app-main">
      <!-- 文件管理 -->
      <div v-show="activeTab === 'files'" class="tab-content">
        <FileUpload @file-processed="onFileProcessed" />
      </div>

      <!-- 搜索功能 -->
      <div v-show="activeTab === 'search'" class="tab-content">
        <SearchComponent />
      </div>

      <!-- 数据可视化 -->
      <div v-show="activeTab === 'visualization'" class="tab-content">
        <Visualization />
      </div>

      <!-- 统计信息 -->
      <div v-show="activeTab === 'statistics'" class="tab-content">
        <Statistics />
      </div>
    </main>

    <!-- 全局加载提示 -->
    <div v-if="globalLoading" class="global-loading">
      <div class="loading-spinner"></div>
      <p>{{ loadingMessage }}</p>
    </div>

    <!-- 消息提示 -->
    <div v-if="message" :class="['message', message.type]">
      {{ message.text }}
      <button @click="clearMessage">×</button>
    </div>
  </div>
</template>

<script>
import FileUpload from './components/FileUpload.vue'
import SearchComponent from './components/SearchComponent.vue'
import Visualization from './components/Visualization.vue'
import DataStatistics from './components/DataStatistics.vue'
import { toolsAPI } from './api/toolsApi'

export default {
  name: 'App',
  components: {
    FileUpload,
    SearchComponent,
    Visualization,
    DataStatistics
  },
  data() {
    return {
      activeTab: 'files',
      healthStatus: false,
      globalLoading: false,
      loadingMessage: '',
      message: null,
      tabs: [
        { key: 'files', label: '文件管理' },
        { key: 'search', label: '搜索' },
        { key: 'visualization', label: '可视化' },
        { key: 'statistics', label: '统计' }
      ]
    }
  },
  methods: {
    async checkHealth() {
      try {
        await toolsAPI.healthCheck()
        this.healthStatus = true
        this.showMessage('服务连接正常', 'success')
      } catch (error) {
        this.healthStatus = false
        this.showMessage('服务连接失败', 'error')
      }
    },

    showMessage(text, type = 'info') {
      this.message = { text, type }
      setTimeout(() => {
        this.clearMessage()
      }, 3000)
    },

    clearMessage() {
      this.message = null
    },

    onFileProcessed() {
      this.showMessage('文件处理完成', 'success')
    },

    setGlobalLoading(loading, message = '') {
      this.globalLoading = loading
      this.loadingMessage = message
    }
  },

  async mounted() {
    // 应用启动时检查服务健康状态
    await this.checkHealth()
    
    // 定期检查服务状态
    setInterval(() => {
      this.checkHealth()
    }, 30000) // 每30秒检查一次
  },

  // 提供全局方法给子组件使用
  provide() {
    return {
      showMessage: this.showMessage,
      setGlobalLoading: this.setGlobalLoading
    }
  }
}
</script>

<style>
/* 全局样式 */
* {
  margin: 0;
  padding: 0;
  box-sizing: border-box;
}

#app {
  font-family: 'Avenir', Helvetica, Arial, sans-serif;
  color: #2c3e50;
  min-height: 100vh;
  background-color: #f5f5f5;
}

.app-header {
  background: #fff;
  padding: 1rem 2rem;
  box-shadow: 0 2px 4px rgba(0,0,0,0.1);
  display: flex;
  justify-content: space-between;
  align-items: center;
}

.health-status {
  display: flex;
  align-items: center;
  gap: 0.5rem;
}

.status-indicator {
  width: 10px;
  height: 10px;
  border-radius: 50%;
}

.status-indicator.online {
  background-color: #4CAF50;
}

.status-indicator.offline {
  background-color: #f44336;
}

.app-nav {
  background: #fff;
  padding: 0 2rem;
  border-bottom: 1px solid #e0e0e0;
  display: flex;
  gap: 1rem;
}

.nav-btn {
  padding: 1rem 1.5rem;
  border: none;
  background: none;
  cursor: pointer;
  border-bottom: 2px solid transparent;
  transition: all 0.3s;
}

.nav-btn:hover {
  background-color: #f5f5f5;
}

.nav-btn.active {
  border-bottom-color: #409EFF;
  color: #409EFF;
}

.app-main {
  padding: 2rem;
  max-width: 1200px;
  margin: 0 auto;
}

.tab-content {
  background: #fff;
  padding: 2rem;
  border-radius: 8px;
  box-shadow: 0 2px 8px rgba(0,0,0,0.1);
}

.global-loading {
  position: fixed;
  top: 0;
  left: 0;
  right: 0;
  bottom: 0;
  background: rgba(0,0,0,0.5);
  display: flex;
  flex-direction: column;
  justify-content: center;
  align-items: center;
  z-index: 9999;
  color: white;
}

.loading-spinner {
  width: 40px;
  height: 40px;
  border: 4px solid #f3f3f3;
  border-top: 4px solid #409EFF;
  border-radius: 50%;
  animation: spin 1s linear infinite;
  margin-bottom: 1rem;
}

@keyframes spin {
  0% { transform: rotate(0deg); }
  100% { transform: rotate(360deg); }
}

.message {
  position: fixed;
  top: 20px;
  right: 20px;
  padding: 1rem 1.5rem;
  border-radius: 4px;
  color: white;
  z-index: 1000;
  display: flex;
  align-items: center;
  gap: 1rem;
}

.message.success {
  background-color: #4CAF50;
}

.message.error {
  background-color: #f44336;
}

.message.info {
  background-color: #2196F3;
}

.message button {
  background: none;
  border: none;
  color: white;
  font-size: 1.2rem;
  cursor: pointer;
}
</style>
