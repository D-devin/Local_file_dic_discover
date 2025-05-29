const API_BASE_URL = 'http://localhost:5000'

class ToolsAPI {
  async request(url, options = {}) {
    const response = await fetch(`${API_BASE_URL}${url}`, {
      headers: {
        'Content-Type': 'application/json',
        ...options.headers
      },
      ...options
    })
    
    if (!response.ok) {
      throw new Error(`HTTP error! status: ${response.status}`)
    }
    
    return await response.json()
  }

  // 健康检查
  async healthCheck() {
    return this.request('/health')
  }

  // 上传配置文件
  async uploadConfig(configPath) {
    return this.request('/upload_config', {
      method: 'POST',
      body: JSON.stringify({ config_path: configPath })
    })
  }

  // 搜索功能
  async tfSearch(sessionId, query) {
    return this.request('/tf_search', {
      method: 'POST',
      body: JSON.stringify({ session_id: sessionId, query })
    })
  }

  async tfidfSearch(sessionId, query) {
    return this.request('/tfidf_search', {
      method: 'POST',
      body: JSON.stringify({ session_id: sessionId, query })
    })
  }

  async wordVectorSearch(sessionId, query) {
    return this.request('/word_vector_search', {
      method: 'POST',
      body: JSON.stringify({ session_id: sessionId, query })
    })
  }

  // 可视化功能
  async generateWordcloud(sessionId, query) {
    return this.request('/generate_wordcloud', {
      method: 'POST',
      body: JSON.stringify({ session_id: sessionId, query })
    })
  }

  async generateKeywordRanking(sessionId, query) {
    return this.request('/generate_keyword_ranking', {
      method: 'POST',
      body: JSON.stringify({ session_id: sessionId, query })
    })
  }

  async analyzeFileExtensions(sessionId) {
    return this.request('/analyze_file_extensions', {
      method: 'POST',
      body: JSON.stringify({ session_id: sessionId })
    })
  }

  async createWordFrequencyChart(sessionId, chartType = 'bar', topN = 20) {
    return this.request('/create_word_frequency_chart', {
      method: 'POST',
      body: JSON.stringify({ 
        session_id: sessionId, 
        chart_type: chartType, 
        top_n: topN 
      })
    })
  }

  // 统计信息
  async getWordStatistics(sessionId) {
    return this.request('/get_word_statistics', {
      method: 'POST',
      body: JSON.stringify({ session_id: sessionId })
    })
  }

  async getFileStatistics(sessionId) {
    return this.request('/get_file_statistics', {
      method: 'POST',
      body: JSON.stringify({ session_id: sessionId })
    })
  }

  // 会话管理
  async clearSession(sessionId) {
    return this.request('/clear_session', {
      method: 'POST',
      body: JSON.stringify({ session_id: sessionId })
    })
  }

  async getSessionInfo() {
    return this.request('/get_session_info')
  }

  // 文件下载
  getDownloadUrl(filename) {
    return `${API_BASE_URL}/download/${filename}`
  }
}

export const toolsAPI = new ToolsAPI()
