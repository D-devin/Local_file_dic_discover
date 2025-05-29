import api from './config'

export const visualizationAPI = {
  // 生成词云图
  generateWordcloud(params) {
    return api.post('/api/visualization/wordcloud', params)
  },

  // 生成关键词排行
  generateRanking(params) {
    return api.post('/api/visualization/ranking', params)
  },

  // 文件类型分析
  analyzeFileTypes(params) {
    return api.post('/api/visualization/file-analysis', params)
  },

  // 生成综合报告
  generateComprehensiveReport(params) {
    return api.post('/api/visualization/comprehensive', params)
  }
}
