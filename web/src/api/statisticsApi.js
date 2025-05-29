import api from './config'

export const statisticsAPI = {
  // 获取词频统计
  getWordCount() {
    return api.get('/api/statistics/word-count')
  },

  // 获取文件统计信息
  getFileStats() {
    return api.get('/api/statistics/file-stats')
  }
}
