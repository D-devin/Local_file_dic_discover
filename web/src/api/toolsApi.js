import api from './config'

export const toolsAPI = {
  // 下载生成的图片文件
  downloadFile(filename) {
    return api.get(`/api/download/${filename}`, {
      responseType: 'blob' // 重要：设置响应类型为blob
    })
  },

  // 健康检查
  healthCheck() {
    return api.get('/api/health')
  }
}
