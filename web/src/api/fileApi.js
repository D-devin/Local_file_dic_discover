import api from './config'

export const fileAPI = {
  // 上传配置文件
  uploadConfig(configFile) {
    const formData = new FormData()
    formData.append('config', configFile)
    
    return api.post('/api/upload-config', formData, {
      headers: {
        'Content-Type': 'multipart/form-data'
      }
    })
  },

  // 处理文件并构建索引
  processFiles(fileData) {
    return api.post('/api/process-files', fileData)
  },

  // 获取文件列表
  getFileList() {
    return api.get('/api/get-file-list')
  }
}
