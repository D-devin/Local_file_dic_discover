import axios from 'axios'

// 创建axios实例
const api = axios.create({
  baseURL: 'http://localhost:5000', // Flask默认端口
  timeout: 30000, // 30秒超时，因为文件处理可能需要较长时间
  headers: {
    'Content-Type': 'application/json'
  }
})

// 请求拦截器
api.interceptors.request.use(
  config => {
    console.log('发送请求:', config.method.toUpperCase(), config.url)
    return config
  },
  error => {
    return Promise.reject(error)
  }
)

// 响应拦截器
api.interceptors.response.use(
  response => {
    console.log('收到响应:', response.status, response.config.url)
    return response.data
  },
  error => {
    console.error('请求错误:', error.response?.status, error.message)
    return Promise.reject(error)
  }
)

export default api
