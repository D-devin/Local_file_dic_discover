export const handleApiError = (error) => {
  if (error.response) {
    // 服务器返回错误状态码
    const status = error.response.status
    const message = error.response.data?.message || '服务器错误'
    
    switch (status) {
      case 400:
        return '请求参数错误'
      case 401:
        return '未授权访问'
      case 404:
        return '请求的资源不存在'
      case 500:
        return '服务器内部错误'
      default:
        return message || `HTTP错误: ${status}`
    }
  } else if (error.request) {
    // 请求发送但没有收到响应
    return '网络连接失败，请检查网络或服务器状态'
  } else {
    // 其他错误
    return error.message || '未知错误'
  }
}
