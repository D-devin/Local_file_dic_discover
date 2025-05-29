import api from './config'

export const searchAPI = {
  // 简单搜索
  simpleSearch(query) {
    return api.post('/api/search/simple', { query })
  },

  // 布尔搜索
  booleanSearch(query) {
    return api.post('/api/search/boolean', { query })
  },

  // 模糊搜索
  fuzzySearch(query) {
    return api.post('/api/search/fuzzy', { query })
  },

  // 关键词搜索
  keywordSearch(keywords) {
    return api.post('/api/search/keyword', { keywords })
  },

  // 高级搜索
  advancedSearch(searchParams) {
    return api.post('/api/search/advanced', searchParams)
  }
}
