<template>
  <div class="search-component">
    <h3>搜索功能</h3>
    
    <!-- 搜索类型选择 -->
    <div class="search-type">
      <label>搜索类型：</label>
      <select v-model="searchType">
        <option value="simple">简单搜索</option>
        <option value="boolean">布尔搜索</option>
        <option value="fuzzy">模糊搜索</option>
        <option value="keyword">关键词搜索</option>
        <option value="advanced">高级搜索</option>
      </select>
    </div>

    <!-- 搜索输入 -->
    <div class="search-input">
      <input 
        v-model="searchQuery" 
        placeholder="请输入搜索内容"
        @keyup.enter="performSearch"
      >
      <button @click="performSearch" :disabled="searching">
        {{ searching ? '搜索中...' : '搜索' }}
      </button>
    </div>

    <!-- 搜索结果 -->
    <div class="search-results" v-if="searchResults.length">
      <h4>搜索结果 ({{ searchResults.length }} 条)</h4>
      <div v-for="result in searchResults" :key="result.id" class="result-item">
        <h5>{{ result.title }}</h5>
        <p>{{ result.content }}</p>
        <small>文件: {{ result.filename }}</small>
      </div>
    </div>
  </div>
</template>

<script>
import { searchAPI } from '@/api/searchApi'

export default {
  name: 'SearchComponent',
  data() {
    return {
      searchType: 'simple',
      searchQuery: '',
      searching: false,
      searchResults: []
    }
  },
  methods: {
    async performSearch() {
      if (!this.searchQuery.trim()) return
      
      this.searching = true
      try {
        let result
        switch (this.searchType) {
          case 'simple':
            result = await searchAPI.simpleSearch(this.searchQuery)
            break
          case 'boolean':
            result = await searchAPI.booleanSearch(this.searchQuery)
            break
          case 'fuzzy':
            result = await searchAPI.fuzzySearch(this.searchQuery)
            break
          case 'keyword':
            result = await searchAPI.keywordSearch(this.searchQuery.split(' '))
            break
          case 'advanced':
            result = await searchAPI.advancedSearch({
              query: this.searchQuery,
              // 其他高级搜索参数
            })
            break
        }
        this.searchResults = result.results || result
      } catch (error) {
        this.$message.error('搜索失败')
        console.error('搜索错误:', error)
      } finally {
        this.searching = false
      }
    }
  }
}
</script>
