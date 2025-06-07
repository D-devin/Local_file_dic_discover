<template>
  <div class="search-panel">
    <div class="search-header">
      <h3>🔍 智能搜索</h3>
      <select v-model="searchType" class="search-type-select">
        <option value="tf">TF 词频搜索</option>
        <option value="tfidf">TF-IDF 搜索</option>
        <option value="word_vector">词向量搜索</option>
      </select>
    </div>
    
    <div class="search-input-group">
      <input 
        v-model="searchQuery" 
        type="text" 
        placeholder="输入搜索关键词..."
        class="search-input"
        @keyup.enter="handleSearch"
      >
      <button @click="handleSearch" :disabled="!searchQuery || loading" class="btn-search">
        {{ loading ? '搜索中...' : '搜索' }}
      </button>
    </div>

    <!-- 搜索结果 -->
    <div v-if="searchResults" class="search-results">
      <div class="results-header">
        <h4>搜索结果 ({{ searchResults.total_results || 0 }} 条)</h4>
      </div>
      
      <div v-if="searchResults.results && searchResults.results.length > 0" class="results-list">
        <div v-for="(result, index) in searchResults.results" :key="index" class="result-card">
          <div class="result-header">
            <span class="file-name">📄 {{ getFileName(result.file_name) }}</span>
            <span class="result-score">
              {{ formatScore(result, searchType) }}
            </span>
          </div>
          <div class="result-preview">{{ result.content_preview }}</div>
          <div v-if="result.matched_words" class="matched-words">
            <span class="tag" v-for="word in result.matched_words" :key="word">{{ word }}</span>
          </div>
          <div v-if="result.positions" class="positions-info">
            位置: {{ result.positions.join(', ') }}
          </div>
        </div>
      </div>
      
      <div v-else class="no-results">
        <div class="no-results-icon">🔍</div>
        <p>未找到相关结果</p>
      </div>
    </div>
  </div>
</template>

<script>
export default {
  name: 'SearchPanel',
  props: {
    loading: {
      type: Boolean,
      default: false
    },
    searchResults: {
      type: Object,
      default: null
    }
  },
  emits: ['search'],
  data() {
    return {
      searchQuery: '',
      searchType: 'tf'
    }
  },
  methods: {
    handleSearch() {
      if (!this.searchQuery) return;
      
      this.$emit('search', {
        query: this.searchQuery,
        type: this.searchType
      });
    },
    
    // 格式化分数显示
    formatScore(result, searchType) {
      switch (searchType) {
        case 'tf':
          return `词频: ${result.word_frequency} (TF: ${result.tf_score})`;
        case 'tfidf':
          return `TF-IDF: ${result.tfidf_score}`;
        case 'word_vector':
          return `向量相似度: ${result.vector_similarity}`;
        default:
          return '';
      }
    },
    
    // 从路径中提取文件名
    getFileName(fullPath) {
      return fullPath.split('/').pop() || fullPath;
    }
  }
}
</script>

<style scoped>
.search-panel {
  background: white;
  border-radius: 8px;
  padding: 1.5rem;
  box-shadow: 0 2px 8px rgba(0, 0, 0, 0.1);
}

.search-header {
  display: flex;
  align-items: center;
  justify-content: space-between;
  margin-bottom: 1.5rem;
}

.search-header h3 {
  margin: 0;
  color: #333;
}

.search-type-select {
  padding: 0.5rem;
  border: 1px solid #ddd;
  border-radius: 4px;
  background: white;
}

.search-input-group {
  display: flex;
  gap: 1rem;
  margin-bottom: 1.5rem;
}

.search-input {
  flex: 1;
  padding: 0.75rem;
  border: 1px solid #ddd;
  border-radius: 6px;
  font-size: 1rem;
}

.btn-search {
  padding: 0.75rem 1.5rem;
  background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
  color: white;
  border: none;
  border-radius: 6px;
  cursor: pointer;
  font-weight: 500;
  transition: all 0.3s ease;
}

.btn-search:hover:not(:disabled) {
  transform: translateY(-2px);
  box-shadow: 0 4px 15px rgba(102, 126, 234, 0.4);
}

.btn-search:disabled {
  opacity: 0.6;
  cursor: not-allowed;
}

.search-results {
  margin-top: 1.5rem;
}

.results-header {
  margin-bottom: 1rem;
}

.results-header h4 {
  margin: 0;
  color: #333;
}

.results-list {
  display: flex;
  flex-direction: column;
  gap: 1rem;
}

.result-card {
  background: #f8f9fa;
  border: 1px solid #e9ecef;
  border-radius: 6px;
  padding: 1rem;
  transition: all 0.3s ease;
}

.result-card:hover {
  transform: translateY(-2px);
  box-shadow: 0 4px 12px rgba(0, 0, 0, 0.1);
}

.result-header {
  display: flex;
  justify-content: space-between;
  align-items: center;
  margin-bottom: 0.5rem;
}

.file-name {
  font-weight: 600;
  color: #495057;
}

.result-score {
  font-size: 0.875rem;
  color: #6c757d;
  background: #e9ecef;
  padding: 0.25rem 0.5rem;
  border-radius: 12px;
}

.result-preview {
  color: #495057;
  line-height: 1.5;
  margin-bottom: 0.5rem;
}

.matched-words {
  display: flex;
  flex-wrap: wrap;
  gap: 0.5rem;
  margin-bottom: 0.5rem;
}

.tag {
  background: #e3f2fd;
  color: #1976d2;
  padding: 0.25rem 0.5rem;
  border-radius: 12px;
  font-size: 0.875rem;
}

.positions-info {
  font-size: 0.875rem;
  color: #6c757d;
}

.no-results {
  text-align: center;
  padding: 2rem;
  color: #6c757d;
}

.no-results-icon {
  font-size: 3rem;
  margin-bottom: 1rem;
}
</style>
