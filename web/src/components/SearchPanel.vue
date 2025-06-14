<template>
  <div class="search-panel">
    <div class="search-header">
      <h3>🔍 智能搜索</h3>
      <select v-model="searchType" class="search-type-select">
      <option value="textrank">TextRank 搜索</option>
      <option value="tfidf">TF-IDF 搜索</option>
      <option value="lsi">LSI 语义搜索</option>
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
          <div v-if="getMatchedWords(result)" class="matched-words">
            <span class="tag" v-for="word in getMatchedWords(result)" :key="word">{{ word }}</span>
          </div>
          <div v-if="result.algorithm === 'LSI' && result.lsi_components" class="lsi-info">
            <div class="lsi-components">
              <span class="info-label">语义主题:</span>
              <span v-for="component in result.lsi_components.slice(0, 3)" :key="component.dimension" class="lsi-tag">
                主题{{ component.dimension }}({{ component.doc_weight }})
              </span>
            </div>
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
      searchType: 'textrank'
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
        case 'textrank':
          return `TextRank: ${result.textrank_score || 0} (相关性: ${result.relevance_score || 0})`;
        case 'tfidf':
          return `TF-IDF: ${result.tfidf_score || 0} (${result.relevance || 'medium'})`;
        case 'lsi':
          return `LSI相似度: ${result.lsi_similarity || 0} (${result.relevance || 'medium'})`;
        default:
          return '未知算法';
      }
    },

    
    // 从路径中提取文件名
    getFileName(fullPath) {
      return fullPath.split('/').pop() || fullPath;
    },
    // 获取匹配的词汇
    getMatchedWords(result) {
      if (result.matched_words) {
        // TextRank和TF-IDF返回的是字符串数组或对象数组
        if (Array.isArray(result.matched_words)) {
          return result.matched_words.map(word => 
            typeof word === 'string' ? word : word.word || word
          );
        }
      }

      if (result.semantic_matched_words) {
        // LSI返回的语义匹配词
        return result.semantic_matched_words.map(item => item.word);
      }

      return [];
},

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
.lsi-info {
  margin-top: 0.5rem;
  padding: 0.5rem;
  background: #f8f9fa;
  border-radius: 4px;
  font-size: 0.875rem;
}

.lsi-components {
  display: flex;
  align-items: center;
  gap: 0.5rem;
  flex-wrap: wrap;
}

.info-label {
  color: #6c757d;
  font-weight: 500;
}

.lsi-tag {
  background: #fff3cd;
  color: #856404;
  padding: 0.25rem 0.5rem;
  border-radius: 12px;
  font-size: 0.75rem;
}

</style>
