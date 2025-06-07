<template>
  <div class="stats-grid">
    <div class="stat-card">
      <div class="stat-icon">📝</div>
      <h4>词汇统计</h4>
      <div v-if="wordStats" class="stat-data">
        <div class="stat-item">
          <span class="stat-label">总词汇数</span>
          <span class="stat-value">{{ wordStats.total_words }}</span>
        </div>
        <div class="stat-item">
          <span class="stat-label">唯一词汇</span>
          <span class="stat-value">{{ wordStats.unique_words }}</span>
        </div>
        <div class="stat-item">
          <span class="stat-label">高频词汇</span>
          <ul class="top-words">
            <li v-for="(count, word, index) in wordStats.top_words" :key="index">
              {{ word }}: {{ count }}
            </li>
          </ul>
        </div>
      </div>
      <button @click="handleGetWordStatistics" :disabled="loading" class="btn-stat">
        获取统计
      </button>
    </div>

    <div class="stat-card">
      <div class="stat-icon">📁</div>
      <h4>文件统计</h4>
      <div v-if="fileStats" class="stat-data">
        <div class="stat-item">
          <span class="stat-label">文件总数</span>
          <span class="stat-value">{{ fileStats.file_count }}</span>
        </div>
        <div class="stat-item">
          <span class="stat-label">文件类型</span>
          <ul class="file-types">
            <li v-for="(count, type) in fileStats.file_types" :key="type">
              {{ type }}: {{ count }}
            </li>
          </ul>
        </div>
      </div>
      <button @click="handleGetFileStatistics" :disabled="loading" class="btn-stat">
        获取统计
      </button>
    </div>
  </div>
</template>

<script>
export default {
  name: 'StatisticsPanel',
  props: {
    loading: {
      type: Boolean,
      default: false
    },
    wordStats: {
      type: Object,
      default: null
    },
    fileStats: {
      type: Object,
      default: null
    }
  },
  emits: ['get-word-statistics', 'get-file-statistics'],
  methods: {
    handleGetWordStatistics() {
      this.$emit('get-word-statistics');
    },
    
    handleGetFileStatistics() {
      this.$emit('get-file-statistics');
    }
  }
}
</script>

<style scoped>
.stats-grid {
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(400px, 1fr));
  gap: 1.5rem;
  padding: 1rem;
}

.stat-card {
  background: white;
  border-radius: 8px;
  padding: 1.5rem;
  box-shadow: 0 2px 8px rgba(0, 0, 0, 0.1);
  border: 1px solid #e9ecef;
}

.stat-icon {
  font-size: 2rem;
  margin-bottom: 0.5rem;
}

.stat-card h4 {
  margin: 0 0 1rem 0;
  color: #333;
  display: flex;
  align-items: center;
  gap: 0.5rem;
}

.stat-data {
  margin-bottom: 1rem;
}

.stat-item {
  display: flex;
  justify-content: space-between;
  align-items: flex-start;
  padding: 0.5rem 0;
  border-bottom: 1px solid #f0f0f0;
}

.stat-item:last-child {
  border-bottom: none;
}

.stat-label {
  font-weight: 500;
  color: #666;
  flex: 1;
}

.stat-value {
  font-weight: bold;
  color: #333;
  font-size: 1.1rem;
}

.top-words,
.file-types {
  list-style: none;
  padding: 0;
  margin: 0;
  max-height: 150px;
  overflow-y: auto;
  background: #f8f9fa;
  border-radius: 4px;
  padding: 0.5rem;
}

.top-words li,
.file-types li {
  padding: 0.25rem 0;
  font-size: 0.9rem;
  color: #495057;
  display: flex;
  justify-content: space-between;
}

.btn-stat {
  width: 100%;
  padding: 0.75rem;
  background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
  color: white;
  border: none;
  border-radius: 6px;
  cursor: pointer;
  font-weight: 500;
  transition: all 0.3s ease;
}

.btn-stat:hover:not(:disabled) {
  transform: translateY(-2px);
  box-shadow: 0 4px 15px rgba(102, 126, 234, 0.4);
}

.btn-stat:disabled {
  opacity: 0.6;
  cursor: not-allowed;
  transform: none;
}

@media (max-width: 768px) {
  .stats-grid {
    grid-template-columns: 1fr;
  }
  
  .stat-item {
    flex-direction: column;
    align-items: flex-start;
    gap: 0.5rem;
  }
}
</style>
