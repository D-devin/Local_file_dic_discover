<template>
  <div class="viz-grid">
    <!-- 词云图 -->
    <div class="viz-card">
      <div class="viz-header">
        <h4>☁️ 词云图</h4>
      </div>
      <div class="viz-controls">
        <input 
          v-model="wordcloudQuery" 
          type="text" 
          placeholder="输入关键词生成词云"
          class="viz-input">
        <button @click="handleGenerateWordcloud" :disabled="!wordcloudQuery || loading" class="btn-viz">
          生成
        </button>
      </div>
      <div v-if="wordcloudResult?.save_path" class="viz-display">
      <div v-if="imageLoading" class="image-loading">
        <span>图片加载中...</span>
      </div>
      <img 
        v-show="!imageLoading && !imageError"
        :src="getImageUrl(wordcloudResult.save_path)" 
        alt="词云图" 
        class="viz-image"
        @load="onImageLoad"
        @error="onImageError"
        crossorigin="anonymous"
      >
      <div v-if="imageError" class="image-error">
        <span>图片加载失败，请重试</span>
        <button @click="retryImageLoad" class="retry-btn">重新加载</button>
      </div>
  <div class="viz-stats">
    <span>相关文档: {{ wordcloudResult.relevant_docs }}</span>
    <span>总词汇: {{ wordcloudResult.total_words }}</span>
  </div>
</div>

      <div v-else-if="wordcloudResult" class="viz-message">
        {{ wordcloudResult.message }}
      </div>
    </div>

    <!-- 关键词排行 -->
    <div class="viz-card">
      <div class="viz-header">
        <h4>📊 关键词排行</h4>
      </div>
      <div class="viz-controls">
        <input 
          v-model="keywordQuery" 
          type="text" 
          placeholder="输入关键词生成排行"
          class="viz-input"
        >
        <button @click="handleGenerateKeywordRanking" :disabled="!keywordQuery || loading" class="btn-viz">
          生成
        </button>
      </div>
      <div v-if="keywordResult?.save_path" class="viz-display">
        <img :src="getImageUrl(keywordResult.save_path)" alt="关键词排行" class="viz-image">
        <div class="viz-stats">
          <span>分析文件: {{ keywordResult.total_analyzed_files }}</span>
        </div>
      </div>
      <div v-else-if="keywordResult" class="viz-message">
        {{ keywordResult.message }}
      </div>
    </div>

    <!-- 文件类型分析 -->
    <div class="viz-card">
      <div class="viz-header">
        <h4>📈 文件类型分析</h4>
      </div>
      <div class="viz-controls">
        <button @click="handleAnalyzeFileTypes" :disabled="loading" class="btn-viz">
          分析文件类型
        </button>
      </div>
      <div v-if="fileTypeResult?.save_path" class="viz-display">
        <img :src="getImageUrl(fileTypeResult.save_path)" alt="文件类型分析" class="viz-image">
        <div class="viz-stats">
          <span>总文件: {{ fileTypeResult.total_files }}</span>
          <span>文件类型: {{ fileTypeResult.unique_extensions }}</span>
        </div>
      </div>
      <div v-else-if="fileTypeResult" class="viz-message">
        {{ fileTypeResult.message }}
      </div>
    </div>

    <!-- 词频统计 -->
    <div class="viz-card">
      <div class="viz-header">
        <h4>📉 词频统计</h4>
      </div>
      <div class="viz-controls">
        <select v-model="chartType" class="chart-select">
          <option value="bar">柱状图</option>
          <option value="line">折线图</option>
          <option value="pie">饼图</option>
        </select>
        <button @click="handleGenerateWordFrequencyChart" :disabled="loading" class="btn-viz">
          生成图表
        </button>
      </div>
      <div v-if="wordFreqResult?.save_path" class="viz-display">
        <img :src="getImageUrl(wordFreqResult.save_path)" alt="词频统计" class="viz-image">
        <div class="viz-stats">
          <span>分析词汇: {{ wordFreqResult.analyzed_words }}</span>
        </div>
      </div>
      <div v-else-if="wordFreqResult" class="viz-message">
        {{ wordFreqResult.message }}
      </div>
    </div>
  </div>
</template>

<script>
export default {
  name: 'VisualizationPanel',
  props: {
    loading: {
      type: Boolean,
      default: false
    },
    wordcloudResult: {
      type: Object,
      default: null
    },
    keywordResult: {
      type: Object,
      default: null
    },
    fileTypeResult: {
      type: Object,
      default: null
    },
    wordFreqResult: {
      type: Object,
      default: null
    }
  },
  emits: ['generate-wordcloud', 'generate-keyword-ranking', 'analyze-file-types', 'generate-word-frequency-chart'],
  data() {
    return {
      wordcloudQuery: '',
      keywordQuery: '',
      chartType: 'bar'
    }
  },
  methods: {
    handleGenerateWordcloud() {
      if (!this.wordcloudQuery) return;
      this.$emit('generate-wordcloud', this.wordcloudQuery);
    },
    
    handleGenerateKeywordRanking() {
      if (!this.keywordQuery) return;
      this.$emit('generate-keyword-ranking', this.keywordQuery);
    },
    
    handleAnalyzeFileTypes() {
      this.$emit('analyze-file-types');
    },
    
    handleGenerateWordFrequencyChart() {
      this.$emit('generate-word-frequency-chart', this.chartType);
    },
    
    // 获取图片URL
    // 获取图片URL
    getImageUrl(path) {
      if (!path) return '';

      // 处理路径，确保只获取文件名（兼容Windows和Linux路径）
      let filename = path;

      // 如果包含路径分隔符，提取最后的文件名
      if (filename.includes('/')) {
        filename = filename.split('/').pop();
      }
      if (filename.includes('\\')) {
        filename = filename.split('\\').pop();
      }

      // 移除可能的子目录前缀（如果文件名格式为 "subdirectory/filename.png"）
      if (filename.includes('/')) {
        filename = filename.split('/').pop();
      }

      // 添加时间戳避免缓存问题
      const timestamp = new Date().getTime();
      return `http://localhost:5000/download/${filename}?t=${timestamp}`;
    }
  }
}
</script>

<style scoped>
.viz-grid {
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(400px, 1fr));
  gap: 1.5rem;
  padding: 1rem;
}

.viz-card {
  background: white;
  border-radius: 8px;
  padding: 1.5rem;
  box-shadow: 0 2px 8px rgba(0, 0, 0, 0.1);
  border: 1px solid #e9ecef;
}

.viz-header {
  margin-bottom: 1rem;
}

.viz-header h4 {
  margin: 0;
  color: #333;
  display: flex;
  align-items: center;
  gap: 0.5rem;
}

.viz-controls {
  display: flex;
  gap: 1rem;
  margin-bottom: 1rem;
  align-items: center;
}

.viz-input {
  flex: 1;
  padding: 0.5rem;
  border: 1px solid #ddd;
  border-radius: 4px;
  font-size: 0.9rem;
}

.chart-select {
  padding: 0.5rem;
  border: 1px solid #ddd;
  border-radius: 4px;
  background: white;
  min-width: 120px;
}

.btn-viz {
  padding: 0.5rem 1rem;
  background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
  color: white;
  border: none;
  border-radius: 4px;
  cursor: pointer;
  font-weight: 500;
  transition: all 0.3s ease;
  white-space: nowrap;
}

.btn-viz:hover:not(:disabled) {
  transform: translateY(-2px);
  box-shadow: 0 4px 15px rgba(102, 126, 234, 0.4);
}

.btn-viz:disabled {
  opacity: 0.6;
  cursor: not-allowed;
}

.viz-display {
  text-align: center;
}

.viz-image {
  max-width: 100%;
  height: auto;
  border-radius: 4px;
  margin-bottom: 1rem;
  box-shadow: 0 2px 8px rgba(0, 0, 0, 0.1);
}

.viz-stats {
  display: flex;
  justify-content: space-around;
  gap: 1rem;
  font-size: 0.875rem;
  color: #6c757d;
  background: #f8f9fa;
  padding: 0.5rem;
  border-radius: 4px;
}

.viz-stats span {
  padding: 0.25rem 0.5rem;
  background: white;
  border-radius: 4px;
  border: 1px solid #e9ecef;
}

.viz-message {
  text-align: center;
  padding: 2rem;
  color: #6c757d;
  background: #f8f9fa;
  border-radius: 4px;
  font-style: italic;
}

@media (max-width: 768px) {
  .viz-grid {
    grid-template-columns: 1fr;
  }
  
  .viz-controls {
    flex-direction: column;
  }
  
  .viz-stats {
    flex-direction: column;
    gap: 0.5rem;
  }
}
/* 添加到现有样式中 */
.image-loading {
  display: flex;
  justify-content: center;
  align-items: center;
  height: 200px;
  color: #6c757d;
  font-style: italic;
}

.image-error {
  display: flex;
  flex-direction: column;
  justify-content: center;
  align-items: center;
  height: 200px;
  color: #dc3545;
  gap: 1rem;
}

.retry-btn {
  padding: 0.5rem 1rem;
  background: #007bff;
  color: white;
  border: none;
  border-radius: 4px;
  cursor: pointer;
}

.retry-btn:hover {
  background: #0056b3;
}

.viz-image {
  transition: opacity 0.3s ease;
}

</style>
