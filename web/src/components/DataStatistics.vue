<template>
  <div class="statistics">
    <h3>统计信息</h3>
    
    <div class="stats-grid">
      <!-- 词频统计 -->
      <div class="stats-card">
        <h4>词频统计</h4>
        <div v-if="wordCountLoading" class="loading">加载中...</div>
        <div v-else-if="wordCount.length" class="word-count-list">
          <div 
            v-for="(item, index) in wordCount.slice(0, 20)" 
            :key="index"
            class="word-item"
          >
            <span class="word">{{ item.word }}</span>
            <span class="count">{{ item.count }}</span>
            <div class="bar" :style="{ width: (item.count / maxCount * 100) + '%' }"></div>
          </div>
        </div>
        <button @click="fetchWordCount" class="refresh-btn">刷新数据</button>
      </div>

      <!-- 文件统计 -->
      <div class="stats-card">
        <h4>文件统计</h4>
        <div v-if="fileStatsLoading" class="loading">加载中...</div>
        <div v-else-if="fileStats" class="file-stats">
          <div class="stat-item">
            <label>总文件数:</label>
            <span>{{ fileStats.total_files }}</span>
          </div>
          <div class="stat-item">
            <label>总词数:</label>
            <span>{{ fileStats.total_words }}</span>
          </div>
          <div class="stat-item">
            <label>唯一词数:</label>
            <span>{{ fileStats.unique_words }}</span>
          </div>
          <div class="stat-item">
            <label>平均文件大小:</label>
            <span>{{ formatFileSize(fileStats.avg_file_size) }}</span>
          </div>
          
          <!-- 文件类型分布 -->
          <div class="file-types" v-if="fileStats.file_types">
            <h5>文件类型分布</h5>
            <div 
              v-for="(count, type) in fileStats.file_types" 
              :key="type"
              class="type-item"
            >
              <span>{{ type.toUpperCase() }}</span>
              <span>{{ count }} 个文件</span>
            </div>
          </div>
        </div>
        <button @click="fetchFileStats" class="refresh-btn">刷新数据</button>
      </div>
    </div>

    <!-- 图表展示 -->
    <div class="charts-section">
      <div class="chart-container">
        <canvas ref="wordChart" width="400" height="200"></canvas>
      </div>
    </div>
  </div>
</template>

<script>
import { statisticsAPI } from '@/api/statisticsApi'

export default {
  name: 'DataStatistics',
  inject: ['showMessage'],
  data() {
    return {
      wordCount: [],
      wordCountLoading: false,
      fileStats: null,
      fileStatsLoading: false,
      maxCount: 0
    }
  },
  methods: {
    async fetchWordCount() {
      this.wordCountLoading = true
      try {
        this.wordCount = await statisticsAPI.getWordCount()
        if (this.wordCount.length > 0) {
          this.maxCount = Math.max(...this.wordCount.map(item => item.count))
        }
        this.drawWordChart()
      } catch (error) {
        this.showMessage('获取词频统计失败', 'error')
        console.error('获取词频统计失败:', error)
      } finally {
        this.wordCountLoading = false
      }
    },

    async fetchFileStats() {
      this.fileStatsLoading = true
      try {
        this.fileStats = await statisticsAPI.getFileStats()
      } catch (error) {
        this.showMessage('获取文件统计失败', 'error')
        console.error('获取文件统计失败:', error)
      } finally {
        this.fileStatsLoading = false
      }
    },

    formatFileSize(bytes) {
      if (bytes === 0) return '0 Bytes'
      const k = 1024
      const sizes = ['Bytes', 'KB', 'MB', 'GB']
      const i = Math.floor(Math.log(bytes) / Math.log(k))
      return parseFloat((bytes / Math.pow(k, i)).toFixed(2)) + ' ' + sizes[i]
    },

 drawWordChart() {
      // 简单的Canvas图表绘制
      const canvas = this.$refs.wordChart
      if (!canvas || !this.wordCount.length) return
      
      const ctx = canvas.getContext('2d')
      const width = canvas.width
      const height = canvas.height
      
      // 清空画布
      ctx.clearRect(0, 0, width, height)
      
      // 绘制柱状图
      const barWidth = width / Math.min(this.wordCount.length, 10)
      const maxHeight = height - 40
      
      this.wordCount.slice(0, 10).forEach((item, index) => {
        const barHeight = (item.count / this.maxCount) * maxHeight
        const x = index * barWidth
        const y = height - barHeight - 20
        
        // 绘制柱子
        ctx.fillStyle = '#409EFF'
        ctx.fillRect(x + 5, y, barWidth - 10, barHeight)
        
        // 绘制标签
        ctx.fillStyle = '#333'
        ctx.font = '12px Arial'
        ctx.textAlign = 'center'
        ctx.fillText(item.word, x + barWidth/2, height - 5)
        
        // 绘制数值
        ctx.fillText(item.count, x + barWidth/2, y - 5)
      })
    }
  },

  async mounted() {
    await this.fetchWordCount()
    await this.fetchFileStats()
  }
}
</script>

<style scoped>
.statistics {
  max-width: 1200px;
  margin: 0 auto;
}

.stats-grid {
  display: grid;
  grid-template-columns: 1fr 1fr;
  gap: 2rem;
  margin-bottom: 2rem;
}

.stats-card {
  background: #f9f9f9;
  padding: 1.5rem;
  border-radius: 8px;
  border: 1px solid #e0e0e0;
}

.stats-card h4 {
  margin-bottom: 1rem;
  color: #333;
}

.word-count-list {
  max-height: 400px;
  overflow-y: auto;
}

.word-item {
  display: flex;
  align-items: center;
  padding: 0.5rem 0;
  position: relative;
}

.word {
  flex: 0 0 120px;
  font-weight: 500;
}

.count {
  flex: 0 0 60px;
  text-align: right;
  font-weight: bold;
  color: #409EFF;
}

.bar {
  height: 4px;
  background-color: #409EFF;
  margin-left: 1rem;
  border-radius: 2px;
  transition: width 0.3s ease;
}

.file-stats {
  display: flex;
  flex-direction: column;
  gap: 1rem;
}

.stat-item {
  display: flex;
  justify-content: space-between;
  padding: 0.5rem 0;
  border-bottom: 1px solid #eee;
}

.stat-item label {
  font-weight: 500;
  color: #666;
}

.stat-item span {
  font-weight: bold;
  color: #333;
}

.file-types {
  margin-top: 1rem;
  padding-top: 1rem;
  border-top: 2px solid #e0e0e0;
}

.type-item {
  display: flex;
  justify-content: space-between;
  padding: 0.25rem 0;
  font-size: 0.9rem;
}

.refresh-btn {
  margin-top: 1rem;
  padding: 0.5rem 1rem;
  background-color: #409EFF;
  color: white;
  border: none;
  border-radius: 4px;
  cursor: pointer;
  transition: background-color 0.3s;
}

.refresh-btn:hover {
  background-color: #367cc7;
}

.charts-section {
  background: #f9f9f9;
  padding: 1.5rem;
  border-radius: 8px;
  border: 1px solid #e0e0e0;
}

.chart-container {
  display: flex;
  justify-content: center;
}

.loading {
  text-align: center;
  padding: 2rem;
  color: #666;
}

@media (max-width: 768px) {
  .stats-grid {
    grid-template-columns: 1fr;
  }
}
</style>