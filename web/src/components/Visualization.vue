<template>
  <div class="visualization">
    <h3>数据可视化</h3>
    
    <div class="viz-buttons">
      <button @click="generateWordcloud">生成词云图</button>
      <button @click="generateRanking">关键词排行</button>
      <button @click="analyzeFiles">文件分析</button>
      <button @click="generateReport">综合报告</button>
    </div>

    <!-- 显示生成的图片 -->
    <div class="viz-results" v-if="imageUrl">
      <img :src="imageUrl" alt="可视化结果" style="max-width: 100%;">
      <button @click="downloadImage">下载图片</button>
    </div>
  </div>
</template>

<script>
import { visualizationAPI } from '@/api/visualizationApi'
import { toolsAPI } from '@/api/toolsApi'

export default {
  name: 'Visualization',
  data() {
    return {
      imageUrl: null,
      currentFilename: null
    }
  },
  methods: {
    async generateWordcloud() {
      try {
        const result = await visualizationAPI.generateWordcloud({
          // 根据后端需要的参数
          max_words: 100,
          width: 800,
          height: 600
        })
        this.handleVisualizationResult(result)
      } catch (error) {
        console.error('生成词云图失败:', error)
      }
    },

    async generateRanking() {
      try {
        const result = await visualizationAPI.generateRanking({
          top_n: 20
        })
        this.handleVisualizationResult(result)
      } catch (error) {
        console.error('生成排行榜失败:', error)
      }
    },

    handleVisualizationResult(result) {
      if (result.filename) {
        this.currentFilename = result.filename
        // 构建图片URL
        this.imageUrl = `http://localhost:5000/api/download/${result.filename}`
      }
    },

    async downloadImage() {
      if (!this.currentFilename) return
      
      try {
        const blob = await toolsAPI.downloadFile(this.currentFilename)
        // 创建下载链接
        const url = window.URL.createObjectURL(blob)
        const link = document.createElement('a')
        link.href = url
        link.download = this.currentFilename
        link.click()
        window.URL.revokeObjectURL(url)
      } catch (error) {
        console.error('下载失败:', error)
      }
    }
  }
}
</script>
