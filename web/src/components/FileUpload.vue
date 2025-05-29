<template>
  <div class="file-upload">
    <h3>文件上传与处理</h3>
    
    <!-- 配置文件上传 -->
    <div class="upload-section">
      <label>上传配置文件：</label>
      <input 
        type="file" 
        @change="handleConfigUpload" 
        accept=".json,.yaml,.yml"
      >
      <button @click="uploadConfig" :disabled="!configFile || uploading">
        {{ uploading ? '上传中...' : '上传配置' }}
      </button>
    </div>

    <!-- 文件处理 -->
    <div class="process-section">
      <button @click="processFiles" :disabled="processing">
        {{ processing ? '处理中...' : '处理文件并构建索引' }}
      </button>
    </div>

    <!-- 文件列表 -->
    <div class="file-list">
      <h4>文件列表</h4>
      <button @click="fetchFileList">刷新列表</button>
      <ul v-if="fileList.length">
        <li v-for="file in fileList" :key="file.id">
          {{ file.name }} - {{ file.size }}
        </li>
      </ul>
      <p v-else>暂无文件</p>
    </div>
  </div>
</template>

<script>
import { fileAPI } from '@/api/fileApi'

export default {
  name: 'FileUpload',
  data() {
    return {
      configFile: null,
      uploading: false,
      processing: false,
      fileList: []
    }
  },
  methods: {
    handleConfigUpload(event) {
      this.configFile = event.target.files[0]
    },

    async uploadConfig() {
      if (!this.configFile) return
      
      this.uploading = true
      try {
        const result = await fileAPI.uploadConfig(this.configFile)
        this.$message.success('配置文件上传成功')
        console.log('上传结果:', result)
      } catch (error) {
        this.$message.error('配置文件上传失败')
        console.error('上传错误:', error)
      } finally {
        this.uploading = false
      }
    },

    async processFiles() {
      this.processing = true
      try {
        const result = await fileAPI.processFiles({
          // 根据后端需要的参数格式传递数据
          action: 'build_index'
        })
        this.$message.success('文件处理完成')
        console.log('处理结果:', result)
        // 处理完成后刷新文件列表
        this.fetchFileList()
      } catch (error) {
        this.$message.error('文件处理失败')
        console.error('处理错误:', error)
      } finally {
        this.processing = false
      }
    },

    async fetchFileList() {
      try {
        this.fileList = await fileAPI.getFileList()
      } catch (error) {
        console.error('获取文件列表失败:', error)
      }
    }
  },

  mounted() {
    this.fetchFileList()
  }
}
</script>
