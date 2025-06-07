<template>
  <div class="upload-section">
    <div class="upload-card">
      <div class="upload-icon">📁</div>
      <h2>开始使用</h2>
      <p>请输入包含文档的文件夹路径</p>
      
      <div class="folder-input-group">
        <input 
          v-model="folderPathInput" 
          type="text" 
          placeholder="例如：C:\MyDocuments 或 /home/user/Documents"
          class="folder-path-input"
          @keyup.enter="handleFolderPath"
        >
        <button @click="handleFolderPath" :disabled="!folderPathInput || loading" class="btn-upload">
          {{ loading ? '处理中...' : '处理文件夹' }}
        </button>
      </div>
      
      <div class="folder-info" v-if="selectedFolder">
        <p>已处理文件夹: <strong>{{ selectedFolder }}</strong></p>
      </div>
      
      <div class="folder-instructions">
        <details>
          <summary>使用说明</summary>
          <ul>
            <li>输入包含文档的文件夹完整路径</li>
            <li>系统会自动处理文件夹中的所有文档</li>
            <li>处理完成后即可使用搜索和分析功能</li>
            <li>示例路径：
              <ul>
                <li>Windows: C:\Users\YourName\Documents</li>
                <li>Mac/Linux: /home/username/Documents</li>
              </ul>
            </li>
          </ul>
        </details>
      </div>
    </div>
  </div>
</template>

<script>
export default {
  name: 'FolderConfig',
  props: {
    loading: {
      type: Boolean,
      default: false
    },
    selectedFolder: {
      type: String,
      default: null
    }
  },
  emits: ['folder-submit'],
  data() {
    return {
      folderPathInput: ''
    }
  },
  methods: {
    handleFolderPath() {
      if (!this.folderPathInput) {
        this.$emit('folder-submit', { error: '请输入文件夹路径' });
        return;
      }
      
      this.$emit('folder-submit', { 
        folderPath: this.folderPathInput,
        error: null 
      });
    }
  }
}
</script>

<style scoped>
.upload-section {
  display: flex;
  justify-content: center;
  align-items: center;
  min-height: 60vh;
  padding: 2rem;
}

.upload-card {
  background: white;
  padding: 3rem;
  border-radius: 12px;
  box-shadow: 0 4px 20px rgba(0, 0, 0, 0.1);
  text-align: center;
  max-width: 600px;
  width: 100%;
}

.upload-icon {
  font-size: 4rem;
  margin-bottom: 1rem;
}

.upload-card h2 {
  color: #333;
  margin-bottom: 0.5rem;
}

.upload-card p {
  color: #666;
  margin-bottom: 2rem;
}

.folder-input-group {
  display: flex;
  gap: 1rem;
  margin-bottom: 1.5rem;
}

.folder-path-input {
  flex: 1;
  padding: 0.75rem;
  border: 1px solid #ddd;
  border-radius: 6px;
  font-size: 1rem;
}

.btn-upload {
  padding: 0.75rem 1.5rem;
  background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
  color: white;
  border: none;
  border-radius: 6px;
  cursor: pointer;
  font-weight: 500;
  transition: all 0.3s ease;
}

.btn-upload:hover:not(:disabled) {
  transform: translateY(-2px);
  box-shadow: 0 4px 15px rgba(102, 126, 234, 0.4);
}

.btn-upload:disabled {
  opacity: 0.6;
  cursor: not-allowed;
}

.folder-info {
  background: #f8f9fa;
  padding: 1rem;
  border-radius: 6px;
  margin-bottom: 1rem;
}

.folder-instructions {
  text-align: left;
  background: #f8f9fa;
  padding: 1rem;
  border-radius: 6px;
}

.folder-instructions summary {
  cursor: pointer;
  font-weight: 500;
  margin-bottom: 0.5rem;
}

.folder-instructions ul {
  margin: 0.5rem 0;
  padding-left: 1.5rem;
}

.folder-instructions li {
  margin: 0.25rem 0;
}
</style>
