<template>
  <nav class="navbar">
    <div class="nav-brand">
      <h1>📚 文档智能检索系统</h1>
    </div>
    <div class="nav-status">
      <div class="status-badge" :class="{ online: healthStatus, offline: !healthStatus }">
        <span class="status-dot"></span>
        {{ healthStatus ? '服务在线' : '服务离线' }}
      </div>
      <button @click="handleCheckHealth" class="btn-refresh">🔄</button>
    </div>
  </nav>
</template>

<script>
export default {
  name: 'NavBar',
  props: {
    healthStatus: {
      type: Boolean,
      default: false
    }
  },
  emits: ['check-health'],
  methods: {
    handleCheckHealth() {
      this.$emit('check-health');
    }
  }
}
</script>

<style scoped>
.navbar {
  background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
  padding: 1rem 2rem;
  display: flex;
  align-items: center;
  justify-content: space-between;
  box-shadow: 0 2px 10px rgba(0, 0, 0, 0.1);
  color: white;
}

.nav-brand h1 {
  margin: 0;
  font-size: 1.5rem;
  font-weight: 600;
}

.nav-status {
  display: flex;
  align-items: center;
  gap: 1rem;
}

.status-badge {
  display: flex;
  align-items: center;
  gap: 0.5rem;
  padding: 0.5rem 1rem;
  border-radius: 20px;
  font-size: 0.9rem;
  font-weight: 500;
}

.status-badge.online {
  background-color: rgba(39, 174, 96, 0.2);
  border: 1px solid rgba(39, 174, 96, 0.5);
}

.status-badge.offline {
  background-color: rgba(231, 76, 60, 0.2);
  border: 1px solid rgba(231, 76, 60, 0.5);
}

.status-dot {
  width: 8px;
  height: 8px;
  border-radius: 50%;
}

.status-badge.online .status-dot {
  background-color: #27ae60;
}

.status-badge.offline .status-dot {
  background-color: #e74c3c;
}

.btn-refresh {
  background: rgba(255, 255, 255, 0.2);
  border: 1px solid rgba(255, 255, 255, 0.3);
  color: white;
  padding: 0.5rem;
  border-radius: 50%;
  cursor: pointer;
  transition: all 0.3s ease;
  width: 40px;
  height: 40px;
  display: flex;
  align-items: center;
  justify-content: center;
}

.btn-refresh:hover {
  background: rgba(255, 255, 255, 0.3);
  transform: rotate(180deg);
}
</style>
