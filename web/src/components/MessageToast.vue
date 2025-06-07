<template>
  <transition name="message">
    <div v-if="message" :class="['message-toast', message.type]">
      <span>{{ message.text }}</span>
      <button @click="$emit('close')" class="message-close">×</button>
    </div>
  </transition>
</template>

<script>
export default {
  name: 'MessageToast',
  props: {
    message: {
      type: Object,
      default: null
    }
  },
  emits: ['close']
}
</script>

<style scoped>
.message-toast {
  position: fixed;
  top: 20px;
  right: 20px;
  padding: 12px 20px;
  border-radius: 4px;
  color: white;
  display: flex;
  align-items: center;
  gap: 10px;
  z-index: 1000;
  min-width: 300px;
  box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
}

.message-toast.info {
  background-color: #3498db;
}

.message-toast.error {
  background-color: #e74c3c;
}

.message-toast.success {
  background-color: #27ae60;
}

.message-toast.warning {
  background-color: #f39c12;
}

.message-close {
  background: none;
  border: none;
  color: white;
  font-size: 18px;
  cursor: pointer;
  padding: 0;
  width: 20px;
  height: 20px;
  display: flex;
  align-items: center;
  justify-content: center;
}

.message-enter-active, .message-leave-active {
  transition: all 0.3s ease;
}

.message-enter-from {
  transform: translateX(100%);
  opacity: 0;
}

.message-leave-to {
  transform: translateX(100%);
  opacity: 0;
}
</style>
