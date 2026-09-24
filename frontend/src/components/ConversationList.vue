<script setup>
import { computed, ref } from "vue";
import { Check, ChevronDown, Filter, MessageCirclePlus, Search, SlidersHorizontal } from "lucide-vue-next";

const props = defineProps({
  conversations: { type: Array, required: true },
  selectedId: { type: String, required: true },
  scope: { type: String, required: true },
  agentName: { type: String, required: true },
});

defineEmits(["select", "notify"]);

const query = ref("");
const statusFilter = ref("active");
const sortNewest = ref(true);

const scopeLabels = {
  inbox: ["会话中心", "所有进行中的客户咨询"],
  mine: ["我的会话", "当前由你负责的咨询"],
  waiting: ["待接入", "尚未分配坐席的客户"],
  ai: ["AI 接待", "由机器人协同处理的会话"],
};

const title = computed(() => scopeLabels[props.scope] || scopeLabels.inbox);
const scopedConversations = computed(() => {
  let items = [...props.conversations];
  if (props.scope === "mine") items = items.filter((item) => item.assignee === props.agentName);
  if (props.scope === "waiting") items = items.filter((item) => item.status === "queued");
  if (props.scope === "ai") items = items.filter((item) => item.aiHandled);
  if (statusFilter.value === "active") items = items.filter((item) => item.status !== "resolved");
  if (statusFilter.value === "unread") items = items.filter((item) => item.unread > 0);
  if (statusFilter.value === "resolved") items = items.filter((item) => item.status === "resolved");
  const keyword = query.value.trim().toLowerCase();
  if (keyword) items = items.filter((item) => `${item.customer.name}${item.preview}${item.customer.phone}${item.order?.id || ""}`.toLowerCase().includes(keyword));
  return items.sort((a, b) => sortNewest.value ? b.updatedAt - a.updatedAt : a.updatedAt - b.updatedAt);
});

function formatTime(timestamp) {
  const diff = Date.now() - timestamp;
  if (diff < 60_000) return "刚刚";
  if (diff < 3_600_000) return `${Math.floor(diff / 60_000)} 分钟`;
  if (diff < 86_400_000) return `${Math.floor(diff / 3_600_000)} 小时`;
  return new Date(timestamp).toLocaleDateString("zh-CN", { month: "numeric", day: "numeric" });
}
</script>

<template>
  <section class="conversation-pane">
    <header class="conversation-header">
      <div>
        <div class="title-line"><h1>{{ title[0] }}</h1><span>{{ scopedConversations.length }}</span></div>
        <p>{{ title[1] }}</p>
      </div>
      <button class="icon-control primary-icon" type="button" title="发起会话" aria-label="发起会话" @click="$emit('notify', '新会话入口已打开')"><MessageCirclePlus :size="18" /></button>
    </header>

    <div class="conversation-tools">
      <label class="search-box">
        <Search :size="17" />
        <input v-model="query" type="search" placeholder="搜索客户、订单或内容" aria-label="搜索会话" />
        <kbd>⌘ K</kbd>
      </label>
      <div class="filter-row">
        <div class="segmented" aria-label="会话状态筛选">
          <button type="button" :class="{ active: statusFilter === 'active' }" @click="statusFilter = 'active'">进行中</button>
          <button type="button" :class="{ active: statusFilter === 'unread' }" @click="statusFilter = 'unread'">未读</button>
          <button type="button" :class="{ active: statusFilter === 'resolved' }" @click="statusFilter = 'resolved'">已结束</button>
        </div>
        <button class="filter-control" type="button" :title="sortNewest ? '当前：最新优先' : '当前：最早优先'" @click="sortNewest = !sortNewest">
          <SlidersHorizontal :size="16" /><span>{{ sortNewest ? "最新" : "最早" }}</span><ChevronDown :size="14" />
        </button>
      </div>
    </div>

    <div class="queue-strip">
      <span><i></i>{{ scopedConversations.filter((item) => item.status === 'queued').length }} 位客户等待接入</span>
      <button type="button" @click="$emit('notify', '已按等待时长优先排序')">优先处理</button>
    </div>

    <div class="conversation-scroll">
      <button
        v-for="conversation in scopedConversations"
        :key="conversation.id"
        class="conversation-item"
        :class="{ active: selectedId === conversation.id, unread: conversation.unread }"
        type="button"
        @click="$emit('select', conversation.id)"
      >
        <div class="customer-avatar" :style="{ '--avatar-color': conversation.customer.color }">
          {{ conversation.customer.avatar }}
          <span :class="conversation.customer.online ? 'online' : ''"></span>
        </div>
        <div class="conversation-copy">
          <div class="item-heading">
            <strong>{{ conversation.customer.name }}</strong>
            <time>{{ formatTime(conversation.updatedAt) }}</time>
          </div>
          <p><Bot v-if="conversation.aiHandled" :size="13" />{{ conversation.preview }}</p>
          <div class="item-meta">
            <span v-if="conversation.channel === '微信'" class="channel wechat">微</span>
            <span v-else-if="conversation.channel === '网页'" class="channel web">W</span>
            <span v-else class="channel app">A</span>
            <span>{{ conversation.channel }}</span>
            <em v-if="conversation.priority === 'urgent'">紧急</em>
            <em v-else-if="conversation.status === 'queued'" class="waiting">待接入</em>
            <em v-else-if="conversation.status === 'resolved'" class="resolved"><Check :size="11" /> 已结束</em>
          </div>
        </div>
        <b v-if="conversation.unread" class="unread-count">{{ conversation.unread }}</b>
      </button>

      <div v-if="!scopedConversations.length" class="list-empty-state">
        <Filter :size="25" />
        <strong>没有匹配的会话</strong>
        <span>调整筛选条件或搜索关键词。</span>
      </div>
    </div>
  </section>
</template>
