<script setup>
import { computed, ref } from "vue";
import { BookOpen, BarChart3, X } from "lucide-vue-next";
import AppSidebar from "./components/AppSidebar.vue";
import ConversationList from "./components/ConversationList.vue";
import ChatWorkspace from "./components/ChatWorkspace.vue";
import { conversations as seedConversations, currentAgent, quickArticles } from "./data/mockConversations";

const conversations = ref(structuredClone(seedConversations));
const selectedId = ref(conversations.value[0]?.id || "");
const navScope = ref("inbox");
const mobileView = ref("list");
const overlay = ref("");
const toast = ref("");
let toastTimer;

const selectedConversation = computed(() => conversations.value.find((item) => item.id === selectedId.value) || conversations.value[0]);
const scopeCounts = computed(() => ({
  inbox: conversations.value.filter((item) => item.status !== "resolved").length,
  mine: conversations.value.filter((item) => item.assignee === currentAgent.name && item.status !== "resolved").length,
  waiting: conversations.value.filter((item) => item.status === "queued").length,
  ai: conversations.value.filter((item) => item.aiHandled && item.status !== "resolved").length,
}));

function notify(message) {
  toast.value = message;
  window.clearTimeout(toastTimer);
  toastTimer = window.setTimeout(() => { toast.value = ""; }, 2600);
}

function selectConversation(id) {
  selectedId.value = id;
  const target = conversations.value.find((item) => item.id === id);
  if (target) target.unread = 0;
  mobileView.value = "chat";
}

function updateConversation(updated) {
  const index = conversations.value.findIndex((item) => item.id === updated.id);
  if (index >= 0) conversations.value[index] = updated;
}

function handleNav(scope) {
  if (["inbox", "mine", "waiting", "ai"].includes(scope)) {
    navScope.value = scope;
    mobileView.value = "list";
    return;
  }
  overlay.value = scope;
}
</script>

<template>
  <div class="service-app" :class="`mobile-${mobileView}`">
    <AppSidebar
      :active="navScope"
      :counts="scopeCounts"
      :agent="currentAgent"
      @select="handleNav"
    />

    <ConversationList
      :conversations="conversations"
      :selected-id="selectedId"
      :scope="navScope"
      :agent-name="currentAgent.name"
      @select="selectConversation"
      @notify="notify"
    />

    <ChatWorkspace
      v-if="selectedConversation"
      :conversation="selectedConversation"
      :agent="currentAgent"
      @update="updateConversation"
      @back="mobileView = 'list'"
      @notify="notify"
    />

    <section v-else class="empty-chat">
      <div class="empty-chat-mark">CS</div>
      <h1>当前队列暂无会话</h1>
      <p>新的客户咨询会自动出现在会话列表中。</p>
    </section>

    <Transition name="toast">
      <div v-if="toast" class="global-toast" role="status">{{ toast }}</div>
    </Transition>

    <div v-if="overlay" class="overlay-backdrop" @click.self="overlay = ''">
      <section class="quick-panel" role="dialog" aria-modal="true" :aria-label="overlay === 'knowledge' ? '知识库' : '服务数据'">
        <header>
          <div>
            <span>{{ overlay === "knowledge" ? "KNOWLEDGE BASE" : "SERVICE INSIGHTS" }}</span>
            <h2>{{ overlay === "knowledge" ? "知识库" : "今日服务数据" }}</h2>
          </div>
          <button class="icon-control" type="button" title="关闭" aria-label="关闭" @click="overlay = ''"><X :size="18" /></button>
        </header>

        <template v-if="overlay === 'knowledge'">
          <label class="panel-search">
            <span class="sr-only">搜索知识库</span>
            <input type="search" placeholder="搜索产品、订单或售后政策" />
          </label>
          <button v-for="article in quickArticles" :key="article.title" class="article-row" type="button" @click="notify(`已打开：${article.title}`)">
            <BookOpen :size="17" />
            <span><strong>{{ article.title }}</strong><small>{{ article.category }} · {{ article.updated }}</small></span>
          </button>
        </template>

        <div v-else class="metric-list">
          <div><span>首次响应</span><strong>42秒</strong><small>较昨日快 8 秒</small></div>
          <div><span>接待会话</span><strong>128</strong><small>当前在线 14</small></div>
          <div><span>一次解决率</span><strong>86.4%</strong><small>团队目标 85%</small></div>
          <div><span>满意度</span><strong>4.92</strong><small>基于 96 份评价</small></div>
          <BarChart3 class="metric-watermark" :size="120" aria-hidden="true" />
        </div>
      </section>
    </div>
  </div>
</template>
