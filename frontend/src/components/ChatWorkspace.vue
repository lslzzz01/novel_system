<script setup>
import { computed, nextTick, ref, watch } from "vue";
import {
  ArrowLeft,
  Bot,
  BookOpen,
  Check,
  ChevronDown,
  Clock3,
  FileText,
  Image,
  Info,
  MessageSquareText,
  MoreHorizontal,
  Paperclip,
  Phone,
  Send,
  Smile,
  Sparkles,
  UserRoundCheck,
} from "lucide-vue-next";
import CustomerPanel from "./CustomerPanel.vue";

const props = defineProps({
  conversation: { type: Object, required: true },
  agent: { type: Object, required: true },
});

const emit = defineEmits(["update", "back", "notify"]);
const draft = ref("");
const replyMode = ref("reply");
const customerPanelOpen = ref(true);
const messageScroll = ref(null);
const aiPanelOpen = ref(true);

const customer = computed(() => props.conversation.customer);
const isResolved = computed(() => props.conversation.status === "resolved");
const isAssigned = computed(() => props.conversation.assignee === props.agent.name);
const quickReplies = computed(() => props.conversation.quickReplies || ["我来帮您查询", "请稍等片刻", "问题已为您处理"]);

watch(() => props.conversation.id, async () => {
  draft.value = "";
  replyMode.value = "reply";
  aiPanelOpen.value = true;
  await scrollToBottom(false);
}, { immediate: true });

function mutate(mutator) {
  const next = structuredClone(props.conversation);
  mutator(next);
  emit("update", next);
}

async function scrollToBottom(smooth = true) {
  await nextTick();
  if (messageScroll.value) messageScroll.value.scrollTo({ top: messageScroll.value.scrollHeight, behavior: smooth ? "smooth" : "auto" });
}

function sendMessage() {
  const content = draft.value.trim();
  if (!content || isResolved.value) return;
  const now = Date.now();
  mutate((next) => {
    next.status = "active";
    next.assignee = props.agent.name;
    next.updatedAt = now;
    next.preview = replyMode.value === "note" ? "[内部备注] 已添加一条团队备注" : content;
    next.messages.push({
      id: `m-${now}`,
      type: replyMode.value === "note" ? "note" : "outgoing",
      sender: replyMode.value === "note" ? "内部备注" : props.agent.name,
      content,
      time: new Date(now).toLocaleTimeString("zh-CN", { hour: "2-digit", minute: "2-digit" }),
    });
  });
  draft.value = "";
  emit("notify", replyMode.value === "note" ? "内部备注已添加" : "消息已发送");
  scrollToBottom();
}

function insertReply(text) {
  draft.value = text;
  nextTick(() => document.querySelector(".composer-textarea")?.focus());
}

function assignToMe() {
  mutate((next) => {
    next.assignee = props.agent.name;
    next.status = "active";
    next.messages.push({ id: `system-${Date.now()}`, type: "system", content: `${props.agent.name} 已接入会话`, time: "刚刚" });
  });
  emit("notify", "会话已分配给你");
  scrollToBottom();
}

function toggleResolve() {
  mutate((next) => {
    next.status = next.status === "resolved" ? "active" : "resolved";
    next.messages.push({
      id: `system-${Date.now()}`,
      type: "system",
      content: next.status === "resolved" ? "会话已结束，等待客户评价" : "会话已重新开启",
      time: "刚刚",
    });
  });
  emit("notify", isResolved.value ? "会话已重新开启" : "会话已结束");
  scrollToBottom();
}

function addCustomerNote(content) {
  mutate((next) => {
    next.customer.notes = [...(next.customer.notes || []), content];
  });
  emit("notify", "客户备注已保存");
}

function handleComposerKeydown(event) {
  if ((event.ctrlKey || event.metaKey) && event.key === "Enter") {
    event.preventDefault();
    sendMessage();
  }
}
</script>

<template>
  <main class="chat-workspace">
    <section class="chat-main">
      <header class="chat-header">
        <button class="icon-control mobile-back" type="button" title="返回会话列表" aria-label="返回会话列表" @click="$emit('back')"><ArrowLeft :size="18" /></button>
        <div class="chat-avatar" :style="{ '--avatar-color': customer.color }">{{ customer.avatar }}<i :class="customer.online ? 'online' : ''"></i></div>
        <div class="chat-identity">
          <div><h2>{{ customer.name }}</h2><span v-if="conversation.priority === 'urgent'">紧急</span></div>
          <p>{{ conversation.channel }} · {{ customer.device }} · <em>{{ customer.online ? "在线" : "离线" }}</em></p>
        </div>
        <div class="chat-actions">
          <button v-if="!isAssigned && !isResolved" class="command-button primary" type="button" @click="assignToMe"><UserRoundCheck :size="16" />接入会话</button>
          <button v-else class="command-button" type="button" @click="$emit('notify', '转接坐席列表已打开')"><UserRoundCheck :size="16" />{{ conversation.assignee || "待分配" }}<ChevronDown :size="14" /></button>
          <button class="icon-control" type="button" title="发起语音通话" aria-label="发起语音通话" @click="$emit('notify', '正在发起语音通话')"><Phone :size="17" /></button>
          <button class="icon-control" type="button" title="客户信息" aria-label="客户信息" :class="{ active: customerPanelOpen }" @click="customerPanelOpen = !customerPanelOpen"><Info :size="18" /></button>
          <button class="icon-control" type="button" title="更多操作" aria-label="更多操作"><MoreHorizontal :size="19" /></button>
          <button class="resolve-button" type="button" :class="{ resolved: isResolved }" @click="toggleResolve"><Check :size="16" />{{ isResolved ? "重新开启" : "结束会话" }}</button>
        </div>
      </header>

      <div class="service-status-bar">
        <div><i :class="conversation.status"></i><strong>{{ conversation.status === "queued" ? "客户等待接入" : conversation.status === "resolved" ? "会话已结束" : "服务进行中" }}</strong></div>
        <span><Clock3 :size="14" />已响应 {{ conversation.responseTime }}</span>
        <span>接待坐席 {{ conversation.assignee || "未分配" }}</span>
        <button type="button" @click="aiPanelOpen = !aiPanelOpen"><Sparkles :size="14" />{{ aiPanelOpen ? "收起 AI 摘要" : "查看 AI 摘要" }}</button>
      </div>

      <div ref="messageScroll" class="message-scroll">
        <div class="message-inner">
          <div class="day-divider"><span>今天</span></div>

          <Transition name="summary">
            <section v-if="aiPanelOpen" class="ai-summary">
              <div class="ai-summary-icon"><Sparkles :size="17" /></div>
              <div><strong>AI 会话摘要</strong><p>{{ conversation.summary }}</p></div>
              <button type="button" title="复制摘要" aria-label="复制摘要" @click="$emit('notify', '会话摘要已复制')"><FileText :size="15" /></button>
            </section>
          </Transition>

          <template v-for="message in conversation.messages" :key="message.id">
            <div v-if="message.type === 'system'" class="system-message"><span>{{ message.content }}</span><time>{{ message.time }}</time></div>
            <div v-else-if="message.type === 'note'" class="internal-note"><MessageSquareText :size="15" /><span><strong>{{ message.sender }}</strong>{{ message.content }}</span><time>{{ message.time }}</time></div>
            <div v-else class="message-row" :class="message.type">
              <div v-if="message.type === 'incoming'" class="message-avatar" :style="{ '--avatar-color': customer.color }">{{ customer.avatar }}</div>
              <div class="message-stack">
                <div class="message-sender"><span>{{ message.type === 'outgoing' ? message.sender : customer.name }}</span><time>{{ message.time }}</time></div>
                <div class="message-bubble">
                  <p>{{ message.content }}</p>
                  <div v-if="message.attachment" class="message-attachment"><FileText :size="17" /><span><strong>{{ message.attachment.name }}</strong><small>{{ message.attachment.size }}</small></span></div>
                </div>
                <small v-if="message.type === 'outgoing'" class="delivery"><Check :size="12" />已送达</small>
              </div>
              <div v-if="message.type === 'outgoing'" class="message-avatar agent">{{ agent.avatar }}</div>
            </div>
          </template>

          <div v-if="isResolved" class="resolved-banner"><Check :size="17" /><span><strong>本次会话已结束</strong>客户仍可在 24 小时内回复并重新开启。</span></div>
        </div>
      </div>

      <footer class="composer" :class="{ disabled: isResolved }">
        <div class="composer-topline">
          <div class="reply-tabs">
            <button type="button" :class="{ active: replyMode === 'reply' }" @click="replyMode = 'reply'">回复客户</button>
            <button type="button" :class="{ active: replyMode === 'note' }" @click="replyMode = 'note'">内部备注</button>
          </div>
          <button type="button" @click="$emit('notify', '快捷回复管理已打开')"><BookOpen :size="15" />快捷回复</button>
        </div>

        <div v-if="!isResolved" class="quick-replies">
          <button v-for="reply in quickReplies" :key="reply" type="button" @click="insertReply(reply)">{{ reply }}</button>
        </div>

        <div class="composer-box">
          <textarea
            v-model="draft"
            class="composer-textarea"
            :disabled="isResolved"
            :placeholder="isResolved ? '重新开启会话后即可回复' : replyMode === 'note' ? '添加内部备注，仅团队成员可见' : `回复 ${customer.name}`"
            rows="3"
            @keydown="handleComposerKeydown"
          ></textarea>
          <div class="composer-actions">
            <div>
              <button type="button" title="添加表情" aria-label="添加表情"><Smile :size="18" /></button>
              <button type="button" title="上传附件" aria-label="上传附件" @click="$emit('notify', '文件选择器已打开')"><Paperclip :size="18" /></button>
              <button type="button" title="发送图片" aria-label="发送图片" @click="$emit('notify', '图片选择器已打开')"><Image :size="18" /></button>
              <button class="ai-write" type="button" title="AI 辅助回复" aria-label="AI 辅助回复" @click="insertReply(conversation.aiSuggestion)"><Sparkles :size="17" /><span>AI 辅助</span></button>
            </div>
            <button class="send-button" type="button" :disabled="!draft.trim() || isResolved" @click="sendMessage"><Send :size="16" />发送</button>
          </div>
        </div>
      </footer>
    </section>

    <CustomerPanel
      :conversation="conversation"
      :open="customerPanelOpen"
      @close="customerPanelOpen = false"
      @notify="$emit('notify', $event)"
      @add-note="addCustomerNote"
    />
    <button v-if="!customerPanelOpen" class="customer-panel-restore" type="button" title="展开客户信息" aria-label="展开客户信息" @click="customerPanelOpen = true"><Info :size="18" /></button>
  </main>
</template>
