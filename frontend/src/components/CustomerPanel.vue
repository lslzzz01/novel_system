<script setup>
import { computed, ref } from "vue";
import {
  ChevronDown,
  Copy,
  ExternalLink,
  Mail,
  MapPin,
  Package,
  Phone,
  Plus,
  ShoppingBag,
  UserRound,
  X,
} from "lucide-vue-next";

const props = defineProps({
  conversation: { type: Object, required: true },
  open: { type: Boolean, default: true },
});

const emit = defineEmits(["close", "notify", "add-note"]);
const note = ref("");
const customer = computed(() => props.conversation.customer);
const order = computed(() => props.conversation.order);

function saveNote() {
  const value = note.value.trim();
  if (!value) return;
  emit("add-note", value);
  note.value = "";
}
</script>

<template>
  <aside class="customer-panel" :class="{ open }">
    <header class="customer-panel-header">
      <strong>客户信息</strong>
      <button class="icon-control" type="button" title="收起客户信息" aria-label="收起客户信息" @click="$emit('close')"><X :size="17" /></button>
    </header>

    <div class="customer-panel-scroll">
      <section class="customer-profile">
        <div class="profile-avatar" :style="{ '--avatar-color': customer.color }">{{ customer.avatar }}</div>
        <div><h2>{{ customer.name }}</h2><p><i :class="customer.online ? 'online' : ''"></i>{{ customer.online ? "当前在线" : "离线" }} · {{ customer.level }}</p></div>
        <button class="icon-control" type="button" title="更多客户操作" aria-label="更多客户操作"><ChevronDown :size="17" /></button>
      </section>

      <section class="detail-section">
        <header><span>基本资料</span><button type="button" @click="$emit('notify', '客户资料已进入编辑状态')">编辑</button></header>
        <dl>
          <div><dt><Phone :size="15" />手机号</dt><dd>{{ customer.phone }}</dd></div>
          <div><dt><Mail :size="15" />邮箱</dt><dd>{{ customer.email }}</dd></div>
          <div><dt><MapPin :size="15" />地区</dt><dd>{{ customer.location }}</dd></div>
          <div><dt><UserRound :size="15" />客户来源</dt><dd>{{ customer.source }}</dd></div>
        </dl>
      </section>

      <section class="detail-section">
        <header><span>客户标签</span><button type="button" title="添加标签" aria-label="添加标签" @click="$emit('notify', '标签选择器已打开')"><Plus :size="15" /></button></header>
        <div class="customer-tags"><span v-for="tag in customer.tags" :key="tag">{{ tag }}</span></div>
      </section>

      <section v-if="order" class="detail-section order-section">
        <header><span>关联订单</span><button type="button" title="查看全部订单" aria-label="查看全部订单" @click="$emit('notify', '已打开客户订单列表')"><ExternalLink :size="15" /></button></header>
        <div class="order-card">
          <div class="order-top"><Package :size="18" /><span><strong>{{ order.product }}</strong><small>{{ order.id }}</small></span></div>
          <div class="order-facts"><span><small>订单金额</small><strong>¥{{ order.amount }}</strong></span><span><small>订单状态</small><strong>{{ order.status }}</strong></span></div>
          <button type="button" @click="$emit('notify', `已复制订单号 ${order.id}`)"><Copy :size="14" />复制订单号</button>
        </div>
      </section>

      <section class="detail-section customer-stats">
        <header><span>服务记录</span></header>
        <div><span><ShoppingBag :size="17" />累计消费<strong>¥{{ customer.totalSpend }}</strong></span><span>历史会话<strong>{{ customer.conversationCount }} 次</strong></span></div>
      </section>

      <section class="detail-section note-section">
        <header><span>客户备注</span></header>
        <form @submit.prevent="saveNote">
          <textarea v-model="note" rows="3" placeholder="记录仅团队可见的信息"></textarea>
          <button type="submit" :disabled="!note.trim()">添加备注</button>
        </form>
      </section>
    </div>
  </aside>
</template>
