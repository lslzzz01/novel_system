<script setup>
import {
  BarChart3,
  Bell,
  BookOpen,
  Bot,
  Headphones,
  Inbox,
  LogOut,
  MessageSquareText,
  Settings,
  UsersRound,
} from "lucide-vue-next";

defineProps({
  active: { type: String, required: true },
  counts: { type: Object, required: true },
  agent: { type: Object, required: true },
});

defineEmits(["select"]);

const primaryItems = [
  { id: "inbox", label: "会话中心", icon: Inbox },
  { id: "mine", label: "我的会话", icon: MessageSquareText },
  { id: "waiting", label: "待接入", icon: Headphones },
  { id: "ai", label: "AI 接待", icon: Bot },
];

const manageItems = [
  { id: "customers", label: "客户管理", icon: UsersRound },
  { id: "knowledge", label: "知识库", icon: BookOpen },
  { id: "analytics", label: "数据看板", icon: BarChart3 },
];
</script>

<template>
  <aside class="app-sidebar">
    <div class="brand-lockup" aria-label="智服云">
      <div class="brand-symbol"><span></span><span></span><span></span></div>
      <div><strong>智服云</strong><small>智能客服工作台</small></div>
    </div>

    <nav class="side-nav" aria-label="客服工作区">
      <span class="nav-section-label">工作区</span>
      <button
        v-for="item in primaryItems"
        :key="item.id"
        class="side-nav-item"
        :class="{ active: active === item.id }"
        type="button"
        :title="item.label"
        @click="$emit('select', item.id)"
      >
        <component :is="item.icon" :size="19" :stroke-width="1.9" />
        <span>{{ item.label }}</span>
        <em v-if="counts[item.id]">{{ counts[item.id] }}</em>
      </button>

      <span class="nav-section-label manage-label">管理</span>
      <button
        v-for="item in manageItems"
        :key="item.id"
        class="side-nav-item"
        type="button"
        :title="item.label"
        @click="$emit('select', item.id)"
      >
        <component :is="item.icon" :size="19" :stroke-width="1.9" />
        <span>{{ item.label }}</span>
      </button>
    </nav>

    <div class="sidebar-bottom">
      <div class="agent-state">
        <div class="agent-avatar">{{ agent.avatar }}</div>
        <div><strong>{{ agent.name }}</strong><small><i></i> 在线接待</small></div>
        <button class="icon-plain" type="button" title="通知" aria-label="通知"><Bell :size="17" /><b>3</b></button>
      </div>
      <div class="bottom-actions">
        <button type="button" title="设置"><Settings :size="17" /><span>系统设置</span></button>
        <button type="button" title="退出"><LogOut :size="17" /><span>退出</span></button>
      </div>
    </div>
  </aside>
</template>
