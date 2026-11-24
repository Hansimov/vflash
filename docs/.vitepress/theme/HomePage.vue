<script setup lang="ts">
import { computed, ref } from 'vue'
import { useData, withBase } from 'vitepress'

const { lang, theme } = useData()
const zh = computed(() => lang.value.startsWith('zh'))
const tr = (en: string, cn: string) => zh.value ? cn : en
const href = (path: string) => withBase((zh.value ? '/zh/' : '/') + path)
const hardware = [
  { name: 'RTX PRO 6000 Server', fp32: 120, memory: '96 GiB', sm: 'SM120', command: 'vflash doctor',
    en: 'Prepared Pika video with joint or Larry audio. Remake uses a separate model environment.',
    cn: '可运行准备权重的 Pika 音画推理。编辑源片时使用独立的 Remake 环境。' },
  { name: 'RTX 4090', fp32: 82.6, memory: '48 GiB', sm: 'SM89', command: 'vflash plan ref2va-turbo4-exact-sm89 --gpu 0',
    en: 'Fixed-profile generation from text, images or keyframes. Published runs used a 48 GiB board.',
    cn: '可从文字、参考图或关键帧生成视频。公开实测使用 48 GiB 显存配置。' },
  { name: 'H100 SXM', fp32: 67, memory: '80 GiB', sm: 'SM90', command: 'vflash plan i2va-turbo4-v01-544-exact-sm90 --gpu 0',
    en: 'First-frame generation with a resident hybrid trunk. Reserve at least 75 GiB for the resident path.',
    cn: '可用常驻显存的混合主干进行首帧生成。该常驻路径需要至少 75 GiB 显存。' },
  { name: '2 × RTX 3080', fp32: 29.8, memory: '2 × 20 GiB', sm: 'SM86', command: 'vflash plan ref2va-turbo4-exact-sm86 --gpu 0 --peer-gpu 1 --strategy sequence-head',
    en: 'Two measured 20 GiB cards share one request. Leave host RAM available for streamed weights.',
    cn: '实测的两张 20 GiB 卡共同处理一个请求，权重按需从主机内存加载。' },
].sort((a, b) => b.fp32 - a.fp32)
const tasks = [
  ['Generate video with sound', '生成有声视频', 'Choose a prompt, a first frame or appearance references.', '从文字、首帧或参考图开始。', 'guide/runtime-selection'],
  ['Edit existing footage', '编辑已有视频', 'Prepare a source window and bind the appearance to replace.', '准备源片窗口，绑定要替换的外观。', 'guide/remake'],
  ['Integrate inference', '接入应用', 'Reuse a Python session or serve compiled bundles over HTTP.', '复用 Python 会话，或通过 HTTP 提交条件张量。', 'guide/python'],
]
const selected = ref(0)
const current = computed(() => hardware[selected.value]!)
const tabs = ref<HTMLButtonElement[]>([])
function navigate(event: KeyboardEvent) {
  const delta = event.key === 'ArrowLeft' ? -1 : event.key === 'ArrowRight' ? 1 : 0
  if (!delta && !['Home', 'End'].includes(event.key)) return
  event.preventDefault()
  selected.value = event.key === 'Home' ? 0 : event.key === 'End' ? hardware.length - 1 : (selected.value + delta + hardware.length) % hardware.length
  tabs.value[selected.value]?.focus()
}
</script>

<template>
  <main class="vf-home">
    <section class="vf-hero" aria-labelledby="vf-title">
      <div class="vf-intro">
        <a class="vf-release" :href="href('reference/releases')">Vflash {{ theme.version }}</a>
        <h1 id="vf-title">{{ tr('Video inference with MiniMax H3', 'MiniMax H3 视频推理引擎') }}</h1>
        <p class="vf-description">{{ tr('Generate a film from a prompt or reference images, or edit a shot you already have. Vflash runs the model, manages GPU memory and delivers the picture and soundtrack.', '从文字或参考图生成带音轨的视频，也可以编辑已有镜头。Vflash 负责模型推理、显存调度和音画交付。') }}</p>
        <div class="vf-actions"><a class="vf-button" :href="href('guide/getting-started')">{{ tr('Install Vflash', '安装 Vflash') }} →</a>
          <a class="vf-text-link" :href="href('reference/architecture')">{{ tr('Inside the engine', '了解引擎架构') }}</a></div>
      </div>
      <div class="vf-gpu-card">
        <div class="vf-card-heading"><span>{{ tr('Choose your GPU', '按显卡选择') }}</span></div>
        <div class="vf-card-body">
          <p class="vf-gpu-note">{{ tr('Ordered by reference single-GPU peak FP32 throughput.', '按参考型号的单卡 FP32 峰值算力降序排列') }}</p>
          <div class="vf-tabs" role="tablist" :aria-label="tr('GPU models','显卡型号')" @keydown="navigate">
            <button v-for="(gpu, i) in hardware" :id="'gpu-tab-'+i" :key="gpu.name" :ref="el => { if(el) tabs[i] = el as HTMLButtonElement }" role="tab" aria-controls="gpu-panel" :aria-selected="i === selected" :tabindex="i===selected ? 0 : -1" @click="selected=i">{{ gpu.name }}</button>
          </div>
          <div id="gpu-panel" role="tabpanel" :aria-labelledby="'gpu-tab-'+selected">
            <div class="vf-gpu-name">{{ current.fp32 }} <span>TFLOPS</span></div>
            <p>{{ current.sm }} · {{ tr('Tested memory', '实测显存') }} {{ current.memory }}</p>
            <p class="vf-gpu-note">{{ tr(current.en,current.cn) }}</p>
            <div class="vf-command"><code>{{ current.command }}</code></div>
            <a class="vf-text-link" :href="href('guide/profiles')">{{ tr('Hardware limits and specifications', '硬件限制与规格来源') }} →</a>
          </div>
        </div>
      </div>
    </section>
    <div class="vf-guide-grid">
      <a v-for="task in tasks" :key="task[4]" class="vf-guide" :href="href(task[4]!)">
        <h2>{{ tr(task[0]!,task[1]!) }}</h2><p>{{ tr(task[2]!,task[3]!) }}</p>
      </a>
    </div>
  </main>
</template>
