<script setup lang="ts">
import { computed, onBeforeUnmount, ref } from 'vue'
import { useData, withBase } from 'vitepress'

const { lang, theme } = useData()
const zh = computed(() => lang.value.startsWith('zh'))
const selected = ref(0)
const tabs = ref<HTMLButtonElement[]>([])
const copied = ref(false)
const copyFailed = ref(false)
let copyTimer: ReturnType<typeof setTimeout> | undefined
const link = (path: string) => withBase(`${zh.value ? '/zh' : ''}${path}`)
const text = computed(() => zh.value ? {
  eyebrow: '正式版',
  title: '原生 H3 推理。',
  accent: '充分发挥你的显卡。',
  intro: '为 SM86 与 SM89 实测优化的 MiniMax H3 引擎。支持 Turbo 与 Base16 关键帧生成，可通过 Python、容器或 HTTP 接入。',
  start: '开始使用',
  support: '查看支持范围',
  choose: '选择原生去噪配置',
  memory: '显存容量',
  execution: '执行方式',
  available: '可用配置',
  inspect: '检查硬件，不加载权重',
  copy: '复制', copied: '已复制', copyFailed: '请手动选择命令复制',
  gpu: [
    { execution: '单卡 · 常驻或分块', profiles: 'Turbo4 / Turbo8 / Base16', note: '支持文字、参考素材和首尾帧任务；完整 pipeline 按阶段复用同一张卡。' },
    { execution: '单卡 · 分块加载', profiles: 'Turbo4', note: '权重从系统内存传入显卡。已测负载需预留至少 64 GiB 可用 RAM。' },
    { execution: '双卡协作 · 分块加载', profiles: 'Ref4 / T2VA / Base16', note: '两张卡共同处理一个请求；使用 sequence-head，需显式选择第二张卡并预留系统内存。' },
    { execution: '双卡协作 · 分块加载', profiles: 'Base16', note: '匹配的两张 4090 只在 peer 原本空闲时用于降低单请求延迟；有两个请求时仍使用独立 worker。' },
  ],
  boundary: '当前公开版的输入与输出',
  boundaryText: 'Turbo profile 保持五秒合同；Base16 I2VA、L2VA 与 FL2VA 支持五至十秒。0.4.0 为 SM86/SM89 双卡加入精确直接重排，并把近似 attention、cache 与量化路径明确留在独立实验范围。',
  inputs: '了解运行前提',
  next: '按你的任务开始',
  guides: [
    { number: '01', title: '检查环境与配置', text: '先安装轻量 CLI，确认显卡、内存和模型配置。无需下载权重。', path: '/guide/getting-started' },
    { number: '02', title: '生成一个视频', text: '准备官方模型资源，通过 Python 或容器命令，从文字、图片或短视频参考生成 MP4。', path: '/guide/complete-pipeline' },
    { number: '03', title: '评估速度与质量', text: '区分冷启动、重复请求和成片耗时，按真实任务检查结果。', path: '/reference/performance' },
  ],
  source: '理解实现，按需扩展。',
  sourceText: 'PyTorch 与 Triton 原生执行，独立实现 LoRA、显存调度和双卡 collective 布局。',
  architecture: '阅读架构',
} : {
  eyebrow: 'Stable release',
  title: 'Native H3 inference.',
  accent: 'Built for your GPU.',
  intro: 'A MiniMax H3 engine measured on SM86 and SM89. Run Turbo or Base16 keyframe generation through Python, containers or HTTP.',
  start: 'Get started',
  support: 'Supported profiles',
  choose: 'Choose a denoising configuration',
  memory: 'VRAM capacity',
  execution: 'Execution',
  available: 'Available profiles',
  inspect: 'Inspect hardware without loading weights',
  copy: 'Copy', copied: 'Copied', copyFailed: 'Select the command to copy it',
  gpu: [
    { execution: 'One GPU · resident/streamed', profiles: 'Turbo4 / Turbo8 / Base16', note: 'Serve text, references and keyframes; the complete pipeline reuses one GPU across stages.' },
    { execution: 'One GPU · streamed', profiles: 'Turbo4', note: 'Weights stream from system memory. Allow 64 GiB+ available RAM for the tested workload.' },
    { execution: 'Two GPUs · streamed', profiles: 'Ref4 / T2VA / Base16', note: 'Both GPUs cooperate on one request with sequence-head. Select the peer explicitly and allow sufficient host RAM.' },
    { execution: 'Two GPUs · streamed', profiles: 'Base16', note: 'Use matching 4090s to reduce one request\'s latency only when the peer would otherwise be idle; independent workers retain higher two-request throughput.' },
  ],
  boundary: 'The current public interface',
  boundaryText: 'Turbo profiles retain a five-second contract. Base16 I2VA, L2VA and FL2VA accept five through ten seconds. Version 0.4.0 adds exact direct two-GPU relayouts on SM86/SM89 while keeping approximate attention, caches and quantization in separate research scope.',
  inputs: 'Check the prerequisites',
  next: 'Start with what you need',
  guides: [
    { number: '01', title: 'Check your setup', text: 'Install the lightweight CLI and inspect your GPU, memory and profiles. No weights required.', path: '/guide/getting-started' },
    { number: '02', title: 'Generate a video', text: 'Prepare official model assets and generate an MP4 from text, images or a short clip with Python or Docker.', path: '/guide/complete-pipeline' },
    { number: '03', title: 'Evaluate the results', text: 'Separate loading, repeated requests and video delivery. Check quality against your task.', path: '/reference/performance' },
  ],
  source: 'Understand it. Build on it.',
  sourceText: 'Native PyTorch and Triton execution, with owned LoRA, memory scheduling and two-GPU collective layouts.',
  architecture: 'Explore the architecture',
})
const gpuNames = ['RTX 4090', 'RTX 3080', '2 × RTX 3080', '2 × RTX 4090']
const gpuMemory = ['48', '20', '2 × 20', '2 × 48']
const profiles = [
  'ref2va-turbo4-exact-sm89',
  'ref2va-turbo4-exact-sm86',
  'ref2va-turbo4-exact-sm86',
  'i2va-base16-bf16-sm89',
]
const profile = computed(() => profiles[selected.value])
const command = computed(() => `vflash plan ${profile.value} \\\n  --gpu 0${selected.value >= 2 ? ' --peer-gpu 1 \\\n  --strategy sequence-head' : ''}`)

function select(index: number) {
  selected.value = index
  copied.value = false
  copyFailed.value = false
}

function selectWithKeyboard(event: KeyboardEvent) {
  if (!['ArrowLeft', 'ArrowRight', 'Home', 'End'].includes(event.key)) return
  event.preventDefault()
  const next = event.key === 'Home' ? 0 : event.key === 'End' ? gpuNames.length - 1
    : (selected.value + (event.key === 'ArrowRight' ? 1 : gpuNames.length - 1)) % gpuNames.length
  select(next)
  tabs.value[next]?.focus()
}

async function copyCommand() {
  try {
    await navigator.clipboard.writeText(command.value)
    copied.value = true
    copyFailed.value = false
    clearTimeout(copyTimer)
    copyTimer = setTimeout(() => { copied.value = false }, 2000)
  } catch {
    copyFailed.value = true
  }
}
onBeforeUnmount(() => clearTimeout(copyTimer))
</script>

<template>
  <div class="vf-home">
    <section class="vf-hero" aria-labelledby="vf-title">
      <div class="vf-intro">
        <p class="vf-eyebrow"><a class="vf-release" :href="link('/reference/releases')">{{ theme.version }}</a>{{ text.eyebrow }}</p>
        <h1 id="vf-title">{{ text.title }}<br><span>{{ text.accent }}</span></h1>
        <p class="vf-description">{{ text.intro }}</p>
        <div class="vf-actions">
          <a class="vf-button" :href="link('/guide/getting-started')">{{ text.start }} <span aria-hidden="true">→</span></a>
          <a class="vf-text-link" :href="link('/guide/profiles')">{{ text.support }} <span aria-hidden="true">↗</span></a>
        </div>
        <p class="vf-meta"><span>SM86 / SM89</span><span>Turbo4 / Turbo8 / Base16</span><span>Apache 2.0</span></p>
      </div>

      <div class="vf-gpu-card">
        <div class="vf-card-heading"><span>{{ text.choose }}</span><span>RTX / CUDA</span></div>
        <div class="vf-card-body">
          <div class="vf-tabs" role="tablist" :aria-label="text.choose" @keydown="selectWithKeyboard">
            <button v-for="(name, index) in gpuNames" :id="`gpu-tab-${index}`" :key="name"
              :ref="element => { if (element) tabs[index] = element as HTMLButtonElement }"
              type="button" role="tab" :aria-selected="selected === index" aria-controls="gpu-panel"
              :tabindex="selected === index ? 0 : -1" @click="select(index)">{{ name }}</button>
          </div>
          <div id="gpu-panel" role="tabpanel" :aria-labelledby="`gpu-tab-${selected}`" tabindex="0">
            <div class="vf-gpu-name">{{ gpuMemory[selected] }}<span>GB</span><small>{{ text.memory }}</small></div>
            <dl class="vf-specs">
              <div><dt>{{ text.execution }}</dt><dd>{{ text.gpu[selected].execution }}</dd></div>
              <div><dt>{{ text.available }}</dt><dd>{{ text.gpu[selected].profiles }}</dd></div>
            </dl>
            <p class="vf-gpu-note">{{ text.gpu[selected].note }}</p>
            <div class="vf-command">
              <div class="vf-command-label"><span>{{ text.inspect }}</span><span role="status"><button type="button" @click="copyCommand">{{ copied ? text.copied : text.copy }}</button></span></div>
              <code>{{ command }}</code>
              <p v-if="copyFailed" role="status" class="vf-gpu-note">{{ text.copyFailed }}</p>
            </div>
          </div>
        </div>
      </div>
    </section>

    <section class="vf-availability" aria-labelledby="vf-availability-title">
      <h2 id="vf-availability-title">{{ text.boundary }}</h2>
      <p>{{ text.boundaryText }} <a :href="link('/reference/runtime-assets')">{{ text.inputs }} <span aria-hidden="true">→</span></a></p>
    </section>

    <section class="vf-guides" aria-labelledby="vf-guides-title">
      <h2 id="vf-guides-title">{{ text.next }}</h2>
      <div class="vf-guide-grid">
        <a v-for="guide in text.guides" :key="guide.number" :href="link(guide.path)" class="vf-guide">
          <span class="vf-guide-top"><span>{{ guide.number }}</span><span aria-hidden="true">↗</span></span>
          <h3>{{ guide.title }}</h3>
          <p>{{ guide.text }}</p>
        </a>
      </div>
    </section>

    <section class="vf-source" aria-labelledby="vf-source-title">
      <div><h2 id="vf-source-title">{{ text.source }}</h2><p>{{ text.sourceText }}</p></div>
      <a class="vf-text-link" :href="link('/reference/architecture')">{{ text.architecture }} <span aria-hidden="true">→</span></a>
    </section>
  </div>
</template>
