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
  intro: '为 RTX 3080 与 4090 构建的 MiniMax H3 引擎。支持 Turbo LoRA，用 Python 或 HTTP 接入你的应用。',
  start: '开始使用',
  support: '查看支持范围',
  choose: '选择原生去噪配置',
  memory: '显存容量',
  execution: '执行方式',
  available: '可用配置',
  inspect: '检查硬件，不加载权重',
  copy: '复制', copied: '已复制', copyFailed: '请手动选择命令复制',
  gpu: [
    { execution: '单卡 · 权重常驻', profiles: 'Turbo4 / Turbo8', note: '在连续请求中复用权重，也可分块加载，为较大输入留出显存。' },
    { execution: '单卡 · 分块加载', profiles: 'Turbo4', note: '权重从系统内存传入显卡。已测负载需预留至少 64 GiB 可用 RAM。' },
    { execution: '双卡协作 · 分块加载', profiles: 'Turbo4', note: '两张卡共同处理一个请求；需要显式选择第二张卡，并预留系统内存。' },
  ],
  boundary: '当前公开版的输入与输出',
  boundaryText: '单 4090 支持文字、图片或一段短视频参考，生成五秒 MP4；双 3080 支持多参考图完整生成。提供 Python、容器命令和官方权重编译器；HTTP 服务使用条件包到 latent 的接口。',
  inputs: '了解运行前提',
  next: '按你的任务开始',
  guides: [
    { number: '01', title: '检查环境与配置', text: '先安装轻量 CLI，确认显卡、内存和模型配置。无需下载权重。', path: '/guide/getting-started' },
    { number: '02', title: '生成一个视频', text: '准备官方模型资源，通过 Python 或容器命令，从文字、图片或短视频参考生成 MP4。', path: '/guide/complete-pipeline' },
    { number: '03', title: '评估速度与质量', text: '区分冷启动、重复请求和成片耗时，按真实任务检查结果。', path: '/reference/performance' },
  ],
  source: '理解实现，按需扩展。',
  sourceText: 'PyTorch 与 Triton 原生执行，独立实现 LoRA 计算与显存调度。',
  architecture: '阅读架构',
} : {
  eyebrow: 'Stable release',
  title: 'Native H3 inference.',
  accent: 'Built for your GPU.',
  intro: 'A MiniMax H3 engine for RTX 3080 and 4090, with Turbo LoRA support. Bring it into your application through Python or HTTP.',
  start: 'Get started',
  support: 'Supported profiles',
  choose: 'Choose a denoising configuration',
  memory: 'VRAM capacity',
  execution: 'Execution',
  available: 'Available profiles',
  inspect: 'Inspect hardware without loading weights',
  copy: 'Copy', copied: 'Copied', copyFailed: 'Select the command to copy it',
  gpu: [
    { execution: 'One GPU · resident', profiles: 'Turbo4 / Turbo8', note: 'Reuse weights across requests, or stream blocks to leave VRAM for larger inputs.' },
    { execution: 'One GPU · streamed', profiles: 'Turbo4', note: 'Weights stream from system memory. Allow 64 GiB+ available RAM for the tested workload.' },
    { execution: 'Two GPUs · streamed', profiles: 'Turbo4', note: 'Both GPUs cooperate on one request. Select the peer explicitly and allow sufficient host RAM.' },
  ],
  boundary: 'The current public interface',
  boundaryText: 'Generate a five-second MP4 from text, images or one short reference video on a 4090. A pair of 3080s also supports complete multi-reference generation through Python or Docker. The HTTP service uses conditioning bundles and returns latents.',
  inputs: 'Check the prerequisites',
  next: 'Start with what you need',
  guides: [
    { number: '01', title: 'Check your setup', text: 'Install the lightweight CLI and inspect your GPU, memory and profiles. No weights required.', path: '/guide/getting-started' },
    { number: '02', title: 'Generate a video', text: 'Prepare official model assets and generate an MP4 from text, images or a short clip with Python or Docker.', path: '/guide/complete-pipeline' },
    { number: '03', title: 'Evaluate the results', text: 'Separate loading, repeated requests and video delivery. Check quality against your task.', path: '/reference/performance' },
  ],
  source: 'Understand it. Build on it.',
  sourceText: 'Native PyTorch and Triton execution, with its own LoRA computation and memory scheduling.',
  architecture: 'Explore the architecture',
})
const profile = computed(() => `ref2va-turbo4-exact-sm${selected.value === 0 ? '89' : '86'}`)
const command = computed(() => `vflash plan ${profile.value} \\\n  --gpu 0${selected.value === 2 ? ' --peer-gpu 1 \\\n  --strategy sequence-head' : ''}`)

function select(index: number) {
  selected.value = index
  copied.value = false
  copyFailed.value = false
}

function selectWithKeyboard(event: KeyboardEvent) {
  if (!['ArrowLeft', 'ArrowRight', 'Home', 'End'].includes(event.key)) return
  event.preventDefault()
  const next = event.key === 'Home' ? 0 : event.key === 'End' ? 2
    : (selected.value + (event.key === 'ArrowRight' ? 1 : 2)) % 3
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
        <p class="vf-meta"><span>SM86 / SM89</span><span>Turbo4 / Turbo8</span><span>Apache 2.0</span></p>
      </div>

      <div class="vf-gpu-card">
        <div class="vf-card-heading"><span>{{ text.choose }}</span><span>RTX / CUDA</span></div>
        <div class="vf-card-body">
          <div class="vf-tabs" role="tablist" :aria-label="text.choose" @keydown="selectWithKeyboard">
            <button v-for="(name, index) in ['RTX 4090', 'RTX 3080', '2 × RTX 3080']" :id="`gpu-tab-${index}`" :key="name"
              :ref="element => { if (element) tabs[index] = element as HTMLButtonElement }"
              type="button" role="tab" :aria-selected="selected === index" aria-controls="gpu-panel"
              :tabindex="selected === index ? 0 : -1" @click="select(index)">{{ name }}</button>
          </div>
          <div id="gpu-panel" role="tabpanel" :aria-labelledby="`gpu-tab-${selected}`" tabindex="0">
            <div class="vf-gpu-name">{{ ['48', '20', '2 × 20'][selected] }}<span>GB</span><small>{{ text.memory }}</small></div>
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
