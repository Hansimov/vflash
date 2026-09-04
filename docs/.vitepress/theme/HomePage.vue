<script setup lang="ts">
import { computed, ref } from 'vue'
import { useData, withBase } from 'vitepress'

const { lang } = useData()
const zh = computed(() => lang.value.startsWith('zh'))
const selected = ref(0)
const tabs = ref<HTMLButtonElement[]>([])
const link = (path: string) => withBase(`${zh.value ? '/zh' : ''}${path}`)
const text = computed(() => zh.value ? {
  eyebrow: '原生 MiniMax H3 推理引擎',
  title: 'H3 推理，',
  accent: '为你的显卡而优化。',
  intro: '按显存容量优化执行方式，支持 LightX2V Turbo LoRA，让 H3 去噪推理更高效。',
  start: '开始使用',
  support: '查看支持范围',
  preview: '开发者预览版',
  boundaryShort: '条件包 → 潜变量',
  choose: '选择你的显卡',
  memory: '显存',
  execution: '权重加载',
  available: '可用配置',
  inspect: '检查运行配置',
  gpu: [
    { execution: '常驻显存', profiles: 'Turbo4 · Turbo8', note: '模型加载后保留在显存中，供后续请求复用。' },
    { execution: '分块加载', profiles: 'Turbo4', note: '已测负载：建议每个 worker 预留 64 GiB 以上可用 RAM。' },
  ],
  boundary: '当前版本能做什么',
  boundaryText: '接收预编译的条件包，输出视频和音频潜变量（latents，即解码前的张量）。运行所需的资源包尚未公开；提示词处理和 MP4 输出还未提供。',
  inputs: '了解所需文件',
  next: '从这里开始',
  guides: [
    { number: '01', title: '安装与检查', text: '安装轻量命令行工具，确认显卡和配置是否匹配。', path: '/guide/getting-started' },
    { number: '02', title: '运行一次推理', text: '准备匹配的运行资源，执行条件包并保存输出。', path: '/guide/getting-started#run-a-bundle' },
    { number: '03', title: '接入你的服务', text: '使用 Docker 启动 HTTP 接口，提交任务并下载结果。', path: '/guide/docker' },
  ],
  source: '独立执行，开放源码。',
  sourceText: 'Vflash 使用 PyTorch 和 Triton 实现 H3 推理，无需安装 LightX2V 执行框架。',
  architecture: '了解实现方式',
} : {
  eyebrow: 'Native MiniMax H3 inference',
  title: 'H3 inference.',
  accent: 'Made for your GPU.',
  intro: 'Hardware-specific execution and support for LightX2V Turbo LoRAs, built to make H3 denoising faster.',
  start: 'Get started',
  support: 'See supported profiles',
  preview: 'Developer preview',
  boundaryShort: 'Compiled bundles → latents',
  choose: 'Choose your GPU',
  memory: 'VRAM',
  execution: 'Weight loading',
  available: 'Available profiles',
  inspect: 'Inspect the configuration',
  gpu: [
    { execution: 'Resident', profiles: 'Turbo4 · Turbo8', note: 'Loaded weights stay in GPU memory for subsequent requests.' },
    { execution: 'Streamed', profiles: 'Turbo4', note: 'Tested workload: allow 64 GiB+ available RAM per worker.' },
  ],
  boundary: 'What runs today',
  boundaryText: 'Compiled conditioning bundles in; video and audio latents (tensors ready for decoding) out. Compatible runtime assets are required and are not yet published. Prompt processing and MP4 output are still in development.',
  inputs: 'See the required files',
  next: 'A clear path to your first run',
  guides: [
    { number: '01', title: 'Install & check', text: 'Install the lightweight CLI and check that your GPU matches a profile.', path: '/guide/getting-started' },
    { number: '02', title: 'Run a bundle', text: 'Bring matching runtime assets, run denoising, and save the output.', path: '/guide/getting-started#run-a-bundle' },
    { number: '03', title: 'Connect your service', text: 'Start the HTTP API with Docker, submit jobs, and download results.', path: '/guide/docker' },
  ],
  source: 'Independent execution. Open source.',
  sourceText: 'Vflash implements H3 inference with PyTorch and Triton. The LightX2V runtime is not required.',
  architecture: 'See how it works',
})
const profile = computed(() => `ref2va-turbo4-exact-sm${selected.value === 0 ? '89' : '86'}`)

function selectWithKeyboard(event: KeyboardEvent) {
  if (!['ArrowLeft', 'ArrowRight', 'Home', 'End'].includes(event.key)) return
  event.preventDefault()
  selected.value = event.key === 'Home' ? 0 : event.key === 'End' ? 1 : 1 - selected.value
  tabs.value[selected.value]?.focus()
}
</script>

<template>
  <div class="vf-home">
    <section class="vf-hero" aria-labelledby="vf-title">
      <div class="vf-intro">
        <p class="vf-eyebrow"><span aria-hidden="true"></span>{{ text.eyebrow }}</p>
        <h1 id="vf-title">{{ text.title }}<br><span>{{ text.accent }}</span></h1>
        <p class="vf-description">{{ text.intro }}</p>
        <div class="vf-actions">
          <a class="vf-button" :href="link('/guide/getting-started')">{{ text.start }} <span aria-hidden="true">→</span></a>
          <a class="vf-text-link" :href="link('/guide/profiles')">{{ text.support }} <span aria-hidden="true">↗</span></a>
        </div>
        <p class="vf-meta">{{ text.preview }}<span aria-hidden="true">/</span>{{ text.boundaryShort }}</p>
      </div>

      <div class="vf-gpu-card">
        <div class="vf-card-label">{{ text.choose }}<span aria-hidden="true">↳</span></div>
        <div class="vf-tabs" role="tablist" :aria-label="text.choose" @keydown="selectWithKeyboard">
          <button v-for="(name, index) in ['RTX 4090', 'RTX 3080']" :id="`gpu-tab-${index}`" :key="name"
            :ref="element => { if (element) tabs[index] = element as HTMLButtonElement }"
            type="button" role="tab" :aria-selected="selected === index" aria-controls="gpu-panel"
            :tabindex="selected === index ? 0 : -1" @click="selected = index">{{ name }}</button>
        </div>
        <div id="gpu-panel" role="tabpanel" :aria-labelledby="`gpu-tab-${selected}`" tabindex="0">
          <div class="vf-gpu-name">{{ selected === 0 ? '48' : '20' }}<span>GB</span><small>{{ text.memory }}</small></div>
          <dl class="vf-specs">
            <div><dt>{{ text.execution }}</dt><dd>{{ text.gpu[selected].execution }}</dd></div>
            <div><dt>{{ text.available }}</dt><dd>{{ text.gpu[selected].profiles }}</dd></div>
          </dl>
          <p class="vf-gpu-note">{{ text.gpu[selected].note }}</p>
          <div class="vf-command">
            <div>{{ text.inspect }}</div>
            <code><span class="vf-prompt" aria-hidden="true">$ </span>vflash plan &#92;<br><span class="vf-command-indent">{{ profile }} &#92;</span><br><span class="vf-command-indent">--gpu 0</span></code>
          </div>
        </div>
      </div>
    </section>

    <section class="vf-availability" aria-labelledby="vf-availability-title">
      <div><span class="vf-preview-dot" aria-hidden="true"></span><h2 id="vf-availability-title">{{ text.boundary }}</h2></div>
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
