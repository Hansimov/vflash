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
  eyebrow: '当前版本',
  title: 'MiniMax H3 视频，原生运行。',
  accent: '从文字或图片到带声音的视频。',
  intro: '选择 Pika 六步画面与可选 Larry 原生声音，或固定配置的文字、关键帧与参考生成。Python 完整运行时交付 MP4，固定配置提供 CLI，原生 HTTP 交付潜变量。每条路径分别声明输入、资产和硬件范围。',
  start: '开始使用',
  support: '选择硬件',
  choose: '按显卡选择',
  memory: '显存',
  execution: '执行方式',
  available: '已实测路径',
  inspect: '检查显卡匹配，不加载权重',
  copy: '复制', copied: '已复制', copyFailed: '请手动选择命令复制',
  gpu: [
    { execution: '单卡 · 常驻或块环', profiles: 'SM89配置；混合v0.1最长15秒、原生4MP', note: '固定配置的完整视频参考平台。原生高清与长片需显式混合模型和对应内存配置；48 GiB 实测范围不等于24 GiB资格。' },
    { execution: '单卡 · FP8 常驻主干', profiles: '准备权重的 Pika T2VA / I2VA；另有混合模式', note: 'Pika6可配联合原声或Larry10／12声音；四秒范围要求Larry10。采用显式Python运行时及资产，下面命令只检查设备。' },
    { execution: '单卡 · 常驻或块环', profiles: 'SXM 混合 I2VA', note: '混合 I2VA 常驻主干需至少 75 GiB；画布、时长和内存策略按此路径的硬件指南选择。' },
    { execution: '双卡协作 · 块流式', profiles: 'Ref4 / T2VA / Base16', note: '两张 3080 共同处理一个请求（sequence-head）。每个 worker 预留至少 64 GiB 可用系统内存。' },
  ],
  boundary: '能生成什么',
  boundaryText: '准备权重的 Pika 提供限定四／五秒文本和首帧路径；内容参考使用独立 Ref 运行时。固定配置提供首尾帧及参考模式，部分混合配置扩展至十五秒或原生4MP。时长、画布、输入数量不能跨运行时组合；人物替换和长视频编排还需要应用工作流。',
  inputs: '查看模式与限制',
  next: '按你的任务开始',
  guides: [
    { number: '01', title: '安装并检查环境', text: '安装轻量 CLI，用 doctor 与 plan 确认显卡和配置。无需下载权重。', path: '/guide/getting-started' },
    { number: '02', title: '生成第一个视频', text: '准备官方模型资源，用命令行、Python 或容器从文字和图片生成带音轨的 MP4。', path: '/guide/complete-pipeline' },
    { number: '03', title: '选择 Pika 音画运行时', text: '在完整96 GiB SM120上，显式选择准备权重的Pika六步画面与联合原声或Larry原生声音。', path: '/guide/pika-flex#prepared-pika-av' },
  ],
  source: '理解实现，按需扩展。',
  sourceText: '原生去噪器、LoRA 合并、显存调度、双卡布局与媒体交付均由 Vflash 实现；编码器与 VAE 使用固定版本的官方代码。',
  architecture: '阅读实现方式',
} : {
  eyebrow: 'Current release',
  title: 'MiniMax H3 video, natively.',
  accent: 'From text or images to video with sound.',
  intro: 'Choose Pika six-step pictures with optional Larry native audio, or fixed-profile text, keyframe and reference generation. Complete Python runtimes deliver MP4, fixed profiles provide a CLI, and native HTTP returns latents. Each path declares its own inputs, assets and hardware scope.',
  start: 'Get started',
  support: 'Choose hardware',
  choose: 'Choose your GPU',
  memory: 'VRAM',
  execution: 'Execution',
  available: 'Measured paths',
  inspect: 'Check the GPU match without loading weights',
  copy: 'Copy', copied: 'Copied', copyFailed: 'Select the command to copy it',
  gpu: [
    { execution: 'One GPU · resident or block ring', profiles: 'SM89 profiles; hybrid v0.1 up to 15 s and native 4 MP', note: 'Reference allocation for fixed-profile complete videos. Native HD and longer clips require explicit hybrid and memory settings; 48 GiB evidence does not qualify 24 GiB cards.' },
    { execution: 'One GPU · FP8 resident trunk', profiles: 'Prepared Pika T2VA / I2VA; separate hybrid paths', note: 'Pika6 supports joint or Larry10/12 native sound; four seconds requires Larry10. Use the explicit Python runtime and assets. The command below only inspects devices.' },
    { execution: 'One GPU · resident or block ring', profiles: 'SXM hybrid I2VA', note: 'A hybrid I2VA resident trunk needs at least 75 GiB. Select canvas, duration and memory strategy from this path’s hardware guide.' },
    { execution: 'Two GPUs · streamed', profiles: 'Ref4 / T2VA / Base16', note: 'Two 3080s cooperate on one request with sequence-head. Allow at least 64 GiB of available host RAM per worker.' },
  ],
  boundary: 'What you can generate',
  boundaryText: 'Prepared Pika provides bounded four/five-second text and first-frame paths; content references use a separate Ref runtime. Fixed profiles provide endpoint and reference modes, with selected hybrids extending to fifteen seconds or native 4 MP. Duration, canvas and input limits cannot be combined across runtimes. Character replacement and long-video orchestration also need an application workflow.',
  inputs: 'See modes and limits',
  next: 'Start with what you need',
  guides: [
    { number: '01', title: 'Install and check', text: 'Install the lightweight CLI and confirm your GPU and profile with doctor and plan. No weights needed.', path: '/guide/getting-started' },
    { number: '02', title: 'Generate your first video', text: 'Prepare the official model assets and create an MP4 with sound from text and images via CLI, Python or Docker.', path: '/guide/complete-pipeline' },
    { number: '03', title: 'Choose a Pika audio/video runtime', text: 'Select prepared Pika six-step pictures with joint or Larry native sound on a complete 96 GiB SM120 allocation.', path: '/guide/pika-flex#prepared-pika-av' },
  ],
  source: 'Understand it. Build on it.',
  sourceText: 'The native denoiser, LoRA merging, memory scheduling, two-GPU layouts and media delivery are Vflash code; encoders and VAEs use pinned official implementations.',
  architecture: 'Explore the architecture',
})
const gpuNames = ['RTX 4090', 'RTX PRO 6000', 'H100', '2 × RTX 3080']
const gpuMemory = ['48', '96', '80', '2 × 20']
const profiles = [
  'i2va-turbo4-v01-544-exact-sm89',
  'i2va-turbo4-v01-544-exact-sm120',
  'i2va-turbo4-v01-544-exact-sm90',
  'ref2va-turbo4-exact-sm86',
]
const profile = computed(() => profiles[selected.value])
const command = computed(() => selected.value === 1 ? "vflash doctor" : `vflash plan ${profile.value} \\\n  --gpu 0${selected.value === 3 ? ' --peer-gpu 1 \\\n  --strategy sequence-head' : ''}`)

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
        <p class="vf-meta"><span>SM86 / SM89 / SM90 / SM120</span><span>T2VA · I2VA · L2VA · FL2VA · Ref2VA</span><span>Apache 2.0</span></p>
      </div>

      <div class="vf-gpu-card">
        <div class="vf-card-heading"><span>{{ text.choose }}</span><span>NVIDIA / CUDA</span></div>
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
      <p>{{ text.boundaryText }} <a :href="link('/guide/complete-pipeline#modes-and-limits')">{{ text.inputs }} <span aria-hidden="true">→</span></a></p>
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
