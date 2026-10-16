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
  accent: '从文字或图片到带声音的 4MP 成片。',
  intro: '原生 PyTorch/Triton 去噪运行时，一次请求生成 5–15 秒带音轨的 MP4：文字、首帧、尾帧、首尾帧或最多九张参考图，图像条件下最高 2720×1536。可选六步音频修复，经 Python、命令行或 HTTP 服务接入。',
  start: '开始使用',
  support: '选择硬件',
  choose: '按显卡选择',
  memory: '显存',
  execution: '执行方式',
  available: '已实测路径',
  inspect: '检查显卡匹配，不加载权重',
  copy: '复制', copied: '已复制', copyFailed: '请手动选择命令复制',
  gpu: [
    { execution: '单卡 · 常驻或块环', profiles: '全部配置；混合 v0.1 最长 15 秒、原生 4MP', note: '参考平台。BF16 混合 Veda 44/6 实测 2720×1536 五秒约 660 秒；928×512 五秒含六步音频修复约 112 秒。' },
    { execution: '单卡 · FP8 常驻主干', profiles: '混合 I2VA（预览）', note: '0.15 起按像素帧预算常驻：含人脸与六步音频修复，1920×1088 五秒 163 秒、2720×1536 五秒 261 秒，显存峰值 42 GiB。' },
    { execution: '单卡 · 常驻或块环', profiles: '混合 I2VA（预览）', note: '常驻主干需至少 75 GiB；1536×864 十五秒常驻实测 227 秒。更大画布使用块环。' },
    { execution: '双卡协作 · 块流式', profiles: 'Ref4 / T2VA / Base16', note: '两张 3080 共同处理一个请求（sequence-head）。每个 worker 预留至少 64 GiB 可用系统内存。' },
  ],
  boundary: '能生成什么',
  boundaryText: 'T2VA、Ref2VA（1–9 张参考图）、I2VA、L2VA、FL2VA 与短视频参考。时长 5–15 秒、24 fps；画布宽高为 32 的倍数，普通请求最多 1MP，图像条件原生高清最多 2048² 像素、最长边 2720，并受 宽×高×帧数 ≤ 2048²×124 的联合预算约束。具体组合以所选配置与硬件的实测范围为准。',
  inputs: '查看模式与限制',
  next: '按你的任务开始',
  guides: [
    { number: '01', title: '安装并检查环境', text: '安装轻量 CLI，用 doctor 与 plan 确认显卡和配置。无需下载权重。', path: '/guide/getting-started' },
    { number: '02', title: '生成第一个视频', text: '准备官方模型资源，用命令行、Python 或容器从文字和图片生成带音轨的 MP4。', path: '/guide/complete-pipeline' },
    { number: '03', title: '调优速度与显存', text: '选择常驻或块环、Veda 注意力与音频修复，并分开测量冷启动与重复请求。', path: '/reference/performance' },
  ],
  source: '理解实现，按需扩展。',
  sourceText: '原生去噪器、LoRA 合并、显存调度、双卡布局与媒体交付均由 Vflash 实现；编码器与 VAE 使用固定版本的官方代码。',
  architecture: '阅读实现方式',
} : {
  eyebrow: 'Current release',
  title: 'MiniMax H3 video, natively.',
  accent: 'From text or images to 4 MP with sound.',
  intro: 'A native PyTorch/Triton denoising runtime that turns one request into a 5–15 second MP4 with audio: text, first frame, last frame, both, or up to nine reference images, up to 2720×1536 when image-conditioned. Optional six-step audio refinement. Use it from Python, the CLI or an HTTP service.',
  start: 'Get started',
  support: 'Choose hardware',
  choose: 'Choose your GPU',
  memory: 'VRAM',
  execution: 'Execution',
  available: 'Measured paths',
  inspect: 'Check the GPU match without loading weights',
  copy: 'Copy', copied: 'Copied', copyFailed: 'Select the command to copy it',
  gpu: [
    { execution: 'One GPU · resident or block ring', profiles: 'Every profile; hybrid v0.1 up to 15 s and native 4 MP', note: 'The reference platform. BF16 hybrid Veda 44/6 measured about 660 s for 2720×1536 at five seconds and about 112 s for 928×512 with six-step audio refinement.' },
    { execution: 'One GPU · FP8 resident trunk', profiles: 'Hybrid I2VA (preview)', note: 'Since 0.15 residency follows the pixel-frame budget: with face and six-step audio repair, 1920×1088/5 s took 163 s and 2720×1536/5 s 261 s at a 42 GiB peak.' },
    { execution: 'One GPU · resident or block ring', profiles: 'Hybrid I2VA (preview)', note: 'A resident trunk needs at least 75 GiB; 1536×864 for fifteen seconds measured 227 s resident. Larger canvases use the block ring.' },
    { execution: 'Two GPUs · streamed', profiles: 'Ref4 / T2VA / Base16', note: 'Two 3080s cooperate on one request with sequence-head. Allow at least 64 GiB of available host RAM per worker.' },
  ],
  boundary: 'What you can generate',
  boundaryText: 'T2VA, Ref2VA with 1–9 reference images, I2VA, L2VA, FL2VA and short reference videos. Five to fifteen seconds at 24 fps; canvases are multiples of 32, up to 1 MP for ordinary requests and up to 2048² pixels with a 2720-pixel long edge for image-conditioned native HD, within the joint width × height × frames ≤ 2048² × 124 budget. Each profile and GPU qualifies its own subset.',
  inputs: 'See modes and limits',
  next: 'Start with what you need',
  guides: [
    { number: '01', title: 'Install and check', text: 'Install the lightweight CLI and confirm your GPU and profile with doctor and plan. No weights needed.', path: '/guide/getting-started' },
    { number: '02', title: 'Generate your first video', text: 'Prepare the official model assets and create an MP4 with sound from text and images via CLI, Python or Docker.', path: '/guide/complete-pipeline' },
    { number: '03', title: 'Tune speed and memory', text: 'Pick residency, Veda attention and audio refinement, and measure cold start separately from repeated requests.', path: '/reference/performance' },
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
const command = computed(() => `vflash plan ${profile.value} \\\n  --gpu 0${selected.value === 3 ? ' --peer-gpu 1 \\\n  --strategy sequence-head' : ''}`)

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
        <p class="vf-meta"><span>SM86 / SM89 / SM90 / SM120</span><span>T2VA · I2VA · FL2VA · Ref2VA</span><span>Apache 2.0</span></p>
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
