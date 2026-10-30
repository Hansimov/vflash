import { defineConfig } from 'vitepress'
import { readFileSync } from 'node:fs'

const version = readFileSync(new URL('../../pyproject.toml', import.meta.url), 'utf8')
  .match(/^version = "([^"]+)"/m)?.[1]
if (!version) throw new Error('Missing package version for documentation')

const sidebar = (zh = false) => {
  const prefix = zh ? '/zh' : ''
  const item = (en: string, cn: string, path: string) => ({ text: zh ? cn : en, link: `${prefix}${path}` })
  return [
    {
      text: zh ? '开始使用' : 'Start here',
      items: [
        item('Get started', '开始使用', '/guide/getting-started'),
        item('Choose hardware', '选择硬件与配置', '/guide/profiles'),
        item('Prepare runtime assets', '准备运行资源', '/reference/runtime-assets'),
        item('Compile official weights', '编译官方权重', '/guide/compile-weights'),
      ],
    },
    {
      text: zh ? '生成视频' : 'Generate',
      items: [
        item('Generate a video', '生成视频', '/guide/complete-pipeline'),
        item('Python integration', 'Python 集成', '/guide/python'),
        item('Segmented video and native audio', '分段画面与原生声音', '/reference/segmented-av'),
        item('Docker and HTTP API', 'Docker 与 HTTP API', '/guide/docker'),
        item('HTTP response contract', 'HTTP响应合同', '/reference/http-api'),
      ],
    },
    {
      text: zh ? '调优与运维' : 'Tune and operate',
      items: [
        item('Speed, memory and quality', '速度、显存与质量', '/reference/performance'),
        item('Veda sparse attention', 'Veda 稀疏注意力', '/guide/veda'),
        item('Troubleshooting', '故障排查', '/guide/troubleshooting'),
      ],
    },
    {
      text: zh ? '可选与实验功能' : 'Optional and experimental',
      collapsed: true,
      items: [
        item('Attention LoRA', '注意力 LoRA', '/reference/attention-lora'),
        item('Video restoration', '画质增强', '/guide/restoration'),
        item('VDN8 community backend', 'VDN8 社区后端', '/guide/community-vdn'),
        item('Sol-Engine alignment', 'Sol-Engine 对齐', '/reference/sol-engine-alignment'),
      ],
    },
    {
      text: zh ? '参考' : 'Reference',
      items: [
        item('Model profiles', '模型配置', '/reference/pipeline-profiles'),
        item('How it works', '实现方式', '/reference/architecture'),
        item('Benchmark results', '实测数据', '/reference/benchmarks'),
        item('Release notes', '版本更新', '/reference/releases'),
        item('Contributing', '参与开发', '/guide/contributing'),
        item('License and acknowledgements', '许可证与致谢', '/reference/license'),
      ],
    },
  ]
}

export default defineConfig({
  base: '/vflash/',
  title: 'Vflash',
  description: 'Native MiniMax H3 video generation: text, image and keyframe modes up to 15 s and native 4 MP, plus an explicit segmented video and native audio runtime, on RTX 4090 48 GB, RTX PRO 6000 Blackwell, H100 and RTX 3080 pairs.',
  cleanUrls: true,
  appearance: true,
  sitemap: { hostname: 'https://hansimov.github.io/vflash/' },
  head: [
    ['link', { rel: 'icon', type: 'image/svg+xml', href: '/vflash/mark.svg' }],
    ['meta', { name: 'theme-color', content: '#fafafa', media: '(prefers-color-scheme: light)' }],
    ['meta', { name: 'theme-color', content: '#1b1b1d', media: '(prefers-color-scheme: dark)' }],
  ],
  locales: {
    root: {
      label: 'English',
      lang: 'en',
      themeConfig: {
        nav: [
          { text: 'Get started', link: '/guide/getting-started' },
          { text: 'Generate', link: '/guide/complete-pipeline', activeMatch: '/guide/(complete-pipeline|docker|python)' },
          { text: 'Hardware', link: '/guide/profiles' },
          { text: version, link: '/reference/releases' },
        ],
        sidebar: sidebar(),
        footer: {
          message: 'Vflash · Native MiniMax H3 inference',
          copyright: 'Apache 2.0 source · Model licenses apply separately',
        },
      },
    },
    zh: {
      label: '简体中文',
      lang: 'zh-CN',
      description: '原生 MiniMax H3 视频生成：文字、图像与关键帧模式，最长 15 秒、原生4MP，以及显式分段画面与原生声音运行时；支持 RTX 4090 48GB、RTX PRO 6000 Blackwell、H100 与双 RTX 3080。',
      markdown: {
        container: {
          infoLabel: '说明', noteLabel: '备注', tipLabel: '提示', warningLabel: '注意',
          dangerLabel: '警告', detailsLabel: '详情', importantLabel: '重要', cautionLabel: '注意',
        },
        codeCopyButton: { tooltipText: '复制代码', copiedText: '已复制' },
      },
      themeConfig: {
        outline: { level: [2, 3], label: '本页内容' },
        nav: [
          { text: '开始使用', link: '/zh/guide/getting-started' },
          { text: '生成视频', link: '/zh/guide/complete-pipeline', activeMatch: '/zh/guide/(complete-pipeline|docker|python)' },
          { text: '硬件', link: '/zh/guide/profiles' },
          { text: version, link: '/zh/reference/releases' },
        ],
        sidebar: sidebar(true),
        docFooter: { prev: '上一页', next: '下一页' },
        footer: { message: 'Vflash · 原生 MiniMax H3 推理', copyright: '源码使用 Apache 2.0 · 模型遵循各自许可证' },
        darkModeSwitchLabel: '外观',
        lightModeSwitchTitle: '切换到浅色主题',
        darkModeSwitchTitle: '切换到深色主题',
        sidebarMenuLabel: '目录',
        returnToTopLabel: '返回顶部',
        langMenuLabel: '切换语言',
        skipToContentLabel: '跳到正文',
        search: {
          provider: 'local',
          options: {
            translations: {
              button: { buttonText: '搜索', buttonAriaLabel: '搜索文档' },
              modal: {
                displayDetails: '显示详细列表', resetButtonTitle: '清除搜索', backButtonTitle: '关闭搜索',
                noResultsText: '没有找到相关结果',
                footer: {
                  selectText: '选择', selectKeyAriaLabel: '回车', navigateText: '切换',
                  navigateUpKeyAriaLabel: '上箭头', navigateDownKeyAriaLabel: '下箭头',
                  closeText: '关闭', closeKeyAriaLabel: '退出键',
                },
              },
            },
          },
        },
      },
    },
  },
  themeConfig: {
    version,
    logo: '/mark.svg',
    outline: { level: [2, 3] },
    socialLinks: [{ icon: 'github', link: 'https://github.com/Hansimov/vflash' }],
    search: { provider: 'local' },
  },
})
