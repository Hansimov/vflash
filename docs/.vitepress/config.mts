import { defineConfig } from 'vitepress'
import { readFileSync } from 'node:fs'

const version = readFileSync(new URL('../../pyproject.toml', import.meta.url), 'utf8')
  .match(/^version = "([^"]+)"/m)?.[1]
if (!version) throw new Error('Missing package version for documentation')

const sidebar = (zh = false) => {
  const prefix = zh ? '/zh' : ''
  return [
    {
      text: zh ? '开始使用' : 'Start here',
      items: [
        { text: zh ? '开始使用' : 'Get started', link: `${prefix}/guide/getting-started` },
        { text: zh ? '配置与硬件' : 'Profiles and hardware', link: `${prefix}/guide/profiles` },
        { text: zh ? '准备运行资源' : 'Prepare runtime assets', link: `${prefix}/reference/runtime-assets` },
      ],
    },
    {
      text: zh ? '接入应用' : 'Build with Vflash',
      items: [
        { text: zh ? 'Docker 与 HTTP API' : 'Docker and HTTP API', link: `${prefix}/guide/docker` },
        { text: zh ? 'Python 集成' : 'Python integration', link: `${prefix}/guide/python` },
        { text: zh ? '用 Python 生成视频' : 'Generate a video', link: `${prefix}/guide/complete-pipeline` },
        { text: zh ? '编译官方权重' : 'Compile official weights', link: `${prefix}/guide/compile-weights` },
        { text: zh ? '测量性能与质量' : 'Measure speed and quality', link: `${prefix}/reference/performance` },
      ],
    },
    {
      text: zh ? '参考' : 'Reference',
      items: [
        { text: zh ? '实现方式' : 'How it works', link: `${prefix}/reference/architecture` },
        { text: zh ? '实测数据' : 'Benchmark results', link: `${prefix}/reference/benchmarks` },
        { text: zh ? '版本更新' : 'Release notes', link: `${prefix}/reference/releases` },
        { text: zh ? '许可证与致谢' : 'License and acknowledgements', link: `${prefix}/reference/license` },
      ],
    },
  ]
}

export default defineConfig({
  base: '/vflash/',
  title: 'Vflash',
  description: 'Native MiniMax H3 inference for RTX 3080 20 GB and RTX 4090 48 GB.',
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
          { text: 'Guides', link: '/guide/docker', activeMatch: '/guide/(docker|python)' },
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
      description: '为 RTX 3080 20 GB 和 RTX 4090 48 GB 优化的原生 MiniMax H3 推理引擎。',
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
          { text: '接入指南', link: '/zh/guide/docker', activeMatch: '/zh/guide/(docker|python)' },
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
