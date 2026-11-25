import { defineConfig } from 'vitepress'
import { readFileSync } from 'node:fs'

const version = readFileSync(new URL('../../pyproject.toml', import.meta.url), 'utf8')
  .match(/^version = "([^"]+)"/m)?.[1]
if (!version) throw new Error('Missing package version for documentation')

const sections = [
  { en: 'Using Vflash', zh: '使用手册', pages: [
    ['Install and verify', '安装与环境检查', 'guide/getting-started'],
    ['Choose a runtime', '按任务选择运行时', 'guide/runtime-selection'],
    ['Hardware requirements', '硬件要求', 'guide/profiles'],
    ['Generate from text or images', '文字与图像生成视频', 'guide/complete-pipeline'],
    ['Pika video and sound', 'Pika 画面与声音', 'guide/pika-flex'],
    ['Prepare and finish footage', '准备源片与导出成片', 'guide/source-window'],
    ['Edit a source video', '编辑源视频', 'guide/remake'],
  ] },
  { en: 'Integration', zh: '集成与部署', pages: [
    ['Prepare model assets', '准备模型资源', 'guide/compile-weights'],
    ['Python sessions', 'Python 会话', 'guide/python'],
    ['Run the service', '部署推理服务', 'guide/docker'],
    ['HTTP API', 'HTTP 接口', 'reference/http-api'],
    ['Diagnose failures', '故障排查', 'guide/troubleshooting'],
  ] },
  { en: 'Engine reference', zh: '引擎参考', pages: [
    ['Architecture', '引擎架构', 'reference/architecture'],
    ['Model profiles', '模型与配置', 'reference/pipeline-profiles'],
    ['Asset formats', '资源格式', 'reference/runtime-assets'],
    ['Attention and memory', '注意力与显存', 'guide/veda'],
    ['Performance measurements', '性能测量', 'reference/benchmarks'],
    ['Compatibility and upgrades', '兼容性与升级', 'reference/releases'],
    ['Contributing', '参与开发', 'guide/contributing'],
    ['Licenses and credits', '许可与致谢', 'reference/license'],
  ] },
]
const sidebar = (zh = false) => sections.map(group => ({
  text: zh ? group.zh : group.en,
  items: group.pages.map(([en, cn, path]) => ({ text: zh ? cn : en, link: (zh ? '/zh/' : '/') + path })),
}))
const nav = (zh = false) => sections.map(group => ({
  text: zh ? group.zh : group.en,
  link: (zh ? '/zh/' : '/') + group.pages[0][2],
  activeMatch: '^' + (zh ? '/zh/' : '/') + '(?:' + group.pages.map(page => page[2]).join('|') + ')(?:$|[.#/])',
}))

export default defineConfig({
  base: '/vflash/',
  title: 'Vflash',
  description: 'MiniMax H3 inference with prepared Pika audio/video runtimes, fixed-profile complete pipelines and native latent execution. Each path has explicit model assets, input limits and hardware scope.',
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
        nav: nav(),
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
      description: 'MiniMax H3 推理：准备权重的Pika音画运行时、固定配置的完整流水线和原生潜变量执行。各路径独立声明模型资产、输入限制和硬件范围。',
      markdown: {
        container: {
          infoLabel: '说明', noteLabel: '备注', tipLabel: '提示', warningLabel: '注意',
          dangerLabel: '警告', detailsLabel: '详情', importantLabel: '重要', cautionLabel: '注意',
        },
        codeCopyButton: { tooltipText: '复制代码', copiedText: '已复制' },
      },
      themeConfig: {
        outline: { level: [2, 3], label: '本页内容' },
        nav: nav(true),
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
