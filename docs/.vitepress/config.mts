import { defineConfig } from 'vitepress'

const sidebar = (zh = false) => {
  const prefix = zh ? '/zh' : ''
  return [
    {
      text: zh ? '使用 Vflash' : 'Use Vflash',
      items: [
        { text: zh ? '开始使用' : 'Get started', link: `${prefix}/guide/getting-started` },
        { text: zh ? 'Docker 与 API' : 'Docker and API', link: `${prefix}/guide/docker` },
        { text: zh ? '配置与硬件' : 'Profiles and hardware', link: `${prefix}/guide/profiles` },
      ],
    },
    {
      text: zh ? '参考' : 'Reference',
      items: [
        { text: zh ? '运行资源' : 'Runtime assets', link: `${prefix}/reference/runtime-assets` },
        { text: zh ? '实现方式' : 'How it works', link: `${prefix}/reference/architecture` },
        { text: zh ? '性能测量' : 'Measuring performance', link: `${prefix}/reference/performance` },
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
    ['meta', { name: 'theme-color', content: '#fcfdfb' }],
  ],
  locales: {
    root: {
      label: 'English',
      lang: 'en',
      themeConfig: {
        nav: [
          { text: 'Get started', link: '/guide/getting-started' },
          { text: 'Docker & API', link: '/guide/docker' },
          { text: 'Support', link: '/guide/profiles' },
        ],
        sidebar: sidebar(),
        footer: {
          message: 'Native H3 inference. Open source.',
          copyright: 'Vflash · Apache 2.0',
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
          { text: 'Docker 与 API', link: '/zh/guide/docker' },
          { text: '支持范围', link: '/zh/guide/profiles' },
        ],
        sidebar: sidebar(true),
        docFooter: { prev: '上一页', next: '下一页' },
        footer: { message: '原生 H3 推理，开放源码。', copyright: 'Vflash · Apache 2.0' },
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
    logo: '/mark.svg',
    outline: { level: [2, 3] },
    socialLinks: [{ icon: 'github', link: 'https://github.com/Hansimov/vflash' }],
    search: { provider: 'local' },
  },
})
