# 在 macOS 发布此项目（GitHub Desktop）

1. 下载并解压 `rekep-mobile-manipulation-github-ready.zip`，得到 `rekep-mobile-manipulation/` 文件夹。**不要再把原始导出包里的文件拖入这个文件夹，否则 README 会被覆盖。**
2. 安装 GitHub Desktop：https://desktop.github.com/ ，登录个人账号。
3. GitHub Desktop → `File` → `Add Local Repository...` → `Choose...` → 选中 `rekep-mobile-manipulation/`。
4. 如果提示此目录不是 Git 仓库，选择 `Create a repository here`，名称保留 `rekep-mobile-manipulation`。
5. 在 `Changes` 中确认只包括代码、配置、轻量结果与文档，没有云端模型/视频/密钥。提交说明写 `Initial project release`，点击 `Commit to main`。
6. 点击 `Publish repository`，选择 **个人账号**；勾选 `Keep this code private`，先以 Private 发布。
7. 在浏览器中核对 README、结果表和文件归属后，再决定是否转公开。

如果先前在 github.com 网页创建了同名空仓库：建议优先使用 GitHub Desktop 的 `Clone Repository` 克隆那个空仓库，**再**把本压缩包内的文件复制进克隆目录，提交并 `Push origin`。不要再点 `Publish repository` 创建第二个同名远程仓库。

GitHub Desktop 官方指南：https://docs.github.com/en/desktop/overview/creating-your-first-repository-using-github-desktop
