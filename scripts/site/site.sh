#!/usr/bin/env bash
# 知识站点：把 docs/ 下的 Markdown 构建成 HTML（产物 site/，不入库）。
#
#   scripts/site/site.sh build            # 构建到 site/
#   scripts/site/site.sh serve [addr]     # 本地预览，默认 127.0.0.1:8732（或 SITE_ADDR），改 md 自动刷新；不需要先 build
#   scripts/site/site.sh check            # 严格构建：有断链 / 锚点错误即失败
#   scripts/site/site.sh deploy           # 构建并推送到 GitHub Pages 仓库（见下）
#   scripts/site/site.sh clean
#
# 依赖：有 uv 时用 uv 临时环境（不碰项目 .venv）；否则在 .venv-site/ 建一个 venv
# （可用 PYTHON=/path/to/python3 指定解释器）。
# 版本固定在 scripts/site/requirements.txt。
#
# 部署：默认推送到 https://github.com/torelax/torelax.github.io 的 main 分支（站点根 https://torelax.github.io/）。
# Pages 仓库的内容完全由 site/ 决定，仓库里其他文件会被删除。换仓库时：
#   PAGES_REPO=https://github.com/<user>/<repo>.git PAGES_BRANCH=main \
#   SITE_URL=https://<user>.github.io/<repo>/ scripts/site/site.sh deploy
# 推送前会显示变更摘要并要求确认；加 -y 跳过确认。
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"
REQ="scripts/site/requirements.txt"
VENV="$ROOT/.venv-site"
DEPLOY_DIR="$ROOT/.site-deploy"

export NO_MKDOCS_2_WARNING=1           # Material 9.7 的 MkDocs 2.0 横幅
export PYTHONWARNINGS="ignore"         # jieba 在 Python 3.12 下的转义字符警告
export PYTHONIOENCODING=utf-8

mkdocs_cmd() {
  if command -v uv >/dev/null 2>&1; then
    uv run --quiet --isolated --no-project --with-requirements "$REQ" mkdocs "$@"
  else
    local stamp="$VENV/.requirements.sha"
    local want; want="$(sha1sum "$REQ" | cut -d' ' -f1)"
    if [[ ! -x "$VENV/bin/mkdocs" || "$(cat "$stamp" 2>/dev/null)" != "$want" ]]; then
      echo ">> 安装站点依赖到 .venv-site/（首次或 requirements 变更）"
      rm -rf "$VENV"
      if ! "${PYTHON:-python3}" -m venv "$VENV" >/dev/null 2>&1; then
        rm -rf "$VENV"
        echo "!! 无法用 ${PYTHON:-python3} 创建 venv（Debian/Ubuntu 常缺 python3-venv）。任选其一：" >&2
        echo "   1) 安装 uv（推荐）：curl -LsSf https://astral.sh/uv/install.sh | sh" >&2
        echo "   2) sudo apt install python3-venv" >&2
        echo "   3) 指定其他 Python：PYTHON=/path/to/python3 $0 $cmd" >&2
        exit 1
      fi
      "$VENV/bin/pip" install --quiet --upgrade pip
      "$VENV/bin/pip" install --quiet -r "$REQ"
      echo "$want" > "$stamp"
    fi
    "$VENV/bin/mkdocs" "$@"
  fi
}

cmd="${1:-build}"; shift || true
case "$cmd" in
  build)
    mkdocs_cmd build --clean "$@"
    echo ">> 站点已生成：$ROOT/site/index.html （$(du -sh site | cut -f1)）"
    ;;
  serve)
    addr="${1:-${SITE_ADDR:-127.0.0.1:8732}}"; shift || true
    port="${addr##*:}"
    echo ">> 预览：http://$addr/ （首次全量构建约 30–60 秒后可访问；改 md 自动刷新）"
    if [[ -n "${SSH_CONNECTION:-}" && "$addr" == 127.0.0.1:* ]]; then
      echo ">> 当前是 SSH 会话，在你自己的电脑上开端口转发后访问 http://127.0.0.1:$port/ ："
      echo "     ssh -N -L $port:127.0.0.1:$port $(whoami)@$(echo "$SSH_CONNECTION" | awk '{print $3}')"
      echo "   或者监听内网地址：$0 serve 0.0.0.0:$port"
    fi
    # --dirtyreload：只重建改动页面，350+ 篇文档时全量重建太慢
    mkdocs_cmd serve --dev-addr "$addr" --dirtyreload "$@"
    ;;
  check)
    mkdocs_cmd build --clean --strict "$@"
    ;;
  deploy)
    yes=0; [[ "${1:-}" == "-y" ]] && yes=1
    PAGES_REPO="${PAGES_REPO:-https://github.com/torelax/torelax.github.io.git}"
    branch="${PAGES_BRANCH:-main}"
    mkdocs_cmd build --clean
    if [[ ! -d "$DEPLOY_DIR/.git" ]]; then
      git clone --quiet --depth 1 --branch "$branch" "$PAGES_REPO" "$DEPLOY_DIR" 2>/dev/null \
        || { rm -rf "$DEPLOY_DIR"; git init --quiet -b "$branch" "$DEPLOY_DIR"; git -C "$DEPLOY_DIR" remote add origin "$PAGES_REPO"; }
    else
      git -C "$DEPLOY_DIR" remote set-url origin "$PAGES_REPO"
      git -C "$DEPLOY_DIR" fetch --quiet --depth 1 origin "$branch" 2>/dev/null \
        && git -C "$DEPLOY_DIR" reset --quiet --hard "origin/$branch" || true
    fi
    rsync -a --delete --exclude '.git' site/ "$DEPLOY_DIR/"
    touch "$DEPLOY_DIR/.nojekyll"       # 关掉 Jekyll，否则 _ 开头的目录会被丢弃
    git -C "$DEPLOY_DIR" add -A
    if git -C "$DEPLOY_DIR" diff --cached --quiet; then
      echo ">> 站点无变化，跳过推送"; exit 0
    fi
    git -C "$DEPLOY_DIR" diff --cached --shortstat
    if [[ $yes -ne 1 ]]; then
      read -r -p ">> 推送到 $PAGES_REPO ($branch)？[y/N] " ans
      [[ "$ans" == y || "$ans" == Y ]] || { echo "已取消"; exit 1; }
    fi
    src_rev="$(git rev-parse --short HEAD)"
    # 沿用本仓库的提交身份（只作用于这次提交，不写 git config）
    export GIT_AUTHOR_NAME="${GIT_AUTHOR_NAME:-$(git config user.name || true)}"
    export GIT_AUTHOR_EMAIL="${GIT_AUTHOR_EMAIL:-$(git config user.email || true)}"
    export GIT_COMMITTER_NAME="${GIT_COMMITTER_NAME:-$GIT_AUTHOR_NAME}"
    export GIT_COMMITTER_EMAIL="${GIT_COMMITTER_EMAIL:-$GIT_AUTHOR_EMAIL}"
    git -C "$DEPLOY_DIR" commit --quiet -m "deploy: ai-learn@$src_rev"
    # 首次推送约 300 MB，调大 HTTP 缓冲（只对本次命令生效，不改 git config）
    git -C "$DEPLOY_DIR" -c http.postBuffer=1048576000 push --quiet origin "HEAD:$branch"
    echo ">> 已推送 ai-learn@$src_rev 到 $PAGES_REPO ($branch)"
    ;;
  clean)
    rm -rf site "$DEPLOY_DIR"
    ;;
  *)
    sed -n '2,18p' "$0"; exit 1
    ;;
esac
