#!/usr/bin/env bash
# AI Scientist 连续论文生成系统 - 快速启动脚本

set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$PROJECT_ROOT"

PYTHON_BIN=""
RESEARCH_DIR=""
MIN_PYTHON="3.10"

# 颜色定义
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

# 打印带颜色的消息
print_info() {
    echo -e "${BLUE}ℹ️  $1${NC}"
}

print_success() {
    echo -e "${GREEN}✅ $1${NC}"
}

print_warning() {
    echo -e "${YELLOW}⚠️  $1${NC}"
}

print_error() {
    echo -e "${RED}❌ $1${NC}"
}

print_header() {
    echo -e "\n${GREEN}═══════════════════════════════════════════════════════${NC}"
    echo -e "${GREEN}  $1${NC}"
    echo -e "${GREEN}═══════════════════════════════════════════════════════${NC}\n"
}

resolve_python_bin() {
    local -a candidates=()
    if [[ -n "${PYTHON:-}" ]]; then
        candidates+=("$PYTHON")
    fi
    candidates+=("python3.11" "python3.10" "python3")

    local candidate
    for candidate in "${candidates[@]}"; do
        if ! command -v "$candidate" >/dev/null 2>&1; then
            continue
        fi
        if "$candidate" - <<'PY' >/dev/null 2>&1
import sys
raise SystemExit(0 if sys.version_info >= (3, 10) else 1)
PY
        then
            printf '%s\n' "$candidate"
            return 0
        fi
    done
    return 1
}

resolve_research_dir() {
    "$PYTHON_BIN" - <<'PY' "$PROJECT_ROOT"
import sys
from pathlib import Path

project_root = Path(sys.argv[1]).resolve()
sys.path.insert(0, str(project_root))

from ai_scientist.config.paths import resolve_output_path  # noqa: E402

print(resolve_output_path())
PY
}

check_python() {
    if ! PYTHON_BIN="$(resolve_python_bin)"; then
        print_error "未找到 Python >= ${MIN_PYTHON} 的解释器（可通过 PYTHON 指定）"
        exit 1
    fi

    local version_output
    version_output="$("$PYTHON_BIN" --version 2>&1)"
    print_success "Python: ${version_output}"

    if ! RESEARCH_DIR="$(resolve_research_dir)"; then
        print_error "无法解析研究输出目录"
        exit 1
    fi
    export RESEARCH_OUTPUT_DIR="$RESEARCH_DIR"
}

check_latex() {
    if ! command -v pdflatex >/dev/null 2>&1; then
        print_warning "未找到 pdflatex，PDF生成可能失败"
        print_info "安装: apt-get install texlive (Linux) 或 MacTeX (macOS)"
    else
        print_success "LaTeX: $(pdflatex --version | head -n1)"
    fi
}

require_zhipu_api_key() {
    if [[ -n "${ZHIPU_API_KEY:-}" ]]; then
        print_success "智谱API密钥已设置"
        return 0
    fi
    print_error "未设置 ZHIPU_API_KEY 环境变量"
    print_info "请运行: export ZHIPU_API_KEY='your_api_key_here'"
    return 1
}

require_login_session() {
    if "$PYTHON_BIN" - <<'PY' >/dev/null 2>&1
from ai_scientist.utils.auth_session import validate_session
ok, _, _ = validate_session()
raise SystemExit(0 if ok else 1)
PY
    then
        print_success "登录状态: 已登录"
        return 0
    fi

    print_error "未登录，无法继续执行任何操作"
    print_info "请先执行: python3 auth_cli.py login --user <你的用户名>"
    return 1
}

# 检查依赖
check_dependencies() {
    print_header "检查依赖"
    check_python
    require_login_session || exit 1
    check_latex
}

# 创建研究目录
setup_research_dir() {
    print_header "设置研究目录"

    if [[ ! -d "$RESEARCH_DIR" ]]; then
        mkdir -p "$RESEARCH_DIR"
        print_success "创建研究目录: $RESEARCH_DIR"
    else
        print_success "研究目录已存在: $RESEARCH_DIR"
    fi

    mkdir -p "$RESEARCH_DIR"/{cache,ideas,experiments,papers,batches,projects}
    print_success "子目录已创建"

    echo ""
    df -h "$RESEARCH_DIR" | tail -1 | awk '{print "可用空间: " $4}'
}

# 显示菜单
show_menu() {
    print_header "AI Scientist 连续论文生成系统"
    echo "项目目录: $PROJECT_ROOT"
    echo "输出目录: $RESEARCH_DIR"
    echo ""
    echo "请选择操作:"
    echo ""
    echo "  1. 快速开始 - 生成3个想法并创建workshop论文"
    echo "  2. 生成所有类型论文 - 为想法创建所有格式论文"
    echo "  3. 自定义生成 - 使用自定义参数"
    echo "  4. 管理研究目录 - 查看/搜索/清理"
    echo "  5. 查看批次状态"
    echo "  6. 守护模式演练"
    echo "  7. 启动稳定守护模式"
    echo "  8. 查看守护状态"
    echo "  9. 退出"
    echo ""
    read -r -p "请输入选项 [1-9]: " choice
}

resolve_topic_file() {
    local topic_file=""
    if [[ -f "$PROJECT_ROOT/examples/example_topic.md" ]]; then
        topic_file="$PROJECT_ROOT/examples/example_topic.md"
    else
        read -r -p "请输入主题文件路径: " topic_file
    fi

    if [[ ! -f "$topic_file" ]]; then
        print_error "主题文件不存在: $topic_file"
        return 1
    fi

    printf '%s\n' "$topic_file"
}

# 快速开始
quick_start() {
    print_header "快速开始"
    require_zhipu_api_key || return 1

    local topic_file
    topic_file="$(resolve_topic_file)" || return 1

    read -r -p "生成想法数量 [默认: 3]: " num_ideas
    num_ideas=${num_ideas:-3}

    print_info "开始生成..."
    "$PYTHON_BIN" continuous_paper_generator.py \
        --topic "$topic_file" \
        --num-ideas "$num_ideas" \
        --paper-types icbinb

    print_success "完成！使用以下命令查看结果:"
    echo "  $PYTHON_BIN research_manager.py list-papers --type icbinb"
}

# 生成所有类型
generate_all_types() {
    print_header "生成所有类型论文"
    require_zhipu_api_key || return 1

    local topic_file
    topic_file="$(resolve_topic_file)" || return 1

    read -r -p "生成想法数量 [默认: 3]: " num_ideas
    num_ideas=${num_ideas:-3}

    read -r -p "并行worker数量 [默认: 1]: " num_workers
    num_workers=${num_workers:-1}

    print_info "开始生成所有类型论文..."
    "$PYTHON_BIN" continuous_paper_generator.py \
        --topic "$topic_file" \
        --num-ideas "$num_ideas" \
        --all-types \
        --num-workers "$num_workers"

    print_success "完成！使用以下命令查看结果:"
    echo "  $PYTHON_BIN research_manager.py list-papers"
}

# 自定义生成
custom_generate() {
    print_header "自定义生成"
    require_zhipu_api_key || return 1

    echo "请输入参数 (留空使用默认值):"

    read -r -p "主题文件路径或已有想法JSON: " input_file
    if [[ -z "$input_file" ]]; then
        print_error "请输入主题文件或想法 JSON 路径"
        return 1
    fi
    if [[ ! -f "$input_file" ]]; then
        print_error "输入文件不存在: $input_file"
        return 1
    fi

    read -r -p "论文类型 (icbinb/normal/journal/extended/all) [默认: icbinb]: " paper_type
    paper_type=${paper_type:-icbinb}

    read -r -p "想法索引 (逗号分隔，留空处理全部): " idea_indices

    local -a cmd=("$PYTHON_BIN" continuous_paper_generator.py)

    if [[ "$input_file" == *.md ]]; then
        cmd+=(--topic "$input_file")
        read -r -p "生成想法数量 [默认: 3]: " num_ideas
        num_ideas=${num_ideas:-3}
        cmd+=(--num-ideas "$num_ideas")
    else
        cmd+=(--ideas "$input_file")
    fi

    if [[ "$paper_type" == "all" ]]; then
        cmd+=(--all-types)
    else
        cmd+=(--paper-types "$paper_type")
    fi

    if [[ -n "$idea_indices" ]]; then
        cmd+=(--idea-indices "$idea_indices")
    fi

    local display_cmd
    printf -v display_cmd '%q ' "${cmd[@]}"
    print_info "执行命令: ${display_cmd% }"
    "${cmd[@]}"

    print_success "完成！"
}

# 管理研究目录
manage_research() {
    print_header "管理研究目录"

    echo "请选择操作:"
    echo "  1. 列出所有批次"
    echo "  2. 列出所有论文"
    echo "  3. 列出所有想法"
    echo "  4. 搜索论文"
    echo "  5. 查看统计信息"
    echo "  6. 清理旧文件"
    echo "  7. 返回"
    echo ""
    read -r -p "请输入选项 [1-7]: " mgmt_choice

    case "$mgmt_choice" in
        1) "$PYTHON_BIN" research_manager.py list-batches ;;
        2) "$PYTHON_BIN" research_manager.py list-papers ;;
        3) "$PYTHON_BIN" research_manager.py list-ideas ;;
        4)
            read -r -p "输入搜索关键词: " query
            "$PYTHON_BIN" research_manager.py search-papers "$query"
            ;;
        5) "$PYTHON_BIN" research_manager.py stats ;;
        6)
            read -r -p "删除多少天前的文件 [默认: 30]: " days
            days=${days:-30}
            "$PYTHON_BIN" research_manager.py cleanup --days "$days" --dry-run
            read -r -p "确认删除? [y/N]: " confirm
            if [[ "$confirm" == "y" ]]; then
                "$PYTHON_BIN" research_manager.py cleanup --days "$days"
            fi
            ;;
        7) return ;;
        *) print_error "无效选项" ;;
    esac
}

# 查看批次状态
view_batch_status() {
    print_header "批次状态"

    "$PYTHON_BIN" research_manager.py list-batches

    echo ""
    read -r -p "输入批次名称查看详情 (留空跳过): " batch_name

    if [[ -n "$batch_name" ]]; then
        "$PYTHON_BIN" research_manager.py batch-summary "$batch_name"
    fi
}

# 守护模式演练
daemon_rehearsal() {
    print_header "守护模式演练"
    print_info "运行 dry-run 演练，检查守护模式关键产物是否能正确生成..."
    "$PYTHON_BIN" run_daemon_rehearsal.py
    print_info "如需额外做严格预检，可执行: bash run_stable_daemon.sh doctor"
}

# 启动稳定守护模式
start_stable_daemon() {
    print_header "启动稳定守护模式"
    require_zhipu_api_key || return 1

    if [[ ! -f "$PROJECT_ROOT/run_stable_daemon.sh" ]]; then
        print_error "未找到脚本: run_stable_daemon.sh"
        return 1
    fi

    echo "请选择稳定模式:"
    echo "  1. 自动选择 (白天/夜间)"
    echo "  2. 平衡长跑"
    echo "  3. 白天打磨"
    echo "  4. 夜间生成"
    read -r -p "请输入选项 [1-4]: " daemon_mode_choice

    local daemon_mode="auto"
    case "${daemon_mode_choice:-1}" in
        1) daemon_mode="auto" ;;
        2) daemon_mode="balanced" ;;
        3) daemon_mode="day" ;;
        4) daemon_mode="night" ;;
        *) daemon_mode="auto" ;;
    esac

    read -r -p "是否后台启动守护模式? [Y/n]: " background_choice
    background_choice=${background_choice:-Y}

    if [[ "$background_choice" =~ ^[Nn]$ ]]; then
        print_info "前台启动稳定守护模式 ($daemon_mode)..."
        bash run_stable_daemon.sh "$daemon_mode"
    else
        print_info "后台启动稳定守护模式 ($daemon_mode)..."
        bash run_stable_daemon.sh "$daemon_mode" --background
        echo "可使用以下命令查看状态:"
        echo "  bash run_stable_daemon.sh status"
        echo "  bash run_stable_daemon.sh open-dashboard"
        echo "  bash run_stable_daemon.sh logs --lines 80"
        echo "  $PYTHON_BIN research_manager.py submission-board --top 5"
        echo "  $PYTHON_BIN research_manager.py rewrite-board --top 10"
    fi
}

# 查看守护状态
view_daemon_status() {
    print_header "守护状态"

    if [[ ! -f "$PROJECT_ROOT/run_stable_daemon.sh" ]]; then
        print_error "未找到脚本: run_stable_daemon.sh"
        return 1
    fi

    bash run_stable_daemon.sh status --lines 60
    echo ""
    print_info "更多运维命令:"
    echo "  bash run_stable_daemon.sh dashboard"
    echo "  bash run_stable_daemon.sh open-dashboard"
    echo "  bash run_stable_daemon.sh logs --lines 80"
    echo "  bash run_stable_daemon.sh tail-heartbeat --lines 40"
    echo "  bash run_stable_daemon.sh pause --target-mode balanced"
    echo "  bash run_stable_daemon.sh resume --target-mode balanced"
    echo "  bash run_stable_daemon.sh disable-source broad_impact_day --target-mode balanced"
    echo "  bash run_stable_daemon.sh source-force-next conference_night_watch --target-mode balanced"
    echo "  bash run_stable_daemon.sh source-summary --target-mode balanced"
    echo "  bash run_stable_daemon.sh source-plan --target-mode balanced"
}

# 主循环
main() {
    check_dependencies
    setup_research_dir

    while true; do
        show_menu

        case "$choice" in
            1) quick_start ;;
            2) generate_all_types ;;
            3) custom_generate ;;
            4) manage_research ;;
            5) view_batch_status ;;
            6) daemon_rehearsal ;;
            7) start_stable_daemon ;;
            8) view_daemon_status ;;
            9)
                print_info "退出"
                exit 0
                ;;
            *)
                print_error "无效选项"
                ;;
        esac

        echo ""
        read -r -p "按 Enter 继续..."
    done
}

main
