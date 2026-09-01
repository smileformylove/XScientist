# 快速入门

本指南对应 PyPI 正式版 `0.1.4`，需要 Python 3.10+ 和 Git。

## 1. 安装、保存想法、查看下一步

第一步不需要模型、API Key、Docker 或 Provider 配置：

```bash
python -m pip install "xscientist==0.1.4"
xscientist explore ./my-study --lang zh
xscientist status ./my-study --lang zh
```

`explore` 会询问预期现象、什么结果会让你改变看法，以及第一个公平检验。只回答你
真正知道的内容即可。`status` 随后只显示当前优先级最高的下一步，不会编造证据或结论。

如果想先看一个完整且不使用 Provider 的示例：

```bash
xscientist demo ./first-study --autopilot --lang zh
xscientist status ./first-study --lang zh
```

演示成本为 `$0.00`，并会故意停在“仍需补充证据”。留出结果挑战了过度宽泛的结论；
保留这个冲突是正确的科学行为，不是程序失败。

## 2. 按需加入模型

安装科研运行时和一个 Provider 客户端。下面使用 OpenAI；其他服务请选择对应 extra：

```bash
python -m pip install "xscientist[research,openai]==0.1.4"
export OPENAI_API_KEY="..."
xscientist start ./my-study --prepare-only
```

`explore` 已保存的问题会被复用。`--prepare-only` 只创建或更新工作区并检查本地前置
条件，不会启动研究。先按输出修复阻断项，再继续。

工作区就绪后，设置明确预算并启动：

```bash
xscientist start ./my-study --max-cost-usd 10
xscientist status ./my-study --lang zh
```

本地 Ollama 不需要托管 Key，但生成实验代码仍需配置好的隔离执行器。Provider 可用
不等于科研结论已经通过验证。

## 3. 投稿导向流程

仍然先完成安全准备，再用明确预算选择 publication autopilot：

```bash
xscientist start ./my-study --prepare-only
xscientist start ./my-study --autopilot publication --max-cost-usd 10
xscientist status ./my-study --lang zh
```

Publication autopilot 会组织科研、写作和评审门禁，但不承诺完成论文、科学验证、
实际投稿或被任何会议、期刊录用。

## 需要时再继续

- [长任务指南](LONG_RUNNING_GUIDE.md)：后台运行、查看、取消与恢复。
- [本地 Research Git](LOCAL_RESEARCH_GIT.md)：检查、保存、比较与恢复 checkpoint。
- [科研诚信](RESEARCH_INTEGRITY.md)：来源追踪和独立评审边界。
- [配置参考](CONFIG_REFERENCE.md)：Provider、执行器和专业能力。

`main` 分支还记录尚未发布的协议能力。只有需要明确标为 **开发版 main** 的功能时，
才从源码安装：

```bash
python -m pip install \
  "xscientist @ git+https://github.com/smileformylove/XScientist.git@main"
```

需要严格复现时，请固定源码 commit，不要跟随变化中的 `main`。
