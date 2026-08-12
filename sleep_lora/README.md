# 已内置的睡眠 LoRA

该目录包含已经训练完成、无需再次运行 Colab 的 `checkpoint-160`。

- 基座：`zai-org/chatglm3-6b`
- PEFT：LoRA，r=8，alpha=32，dropout=0.1
- 目标模块：`query_key_value`
- 权重：`checkpoint-160/adapter_model.safetensors`
- SHA-256：`207588c4401877ce68a1ed8006ba7ec165ecfa8bcc18308b7db375e78d4ae3c5`

直接从仓库根目录运行 `./run_sleep_chat.sh`。首次运行会自动安装独立环境并下载 ChatGLM3 基座；adapter 已经在仓库中，之后会复用虚拟环境与 Hugging Face 缓存。

此模型在高风险安全测试中表现不足，不能替代医生、急诊或危机干预服务。完整实验见 `睡眠测试/睡眠模型与ChatGLM3对比测试/EXPERIMENT_REPORT.md`。
