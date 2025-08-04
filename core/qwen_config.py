# core/qwen_config.py
"""
通义千问模型配置和工具函数
"""

# 通义千问模型配置
QWEN_MODELS = {
    "qwen-plus": {
        "name": "通义千问-Plus",
        "description": "平衡性能与成本，适合大多数场景",
        "context_length": 131072,
        "max_output": 8192,
        "cost_per_1k_tokens": {
            "input": 0.004,
            "output": 0.012
        },
        "features": ["文本生成", "代码生成", "多轮对话", "长文本理解"],
        "use_cases": ["日常对话", "文档分析", "代码辅助", "创意写作"]
    },
    "qwen-turbo": {
        "name": "通义千问-Turbo", 
        "description": "响应速度快，成本较低",
        "context_length": 131072,
        "max_output": 8192,
        "cost_per_1k_tokens": {
            "input": 0.002,
            "output": 0.006
        },
        "features": ["快速响应", "文本生成", "简单对话"],
        "use_cases": ["快速问答", "简单任务", "批量处理"]
    },
    "qwen-max": {
        "name": "通义千问-Max",
        "description": "最强性能，适合复杂任务",
        "context_length": 1000000,
        "max_output": 65536,
        "cost_per_1k_tokens": {
            "input": 0.02,
            "output": 0.06
        },
        "features": ["超长文本", "复杂推理", "专业分析", "高质量生成"],
        "use_cases": ["复杂分析", "长文档处理", "专业咨询", "高质量创作"]
    }
}

# 通义千问API配置
QWEN_API_CONFIG = {
    "base_url": "https://dashscope.aliyuncs.com/compatible-mode/v1",
    "timeout": 60,
    "max_retries": 3,
    "temperature_range": (0.1, 2.0),
    "top_p_range": (0.1, 1.0)
}

def get_qwen_model_info(model_name: str) -> dict:
    """获取通义千问模型信息"""
    return QWEN_MODELS.get(model_name, {})

def calculate_qwen_cost(model_name: str, input_tokens: int, output_tokens: int) -> float:
    """计算通义千问模型使用成本"""
    model_info = get_qwen_model_info(model_name)
    if not model_info:
        return 0.0
    
    cost_config = model_info.get("cost_per_1k_tokens", {})
    input_cost = (input_tokens / 1000) * cost_config.get("input", 0)
    output_cost = (output_tokens / 1000) * cost_config.get("output", 0)
    
    return round(input_cost + output_cost, 6)

def get_recommended_qwen_model(task_type: str) -> str:
    """根据任务类型推荐通义千问模型"""
    recommendations = {
        "simple": "qwen-turbo",      # 简单任务
        "balanced": "qwen-plus",     # 平衡任务
        "complex": "qwen-max",       # 复杂任务
        "long_text": "qwen-max",     # 长文本处理
        "code": "qwen-plus",         # 代码相关
        "creative": "qwen-max",      # 创意写作
        "analysis": "qwen-max",      # 分析任务
        "chat": "qwen-plus"          # 对话任务
    }
    return recommendations.get(task_type, "qwen-plus")

def validate_qwen_api_key(api_key: str) -> bool:
    """验证通义千问API密钥格式"""
    if not api_key:
        return False
    
    # 通义千问API密钥通常以sk-开头
    if not api_key.startswith("sk-"):
        return False
    
    # 检查长度（通常32-64字符）
    if len(api_key) < 32 or len(api_key) > 64:
        return False
    
    return True

def get_qwen_model_list() -> list:
    """获取所有可用的通义千问模型列表"""
    return [
        {
            "model_name": model_name,
            "display_name": config["name"],
            "description": config["description"],
            "context_length": config["context_length"],
            "cost_info": f"输入: ¥{config['cost_per_1k_tokens']['input']}/1K tokens, 输出: ¥{config['cost_per_1k_tokens']['output']}/1K tokens"
        }
        for model_name, config in QWEN_MODELS.items()
    ]