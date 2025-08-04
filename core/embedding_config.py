# core/embedding_config.py
"""
Embedding模型配置文件
提供各种embedding模型的配置信息和工具函数
"""

# Embedding模型配置 - 针对中文内容优化
EMBEDDING_MODELS = {
    "dengcao/Qwen3-Embedding-0.6B:Q8_0": {
        "display_name": "Qwen3 Embedding 0.6B Q8 (推荐)",
        "provider": "Ollama",
        "model_name": "dengcao/Qwen3-Embedding-0.6B:Q8_0",
        "base_url": "http://localhost:11434",
        "description": "Qwen3中文embedding模型，量化版本，速度快，中文效果优秀",
        "dimension": 1536,
        "max_tokens": 8192,
        "features": ["中文专优", "量化加速", "本地部署", "免费使用"],
        "recommended": True,
        "chinese_optimized": True
    },
    "dengcao/Qwen3-Embedding-0.6B:F16": {
        "display_name": "Qwen3 Embedding 0.6B F16 (高精度)",
        "provider": "Ollama",
        "model_name": "dengcao/Qwen3-Embedding-0.6B:F16",
        "base_url": "http://localhost:11434",
        "description": "Qwen3中文embedding模型，全精度版本，精度最高",
        "dimension": 1536,
        "max_tokens": 8192,
        "features": ["中文专优", "全精度", "最高质量", "本地部署"],
        "recommended": False,
        "chinese_optimized": True
    },
    "nomic-embed-text": {
        "display_name": "Nomic Embed Text",
        "provider": "Ollama",
        "model_name": "nomic-embed-text",
        "base_url": "http://localhost:11434", 
        "description": "Nomic AI的开源embedding模型，主要适用于英文内容",
        "dimension": 768,
        "max_tokens": 8192,
        "features": ["开源", "英文优化", "本地部署"],
        "recommended": False,
        "chinese_optimized": False
    }
}

# API配置
API_CONFIG = {
    "timeout": 30,
    "max_retries": 3,
    "retry_delay": 1.0
}

def get_embedding_model_list():
    """获取所有embedding模型列表"""
    return list(EMBEDDING_MODELS.values())

def get_embedding_model_info(model_name):
    """根据模型名称获取模型信息"""
    return EMBEDDING_MODELS.get(model_name, None)

def get_recommended_models():
    """获取推荐的embedding模型列表"""
    return {k: v for k, v in EMBEDDING_MODELS.items() if v.get("recommended", False)}

def get_chinese_optimized_models():
    """获取中文优化的embedding模型列表"""
    return {k: v for k, v in EMBEDDING_MODELS.items() if v.get("chinese_optimized", False)}

def get_ollama_models():
    """获取Ollama本地模型列表"""
    return {k: v for k, v in EMBEDDING_MODELS.items() if v.get("provider") == "Ollama"}

def get_cloud_models():
    """获取云端API模型列表"""
    return {k: v for k, v in EMBEDDING_MODELS.items() if v.get("provider") != "Ollama"}

def validate_model_name(model_name):
    """验证模型名称是否有效"""
    return model_name in EMBEDDING_MODELS

def get_model_display_name(model_name):
    """获取模型的显示名称"""
    model_info = EMBEDDING_MODELS.get(model_name)
    return model_info.get("display_name", model_name) if model_info else model_name

def get_default_model():
    """获取默认推荐的embedding模型（中文优化）"""
    for model_id, config in EMBEDDING_MODELS.items():
        if config.get("recommended", False):
            return model_id
    return "dengcao/Qwen3-Embedding-0.6B:Q8_0"  # 备用默认值

def format_model_option(model_name):
    """格式化模型选项显示"""
    config = EMBEDDING_MODELS.get(model_name, {})
    display_name = config.get("display_name", model_name)
    provider = config.get("provider", "Unknown")
    return f"{display_name} ({provider})"

def is_chinese_optimized(model_name):
    """检查模型是否针对中文优化"""
    config = EMBEDDING_MODELS.get(model_name, {})
    return config.get("chinese_optimized", False)