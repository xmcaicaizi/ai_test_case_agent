# core/doubao_config.py
"""
豆包大模型配置文件
火山引擎豆包大模型的配置信息和辅助函数
"""

# 豆包模型配置
DOUBAO_MODELS = {
    "doubao-seed-1.6": {
        "display_name": "豆包 Seed 1.6",
        "model_name": "doubao-seed-1.6",
        "description": "最新一代豆包大模型，具备更强的推理能力、多模态理解能力、GUI操作能力和前端页面编程能力",
        "context_length": 256000,  # 256k
        "max_output": 32000,  # 32k
        "cost_info": {
            "input_0_32k": 0.0008,  # 输入长度[0,32k]时的输入价格(元/千token)
            "output_0_32k_short": 0.002,  # 输入[0,32k]且输出[0,0.2k]时的输出价格
            "output_0_32k_long": 0.008,  # 输入[0,32k]且输出>0.2k时的输出价格
            "input_32_128k": 0.0012,  # 输入长度(32k,128k]时的输入价格
            "output_32_128k": 0.016,  # 输入(32k,128k]时的输出价格
            "input_128_256k": 0.0024,  # 输入长度(128k,256k]时的输入价格
            "output_128_256k": 0.024,  # 输入(128k,256k]时的输出价格
        },
        "features": ["深度思考", "多模态理解", "GUI操作", "工具调用", "结构化输出"]
    },
    "doubao-seed-1.6-flash": {
        "display_name": "豆包 Seed 1.6 Flash",
        "model_name": "doubao-seed-1.6-flash",
        "description": "轻量版深度思考模型，极致推理速度，在轻量版语言模型中处于全球一流水平",
        "context_length": 256000,  # 256k
        "max_output": 32000,  # 32k
        "cost_info": {
            "input_0_32k": 0.00015,  # 输入长度[0,32k]时的输入价格
            "output_0_32k": 0.0015,  # 输入[0,32k]时的输出价格
            "input_32_128k": 0.0003,  # 输入长度(32k,128k]时的输入价格
            "output_32_128k": 0.003,  # 输入(32k,128k]时的输出价格
            "input_128_256k": 0.0006,  # 输入长度(128k,256k]时的输入价格
            "output_128_256k": 0.006,  # 输入(128k,256k]时的输出价格
        },
        "features": ["深度思考", "多模态理解", "视觉定位", "工具调用", "结构化输出"]
    },
    "doubao-seed-1.6-thinking": {
        "display_name": "豆包 Seed 1.6 Thinking",
        "model_name": "doubao-seed-1.6-thinking",
        "description": "专注深度思考的模型，在编程、数学、逻辑推理等基础能力上进一步提升",
        "context_length": 256000,  # 256k
        "max_output": 32000,  # 32k
        "cost_info": {
            "input_0_32k": 0.0008,  # 输入长度[0,32k]时的输入价格
            "output_0_32k": 0.008,  # 输入[0,32k]时的输出价格
            "input_32_128k": 0.0012,  # 输入长度(32k,128k]时的输入价格
            "output_32_128k": 0.016,  # 输入(32k,128k]时的输出价格
            "input_128_256k": 0.0024,  # 输入长度(128k,256k]时的输入价格
            "output_128_256k": 0.024,  # 输入(128k,256k]时的输出价格
        },
        "features": ["深度思考", "多模态理解", "视频理解", "工具调用", "结构化输出"]
    },
    "doubao-1.5-pro-32k": {
        "display_name": "豆包 1.5 Pro 32K",
        "model_name": "doubao-1.5-pro-32k",
        "description": "经典版本豆包模型，稳定可靠，适合生产环境使用",
        "context_length": 32000,  # 32k
        "max_output": 4000,  # 4k
        "cost_info": {
            "input": 0.0008,  # 输入价格(元/千token)
            "output": 0.002,  # 输出价格(元/千token)
        },
        "features": ["文本生成", "对话理解", "工具调用"]
    },
    "doubao-lite-32k": {
        "display_name": "豆包 Lite 32K",
        "model_name": "doubao-lite-32k",
        "description": "轻量级模型，成本更低，适合大规模调用场景",
        "context_length": 32000,  # 32k
        "max_output": 4000,  # 4k
        "cost_info": {
            "input": 0.0003,  # 输入价格(元/千token)
            "output": 0.0006,  # 输出价格(元/千token)
        },
        "features": ["文本生成", "对话理解"]
    }
}

# API配置
DOUBAO_API_CONFIG = {
    "base_url": "https://ark.cn-beijing.volces.com/api/v3",
    "timeout": 60,
    "max_retries": 3,
    "temperature_range": (0.0, 1.0),
    "top_p_range": (0.0, 1.0),
}

def get_doubao_model_info(model_name):
    """获取豆包模型信息"""
    return DOUBAO_MODELS.get(model_name, {})

def calculate_doubao_cost(model_name, input_tokens, output_tokens, input_length=None):
    """
    计算豆包模型调用成本
    
    Args:
        model_name: 模型名称
        input_tokens: 输入token数量
        output_tokens: 输出token数量
        input_length: 输入长度(用于分层定价模型)
    
    Returns:
        dict: 包含成本信息的字典
    """
    model_info = DOUBAO_MODELS.get(model_name)
    if not model_info:
        return {"error": "未知模型"}
    
    cost_info = model_info["cost_info"]
    
    # 处理分层定价模型(Seed系列)
    if "input_0_32k" in cost_info:
        if input_length is None:
            input_length = input_tokens  # 简化处理
        
        if input_length <= 32000:
            input_cost = input_tokens * cost_info["input_0_32k"] / 1000
            if model_name == "doubao-seed-1.6" and output_tokens <= 200:
                output_cost = output_tokens * cost_info["output_0_32k_short"] / 1000
            else:
                output_cost = output_tokens * cost_info.get("output_0_32k_long", cost_info.get("output_0_32k", 0)) / 1000
        elif input_length <= 128000:
            input_cost = input_tokens * cost_info["input_32_128k"] / 1000
            output_cost = output_tokens * cost_info["output_32_128k"] / 1000
        else:
            input_cost = input_tokens * cost_info["input_128_256k"] / 1000
            output_cost = output_tokens * cost_info["output_128_256k"] / 1000
    else:
        # 处理固定定价模型
        input_cost = input_tokens * cost_info["input"] / 1000
        output_cost = output_tokens * cost_info["output"] / 1000
    
    total_cost = input_cost + output_cost
    
    return {
        "input_cost": round(input_cost, 6),
        "output_cost": round(output_cost, 6),
        "total_cost": round(total_cost, 6),
        "currency": "CNY"
    }

def recommend_doubao_model(use_case="general"):
    """
    根据使用场景推荐豆包模型
    
    Args:
        use_case: 使用场景 ("general", "cost_sensitive", "reasoning", "multimodal", "production")
    
    Returns:
        str: 推荐的模型名称
    """
    recommendations = {
        "general": "doubao-seed-1.6",
        "cost_sensitive": "doubao-lite-32k", 
        "reasoning": "doubao-seed-1.6-thinking",
        "multimodal": "doubao-seed-1.6",
        "production": "doubao-1.5-pro-32k",
        "fast": "doubao-seed-1.6-flash"
    }
    return recommendations.get(use_case, "doubao-seed-1.6")

def validate_doubao_api_key(api_key):
    """
    验证豆包API Key格式
    
    Args:
        api_key: API密钥
    
    Returns:
        bool: 是否有效
    """
    if not api_key:
        return False
    
    # 豆包API Key通常是较长的字符串，包含字母数字和特殊字符
    if len(api_key) < 20:
        return False
    
    # 基本格式检查
    return True

def get_doubao_model_list():
    """获取豆包模型列表"""
    return [
        {
            "model_name": model_name,
            "display_name": info["display_name"],
            "description": info["description"],
            "context_length": info["context_length"],
            "max_output": info["max_output"],
            "cost_info": info["cost_info"],
            "features": info["features"]
        }
        for model_name, info in DOUBAO_MODELS.items()
    ]

def get_endpoint_id_info():
    """
    获取豆包推理接入点相关信息
    """
    return {
        "info": "豆包模型需要先创建推理接入点才能使用",
        "steps": [
            "1. 登录火山引擎控制台",
            "2. 进入火山方舟 > 模型推理",
            "3. 点击【创建推理接入点】",
            "4. 选择豆包模型和版本",
            "5. 创建后获得以 ep- 开头的接入点ID",
            "6. 使用接入点ID作为model_name参数"
        ],
        "example_endpoint": "ep-20240101000000-xxxxx",
        "note": "每个模型都需要单独创建接入点"
    }