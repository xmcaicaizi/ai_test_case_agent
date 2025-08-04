from core.factory import AppFactory
from langchain_core.prompts import ChatPromptTemplate
from core.models import TestCases
from langchain_core.output_parsers import JsonOutputParser
import streamlit as st

def get_optimal_batch_size(model_settings):
    """
    根据模型提供商和模型名称确定最优批次大小
    """
    provider = model_settings.get('provider', '').lower()
    model_name = model_settings.get('model_name', '').lower()
    
    # Ollama 本地模型 - 较小的批次大小（受本地计算资源限制）
    if provider == 'ollama':
        if 'qwen3:4b' in model_name or 'qwen3:8b' in model_name:
            return 5  # 小模型，保持较小批次
        elif any(size in model_name for size in ['4b', '7b', '8b']):
            return 8  # 中等模型
        else:
            return 10  # 大模型
    
    # 远程API调用 - 根据模型上下文能力适度放开限制（考虑实际性能）
    elif provider in ['openaicompatible', 'openai', 'gemini', 'qwen', 'doubao']:
        # Qwen系列：支持1M上下文，但考虑实际生成速度
        if 'qwen' in model_name:
            return 20  # Qwen系列适度增加批次，避免超时
        
        # Doubao系列：支持256K上下文，适度增加
        elif 'doubao' in model_name or provider == 'doubao':
            return 15  # Doubao系列适度增加批次
        
        # GPT-4系列：支持128K上下文
        elif 'gpt-4' in model_name:
            return 15  # GPT-4适度增加批次
        
        # Claude系列：支持200K上下文
        elif 'claude' in model_name:
            return 15  # Claude适度增加批次
        
        # GPT-3.5和其他模型：标准上下文
        elif 'gpt-3.5' in model_name:
            return 12  # GPT-3.5标准批次
        
        else:
            return 10  # 其他远程模型默认批次
    
    # 默认值（保守估计）
    return 8

def generate_test_cases_with_llm(model_settings, num_cases, requirement):
    """
    使用LLM生成测试用例，支持分批次生成
    """
    BATCH_SIZE = get_optimal_batch_size(model_settings)  # 动态确定批次大小
    
    # 如果测试用例数量较少，直接生成
    if num_cases <= BATCH_SIZE:
        return _generate_single_batch_llm(model_settings, num_cases, requirement)
    
    # 分批次生成
    provider_name = model_settings.get('provider', 'Unknown')
    model_name = model_settings.get('model_name', 'Unknown')
    st.info(f"🤖 使用 {provider_name} - {model_name}")
    st.info(f"📦 将分 {(num_cases + BATCH_SIZE - 1) // BATCH_SIZE} 个批次生成测试用例（每批次 {BATCH_SIZE} 条）")
    
    all_test_cases = []
    remaining_cases = num_cases
    batch_num = 1
    
    # 创建进度条
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    while remaining_cases > 0:
        current_batch_size = min(BATCH_SIZE, remaining_cases)
        
        status_text.text(f"正在生成第 {batch_num} 批次 ({current_batch_size} 个测试用例)...")
        
        try:
            # 生成当前批次
            batch_result = _generate_single_batch_llm(
                model_settings, current_batch_size, requirement, 
                batch_num=batch_num, existing_cases=all_test_cases
            )
            
            if hasattr(batch_result, 'test_cases'):
                batch_cases = batch_result.test_cases
            elif isinstance(batch_result, dict) and 'test_cases' in batch_result:
                batch_cases = batch_result['test_cases']
            else:
                batch_cases = []
            
            all_test_cases.extend(batch_cases)
            remaining_cases -= current_batch_size
            
            # 更新进度
            progress = (num_cases - remaining_cases) / num_cases
            progress_bar.progress(progress)
            
            st.success(f"✅ 第 {batch_num} 批次完成，生成了 {len(batch_cases)} 个测试用例")
            
        except Exception as e:
            st.error(f"❌ 第 {batch_num} 批次生成失败: {str(e)}")
            # 继续下一批次
        
        batch_num += 1
    
    status_text.text("生成完成！")
    progress_bar.progress(1.0)
    
    # 清理nan值并返回结果
    cleaned_test_cases = []
    for case in all_test_cases:
        if hasattr(case, 'model_dump'):
            case_dict = case.model_dump()
        elif hasattr(case, 'dict'):
            case_dict = case.dict()
        else:
            case_dict = case
        
        # 清理字典中的nan值
        cleaned_case_dict = {}
        for key, value in case_dict.items():
            if value is None or str(value).lower() in ['nan', 'none', 'null']:
                cleaned_case_dict[key] = ""
            else:
                cleaned_case_dict[key] = str(value) if value is not None else ""
        
        # 重新创建TestCase对象
        from core.models import TestCase
        cleaned_test_cases.append(TestCase(**cleaned_case_dict))
    
    return TestCases(test_cases=cleaned_test_cases)

def _generate_single_batch_llm(model_settings, num_cases, requirement, batch_num=1, existing_cases=None):
    """
    生成单个批次的测试用例
    """
    try:
        # 显示调试信息
        provider = model_settings.get('provider', 'Unknown')
        model_name = model_settings.get('model_name', 'Unknown')
        
        if batch_num == 1:  # 只在第一批次显示详细信息
            st.info(f"🔧 正在初始化 {provider} - {model_name}")
        
        llm_integrator = AppFactory.create_llm_integrator(
            model_settings['provider'],
            model_settings.get('model_name', 'qwen3:4b'),
            model_settings['api_key'],
            model_settings['base_url']
        )
        llm = llm_integrator.get_llm()

        parser = JsonOutputParser(pydantic_object=TestCases)
        
    except Exception as e:
        st.error(f"❌ 初始化LLM失败: {str(e)}")
        raise

    # 如果有已存在的测试用例，添加去重指令
    existing_cases_instruction = ""
    if existing_cases and len(existing_cases) > 0:
        existing_cases_summary = []
        for i, case in enumerate(existing_cases[-10:], 1):  # 只显示最近10个用例
            if hasattr(case, 'dict'):
                case_dict = case.dict()
            elif hasattr(case, 'model_dump'):
                case_dict = case.model_dump()
            else:
                case_dict = case
            
            title = case_dict.get('测试用例标题', case_dict.get('title', f'用例{i}'))
            existing_cases_summary.append(f"{i}. {title}")
        
        existing_cases_instruction = f"""
【已有测试用例】
{chr(10).join(existing_cases_summary)}

【重要】请确保本批次生成的测试用例与上述已有用例不重复，要有不同的测试角度和场景。
"""

    template = f"""你是一个专业的软件测试工程师。请根据以下需求描述生成 {{num_cases}} 个结构化的高质量测试用例。

【重要指令】
1. 输出必须是一个完整的JSON对象，包含一个名为 'test_cases' 的键，其值是测试用例列表
2. 每个测试用例都必须包含所有必需字段
3. 请使用中文生成测试用例内容
4. 所有字段都必须有值，不能为null、undefined或空值，如果某个字段暂时没有内容，请使用空字符串""
5. 确保JSON格式正确，所有字符串字段都用双引号包围
{existing_cases_instruction}

需求描述: {{requirement}}

{{format_instructions}}

请确保输出是有效的JSON格式：
"""
    
    prompt = ChatPromptTemplate.from_template(
        template,
        partial_variables={"format_instructions": parser.get_format_instructions()}
    )

    chain = prompt | llm | parser
    
    try:
        # 显示正在调用API的信息
        if batch_num == 1:
            st.info(f"🚀 正在调用API生成 {num_cases} 条测试用例...")
        
        result = chain.invoke({"num_cases": num_cases, "requirement": requirement})
        
        if batch_num == 1:
            st.success(f"✅ API调用成功")
            
        return result
        
    except Exception as e:
        error_msg = str(e)
        st.error(f"❌ API调用失败: {error_msg}")
        
        # 提供更详细的错误信息
        if "timeout" in error_msg.lower():
            st.warning("⏰ 可能是网络超时，请检查网络连接或稍后重试")
        elif "api" in error_msg.lower() or "key" in error_msg.lower():
            st.warning("🔑 可能是API密钥问题，请检查配置")
        elif "rate" in error_msg.lower() or "limit" in error_msg.lower():
            st.warning("🚦 可能触发了API速率限制，请稍后重试")
        
        raise