from core.factory import AppFactory
from langchain_core.prompts import ChatPromptTemplate
from core.models import TestCases
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.runnables import RunnablePassthrough
import json
import re
import streamlit as st

def clean_and_parse_json(text):
    """
    清理 LLM 输出并提取 JSON 内容
    """
    # 移除可能的思考标签和其他非JSON内容
    text = re.sub(r'<think>.*?</think>', '', text, flags=re.DOTALL)
    text = re.sub(r'<.*?>', '', text)
    
    # 尝试找到JSON对象
    json_match = re.search(r'\{.*\}', text, flags=re.DOTALL)
    if json_match:
        json_str = json_match.group()
        try:
            # 尝试直接解析
            parsed = json.loads(json_str)
            
            # 检查是否需要修复格式
            if 'test_cases' in parsed and isinstance(parsed['test_cases'], list):
                # 检查数组元素是否为字符串格式的TestCase对象
                fixed_cases = []
                for case in parsed['test_cases']:
                    if isinstance(case, str) and case.startswith('TestCase('):
                        # 这是字符串格式的TestCase，需要解析
                        try:
                            # 提取TestCase中的参数
                            case_content = case[9:-1]  # 移除 'TestCase(' 和 ')'
                            case_dict = {}
                            
                            # 简单的参数解析
                            params = re.findall(r"(\w+)='([^']*)'", case_content)
                            for key, value in params:
                                case_dict[key] = value
                            
                            fixed_cases.append(case_dict)
                        except:
                            # 如果解析失败，跳过这个用例
                            continue
                    elif isinstance(case, dict):
                        fixed_cases.append(case)
                
                parsed['test_cases'] = fixed_cases
            
            return parsed
        except json.JSONDecodeError:
            pass
    
    # 如果找不到完整的JSON，尝试直接解析
    try:
        return json.loads(text.strip())
    except json.JSONDecodeError as e:
        raise ValueError(f"无法解析JSON输出: {str(e)}\n原始输出: {text[:500]}...")

def extract_thinking_content(text):
    """
    提取思考过程内容
    """
    # 查找 <think> 标签内容
    think_match = re.search(r'<think>(.*?)</think>', text, flags=re.DOTALL)
    if think_match:
        return think_match.group(1).strip()
    return None

def generate_test_cases_with_rag_and_think(model_settings, num_cases, requirement, kb_manager, enable_think=False, think_container=None):
    """
    生成测试用例，支持 think 功能和批次生成
    
    Args:
        model_settings: 模型配置
        num_cases: 用例数量
        requirement: 需求描述
        kb_manager: 知识库管理器
        enable_think: 是否启用思考模式
        think_container: Streamlit 容器，用于显示思考过程
    """
    # 动态确定批次大小
    BATCH_SIZE = get_optimal_batch_size(model_settings)
    
    # 如果用例数量超过批次大小，使用批次生成
    if num_cases > BATCH_SIZE:
        return generate_test_cases_in_batches(
            model_settings, num_cases, requirement, kb_manager, 
            enable_think, think_container
        )
    
    # 单批次生成
    return generate_single_batch(
        model_settings, num_cases, requirement, kb_manager, 
        enable_think, think_container, 1, []
    )

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

def generate_test_cases_in_batches(model_settings, num_cases, requirement, kb_manager, enable_think, think_container):
    """
    分批次生成测试用例，带去重机制
    """
    batch_size = get_optimal_batch_size(model_settings)  # 动态确定批次大小
    all_test_cases = []
    batches = []
    
    # 计算批次
    remaining_cases = num_cases
    batch_num = 1
    
    while remaining_cases > 0:
        current_batch_size = min(batch_size, remaining_cases)
        batches.append((batch_num, current_batch_size))
        remaining_cases -= current_batch_size
        batch_num += 1
    
    # 如果有思考容器，显示批次信息
    if think_container:
        with think_container:
            provider_name = model_settings.get('provider', 'Unknown')
            model_name = model_settings.get('model_name', 'Unknown')
            st.info(f"🤖 使用 {provider_name} - {model_name}")
            st.info(f"📦 检测到需要生成 {num_cases} 条用例，将分 {len(batches)} 个批次进行，每批次最多 {batch_size} 条")
            st.info("🔄 系统将自动避免生成重复的测试用例")
            
            # 创建批次进度条
            batch_progress = st.progress(0)
            batch_status = st.empty()
    
    # 逐批次生成
    for i, (batch_num, current_batch_size) in enumerate(batches):
        if think_container:
            batch_status.info(f"🔄 正在生成第 {batch_num} 批次 ({current_batch_size} 条用例)...")
            batch_progress.progress((i) / len(batches))
        
        try:
            # 为每个批次创建独立的思考容器
            batch_think_container = None
            if think_container and enable_think:
                with think_container:
                    with st.expander(f"🤔 第 {batch_num} 批次思考过程", expanded=False):
                        batch_think_container = st.container()
            
            # 生成当前批次（传递已生成的用例信息）
            batch_result = generate_single_batch(
                model_settings, current_batch_size, requirement, kb_manager,
                enable_think, batch_think_container, batch_num, all_test_cases
            )
            
            # 合并结果并去重
            if batch_result and hasattr(batch_result, 'test_cases'):
                # 去重处理
                new_cases = deduplicate_test_cases(batch_result.test_cases, all_test_cases)
                all_test_cases.extend(new_cases)
                
                if think_container:
                    original_count = len(batch_result.test_cases)
                    final_count = len(new_cases)
                    if original_count > final_count:
                        batch_status.warning(f"⚠️ 第 {batch_num} 批次去重：原生成 {original_count} 条，去重后 {final_count} 条")
                    else:
                        batch_status.success(f"✅ 第 {batch_num} 批次完成，生成了 {final_count} 条用例")
            else:
                if think_container:
                    batch_status.error(f"❌ 第 {batch_num} 批次生成失败")
                
        except Exception as e:
            if think_container:
                batch_status.error(f"❌ 第 {batch_num} 批次生成失败: {str(e)}")
            continue
    
    # 完成所有批次
    if think_container:
        batch_progress.progress(1.0)
        batch_status.success(f"🎉 所有批次完成！总共生成了 {len(all_test_cases)} 条测试用例")
    
    # 返回合并后的结果，并清理nan值
    from core.models import TestCases, TestCase
    # 确保all_test_cases中的每个元素都是TestCase实例，并清理nan值
    processed_cases = []
    for case in all_test_cases:
        # 获取字典形式的数据
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
        
        # 创建清理后的TestCase实例
        processed_cases.append(TestCase(**cleaned_case_dict))
    
    return TestCases(test_cases=processed_cases)

def deduplicate_test_cases(new_cases, existing_cases):
    """
    去重函数：移除与已有用例重复的测试用例
    """
    if not existing_cases:
        return new_cases
    
    unique_cases = []
    existing_titles = {case.用例名称.strip().lower() for case in existing_cases}
    existing_descriptions = {case.步骤描述.strip().lower() for case in existing_cases}
    
    for case in new_cases:
        title_lower = case.用例名称.strip().lower()
        desc_lower = case.步骤描述.strip().lower()
        
        # 检查标题和描述是否重复
        is_duplicate = (
            title_lower in existing_titles or
            desc_lower in existing_descriptions or
            any(similarity_check(title_lower, existing_title) > 0.8 for existing_title in existing_titles) or
            any(similarity_check(desc_lower, existing_desc) > 0.8 for existing_desc in existing_descriptions)
        )
        
        if not is_duplicate:
            unique_cases.append(case)
            existing_titles.add(title_lower)
            existing_descriptions.add(desc_lower)
    
    return unique_cases

def similarity_check(text1, text2):
    """
    简单的文本相似度检查
    """
    if not text1 or not text2:
        return 0
    
    # 简单的字符级相似度
    set1 = set(text1)
    set2 = set(text2)
    intersection = len(set1.intersection(set2))
    union = len(set1.union(set2))
    
    return intersection / union if union > 0 else 0

def generate_single_batch(model_settings, num_cases, requirement, kb_manager, enable_think, think_container, batch_num=1, existing_cases=None):
    """
    生成单个批次的测试用例
    """
    llm_integrator = AppFactory.create_llm_integrator(
        model_settings['provider'],
        model_settings.get('model_name', 'qwen3:4b'),
        model_settings['api_key'],
        model_settings['base_url']
    )
    
    # 获取 LLM，如果是 Qwen3 模型且启用思考模式，则添加相应配置
    llm = llm_integrator.get_llm()
    
    # 检查是否是 Qwen3 模型
    is_qwen3 = 'qwen3' in model_settings.get('model_name', '').lower()
    
    retriever = kb_manager.get_retriever()
    parser = JsonOutputParser(pydantic_object=TestCases)

    # 构建提示词，根据是否启用思考模式调整
    think_instruction = ""
    if enable_think and is_qwen3:
        think_instruction = "/think\n"
    elif not enable_think and is_qwen3:
        think_instruction = "/no_think\n"

    # 构建已生成用例的摘要信息
    existing_info = ""
    if existing_cases and len(existing_cases) > 0:
        existing_titles = [case.用例名称 for case in existing_cases[-10:]]  # 只显示最近10个
        existing_info = f"""
【已生成的测试用例摘要】（第{batch_num}批次，请避免重复）
前面批次已生成的用例标题：
{chr(10).join(f"- {title}" for title in existing_titles)}

【重要】请确保本批次生成的测试用例与上述已有用例不重复，要有不同的测试角度和场景。
"""

    template = f"""{think_instruction}你是一个专业的软件测试工程师。请根据以下需求描述和上下文信息，生成 {{num_cases}} 个结构化的高质量测试用例。

【重要指令】
1. 优先使用提供的上下文信息来生成更贴切的测试用例。
2. 输出必须是一个完整的JSON对象，包含一个名为 'test_cases' 的键，其值是测试用例列表。
3. 每个测试用例都必须包含所有必需字段。
4. 请使用中文生成测试用例内容。
5. 最终输出只包含JSON格式的测试用例，不要包含其他解释文本。
6. 确保测试用例具有多样性，覆盖不同的测试场景和边界条件。
7. 所有字段都必须有值，不能为null、undefined或空值，如果某个字段暂时没有内容，请使用空字符串""
8. 确保JSON格式正确，所有字符串字段都用双引号包围

【上下文信息】
{{context}}

【需求描述】
{{requirement}}

{existing_info}

{{format_instructions}}

请直接输出有效的JSON格式：
"""
    
    prompt = ChatPromptTemplate.from_template(
        template,
        partial_variables={"format_instructions": parser.get_format_instructions()}
    )

    # 创建链
    rag_chain = (
        {"context": retriever, "requirement": RunnablePassthrough(), "num_cases": RunnablePassthrough()}
        | prompt
        | llm
    )

    # 如果启用思考模式且提供了容器，使用流式输出
    if enable_think and is_qwen3 and think_container:
        return generate_with_streaming_think(rag_chain, requirement, num_cases, think_container)
    else:
        # 标准生成模式
        return generate_standard(rag_chain, requirement, num_cases)

def generate_with_streaming_think(rag_chain, requirement, num_cases, think_container):
    """
    带有流式思考过程的生成
    """
    try:
        # 获取流式输出
        full_response = ""
        thinking_content = ""
        json_content = ""
        in_thinking = False
        
        with think_container:
            # 创建一个可展开的思考过程区域
            with st.expander("🤔 AI 思考过程", expanded=True):
                st.info("💡 AI 正在分析您的需求，思考过程将实时显示...")
                
                thinking_placeholder = st.empty()
                progress_bar = st.progress(0)
                
            # 调用链并处理流式输出
            chunk_count = 0
            for chunk in rag_chain.stream({"requirement": requirement, "num_cases": num_cases}):
                if hasattr(chunk, 'content'):
                    content = chunk.content
                    full_response += content
                    chunk_count += 1
                    
                    # 更新进度条（估算）
                    progress = min(chunk_count * 0.02, 0.9)  # 最多到90%
                    progress_bar.progress(progress)
                    
                    # 检查是否进入思考模式
                    if '<think>' in content:
                        in_thinking = True
                    
                    if in_thinking:
                        thinking_content += content
                        # 实时显示思考过程 - 使用简洁的格式
                        clean_thinking = re.sub(r'</?think>', '', thinking_content).strip()
                        if clean_thinking:
                            thinking_placeholder.markdown(
                                f"""
                                <div style="
                                    background-color: #f8f9fa;
                                    border: 1px solid #e9ecef;
                                    border-left: 4px solid #28a745;
                                    padding: 20px;
                                    border-radius: 8px;
                                    margin: 15px 0;
                                    font-family: 'SF Mono', 'Monaco', 'Inconsolata', 'Roboto Mono', monospace;
                                    white-space: pre-wrap;
                                    word-wrap: break-word;
                                    max-height: 500px;
                                    overflow-y: auto;
                                    line-height: 1.6;
                                    font-size: 14px;
                                    color: #2c3e50;
                                    box-shadow: 0 2px 4px rgba(0,0,0,0.1);
                                ">
                                <div style="
                                    color: #28a745;
                                    font-weight: bold;
                                    margin-bottom: 10px;
                                    border-bottom: 1px solid #e9ecef;
                                    padding-bottom: 8px;
                                ">
                                🧠 思考中...
                                </div>
                                <div>
                                {clean_thinking}
                                </div>
                                </div>
                                """,
                                unsafe_allow_html=True
                            )
                    
                    # 检查是否退出思考模式
                    if '</think>' in content:
                        in_thinking = False
                        progress_bar.progress(1.0)  # 完成
                        json_content = full_response.split('</think>')[-1] if '</think>' in full_response else ""
        
        # 解析最终结果
        if json_content.strip():
            parsed_data = clean_and_parse_json(json_content)
        else:
            parsed_data = clean_and_parse_json(full_response)
            
        return TestCases(**parsed_data)
        
    except Exception as e:
        st.error(f"生成过程中出现错误: {str(e)}")
        # 回退到标准模式
        return generate_standard(rag_chain, requirement, num_cases)

def generate_standard(rag_chain, requirement, num_cases):
    """
    标准生成模式
    """
    # 获取原始输出并手动解析
    raw_output = rag_chain.invoke({"requirement": requirement, "num_cases": num_cases})
    
    # 提取文本内容
    if hasattr(raw_output, 'content'):
        text_output = raw_output.content
    else:
        text_output = str(raw_output)
    
    # 使用自定义解析器
    try:
        parsed_data = clean_and_parse_json(text_output)
        return TestCases(**parsed_data)
    except Exception as e:
        print(f"解析错误: {str(e)}")
        print(f"原始输出: {text_output[:1000]}...")
        raise

# 保持向后兼容性
def generate_test_cases_with_rag(model_settings, num_cases, requirement, kb_manager):
    """
    原有的生成函数，保持向后兼容
    """
    return generate_test_cases_with_rag_and_think(
        model_settings, num_cases, requirement, kb_manager, enable_think=False
    )