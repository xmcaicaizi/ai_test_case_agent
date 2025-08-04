from core.factory import AppFactory

from langchain_core.prompts import ChatPromptTemplate
from core.models import TestCases
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.runnables import RunnablePassthrough
import json
import re

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

def generate_test_cases_with_rag(model_settings, num_cases, requirement, kb_manager):
    llm_integrator = AppFactory.create_llm_integrator(
        model_settings['provider'],
        model_settings.get('model_name', 'qwen3:4b'),
        model_settings['api_key'],
        model_settings['base_url']
    )
    llm = llm_integrator.get_llm()
    retriever = kb_manager.get_retriever()

    parser = JsonOutputParser(pydantic_object=TestCases)

    template = """你是一个专业的软件测试工程师。请根据以下需求描述和上下文信息，生成 {num_cases} 个结构化的高质量测试用例。

【重要指令】
1. 优先使用提供的上下文信息来生成更贴切的测试用例。
2. 输出必须是一个完整的JSON对象，包含一个名为 'test_cases' 的键，其值是测试用例列表。
3. 每个测试用例都必须包含所有必需字段。
4. 请使用中文生成测试用例内容。
5. 不要包含任何思考过程、解释或其他文本，只输出纯JSON格式。
6. 不要使用 <think> 标签或任何其他标签。

【上下文信息】
{context}

【需求描述】
{requirement}

{format_instructions}

请直接输出有效的JSON格式，不要包含任何其他内容：
"""
    prompt = ChatPromptTemplate.from_template(
        template,
        partial_variables={"format_instructions": parser.get_format_instructions()}
    )

    # 创建不使用解析器的链，我们手动处理输出
    rag_chain = (
        {"context": retriever, "requirement": RunnablePassthrough(), "num_cases": RunnablePassthrough()}
        | prompt
        | llm
    )

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