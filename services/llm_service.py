from core.factory import AppFactory
from langchain_core.prompts import ChatPromptTemplate
from core.models import TestCases
from langchain_core.output_parsers import JsonOutputParser

def generate_test_cases_with_llm(model_settings, num_cases, requirement):
    llm_integrator = AppFactory.create_llm_integrator(
        model_settings['provider'],
        model_settings.get('model_name', 'qwen3:4b'),
        model_settings['api_key'],
        model_settings['base_url']
    )
    llm = llm_integrator.get_llm()

    parser = JsonOutputParser(pydantic_object=TestCases)

    template = """你是一个专业的软件测试工程师。请根据以下需求描述生成 {num_cases} 个结构化的高质量测试用例。

【重要指令】
1. 输出必须是一个完整的JSON对象，包含一个名为 'test_cases' 的键，其值是测试用例列表
2. 每个测试用例都必须包含所有必需字段
3. 请使用中文生成测试用例内容

需求描述: {requirement}

{format_instructions}

请确保输出是有效的JSON格式：
"""
    prompt = ChatPromptTemplate.from_template(
        template,
        partial_variables={"format_instructions": parser.get_format_instructions()}
    )

    chain = prompt | llm | parser
    return chain.invoke({"num_cases": num_cases, "requirement": requirement})