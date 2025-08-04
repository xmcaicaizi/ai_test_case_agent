from core.factory import AppFactory

from langchain_core.prompts import ChatPromptTemplate
from core.models import TestCases
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.runnables import RunnablePassthrough

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

【上下文信息】
{context}

【需求描述】
{requirement}

{format_instructions}

请确保输出是有效的JSON格式：
"""
    prompt = ChatPromptTemplate.from_template(
        template,
        partial_variables={"format_instructions": parser.get_format_instructions()}
    )

    rag_chain = (
        {"context": retriever, "requirement": RunnablePassthrough(), "num_cases": RunnablePassthrough()}
        | prompt
        | llm
        | parser
    )

    return rag_chain.invoke({"requirement": requirement, "num_cases": num_cases})