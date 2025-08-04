# core/rag_chain.py
"""
该文件构建并封装了完整的RAG链。
"""
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.runnables import RunnableLambda
import re
from .models import TestCases

def create_rag_chain(retriever, llm, num_cases=3):
    """
    创建并返回一个RAG链，用于根据用户需求和知识库生成测试用例。

    :param retriever: 从知识库检索相关文档的Retriever。
    :param llm: 用于生成内容的语言模型。
    :param num_cases: 希望生成的测试用例数量。
    :return: 一个可执行的LangChain RAG链。
    """
    
    # 1. 定义Prompt模板
    prompt_template = """
    你是一个专业的软件测试工程师。你的任务是根据用户提供的需求和下文中【知识库检索到的相关信息】，生成 {num_cases} 个结构化的高质量测试用例。

    【重要指令】
    1.  **强制使用知识库**: 你必须仔细分析【知识库检索到的相关信息】。这些信息是生成测试用例的核心依据。用例中的模块划分、功能点、前置条件和特定术语都应尽可能地来源于此。
    2.  **严格遵循JSON格式**: 输出必须是一个完整的JSON对象。该对象的核心是一个名为 'test_cases' 的键，其值是一个测试用例列表。
    3.  **完整的字段**: 每个测试用例都必须包含以下所有字段，不得遗漏或增删：'ID', '一级模块', '二级模块', '三级模块', '四级模块', '五级模块', '六级模块', '七级模块', '八级模块', '九级模块', '用例名称', '优先级', '用例类型', '前置条件', '步骤描述', '预期结果', '备注', '维护人', '测试方式', '创建版本', '更新版本', '评估是否可实现自动化', '是否重新执行', 'Summary', '所属产品', '预估执行时间_h'。
    4.  **内容生成**: 
        -   **模块字段**: 根据【知识库检索到的相关信息】和【用户需求】来决定模块层级。如果信息不足，可以留空或填写通用值。
        -   **用例名称/步骤/结果**: 必须与用户需求和知识库内容紧密相关。
        -   **其他字段**: 如无特殊信息，可使用合理的默认值（例如，优先级: '中', 测试方式: '手动'）。

    --- 
    【知识库检索到的相关信息】
    {context}
    --- 
    【用户需求】
    {user_requirement}
    --- 
    【生成的测试用例 (JSON格式)】:
    """
    
    prompt = PromptTemplate.from_template(prompt_template)

    # 2. 定义输出解析器
    output_parser = JsonOutputParser(pydantic_object=TestCases)
    
    # 3. 构建RAG链
    def format_docs(docs):
        # 将检索到的文档内容格式化为字符串
        return "\n\n".join(doc.page_content for doc in docs)

    def extract_json_from_think_tags(message):
        # 处理 AIMessage 对象，提取其 content 属性
        if hasattr(message, 'content'):
            text = message.content
        else:
            text = str(message)
        
        match = re.search(r'<think>.*?</think>\s*({.*})', text, re.DOTALL)
        if match:
            return match.group(1)
        return text # Fallback if no think tags are found

    rag_chain = (
        {"context": retriever | format_docs, "user_requirement": RunnablePassthrough(), "num_cases": lambda x: num_cases}
        | prompt
        | llm
        | RunnableLambda(extract_json_from_think_tags)
        | output_parser
    )
    
    return rag_chain

# Example usage (for testing purposes)
if __name__ == '__main__':
    # This is a mock setup for demonstrating the chain structure.
    # To run this, you would need actual instances of a retriever and an llm.
    
    from .factory import AppFactory
    import os

    print("--- RAG Chain Test ---")
    
    # 1. 设置测试环境
    KB_DIR = "temp_kb_for_rag"
    DB_DIR = "temp_db_for_rag"
    if not os.path.exists(KB_DIR):
        os.makedirs(KB_DIR)
        
    with open(os.path.join(KB_DIR, "login_spec.txt"), "w", encoding='utf-8') as f:
        f.write("系统登录功能规格说明：\n1. 用户需使用已注册的邮箱和密码进行登录。\n2. 登录成功后，应跳转到用户仪表盘页面。\n3. 如果密码错误，应提示'用户名或密码不正确'。\n4. 连续5次密码错误，账户将被临时锁定15分钟。")

    # 2. 初始化知识库和LLM
    try:
        kb_manager = AppFactory.create_kb_manager(KB_DIR, DB_DIR, 'nomic-embed-text')
        kb_manager.load_and_process_documents()
        retriever = kb_manager.get_retriever()

        # 使用Ollama进行测试，请确保Ollama服务正在运行
        llm_integrator = AppFactory.create_llm_integrator('Ollama', 'qwen3:4b')
        llm = llm_integrator.get_llm()

        # 3. 创建RAG链
        rag_chain = AppFactory.create_rag_chain(retriever, llm, 2)
        print("RAG Chain created successfully.")
        
        # 4. 调用链并打印结果
        user_requirement = "请为用户登录功能设计测试用例"
        print(f"\nInvoking chain with user requirement: '{user_requirement}'")
        
        try:
            generated_test_cases = rag_chain.invoke(user_requirement)
            print("\n--- Generated Test Cases ---")
            import json
            print(json.dumps(generated_test_cases, indent=2, ensure_ascii=False))
            print("\nTest finished successfully.")
        except Exception as e:
            print(f"An error occurred during chain invocation: {e}")
            print("Please ensure your Ollama instance is running and the model 'qwen:0.5b-chat-v1.5-q4_0' is available.")

    except Exception as e:
        print(f"An error occurred during setup: {e}")
    
    finally:
        # 5. 清理测试文件
        import shutil
        if os.path.exists(KB_DIR):
            shutil.rmtree(KB_DIR)
        if os.path.exists(DB_DIR):
            shutil.rmtree(DB_DIR)
        print("\nCleanup complete.")
