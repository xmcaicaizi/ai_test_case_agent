# core/rag_chain.py
"""
该文件构建并封装了完整的RAG链。
"""
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import JsonOutputParser
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
    你是一个专业的软件测试工程师。你的任务是根据用户提供的需求和相关的知识库信息，生成 {num_cases} 个结构化的高质量测试用例。
    请严格按照我提供的JSON格式输出。JSON对象应该有一个名为 'test_cases' 的键，其值是一个包含所有测试用例的列表。
    每个测试用例都必须包含以下字段: 'title', 'preconditions', 'steps', 'expected_result', 'priority'。

    ---
    【知识库检索到的相关信息】
    {context}
    ---
    【用户需求】
    {user_requirement}
    ---
    【生成的测试用例 (JSON格式)】:
    """
    
    prompt = ChatPromptTemplate.from_template(prompt_template)

    # 2. 定义输出解析器
    output_parser = JsonOutputParser(pydantic_object=TestCases)
    
    # 3. 构建RAG链
    def format_docs(docs):
        # 将检索到的文档内容格式化为字符串
        return "\n\n".join(doc.page_content for doc in docs)

    rag_chain = (
        {"context": retriever | format_docs, "user_requirement": RunnablePassthrough(), "num_cases": lambda x: num_cases}
        | prompt
        | llm
        | output_parser
    )
    
    return rag_chain

# Example usage (for testing purposes)
if __name__ == '__main__':
    # This is a mock setup for demonstrating the chain structure.
    # To run this, you would need actual instances of a retriever and an llm.
    
    from .llm_integrator import LLMIntegrator
    from .knowledge_base import KnowledgeBaseManager
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
        kb_manager = KnowledgeBaseManager(knowledge_base_dir=KB_DIR, chroma_db_dir=DB_DIR)
        kb_manager.load_and_process_documents()
        retriever = kb_manager.get_retriever()

        # 使用Ollama进行测试，请确保Ollama服务正在运行
        llm_integrator = LLMIntegrator(model_provider='Ollama', model_name='qwen:0.5b-chat-v1.5-q4_0')
        llm = llm_integrator.get_llm()

        # 3. 创建RAG链
        rag_chain = create_rag_chain(retriever, llm, num_cases=2)
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
