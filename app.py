# app.py
"""
Streamlit 主应用文件
"""

import streamlit as st
import os
import pandas as pd
import io
from core.knowledge_base import KnowledgeBaseManager
from core.llm_integrator import LLMIntegrator
from core.rag_chain import create_rag_chain

# --- 页面配置 ---
st.set_page_config(
    page_title="AI-TGA 智能测试用例生成平台",
    page_icon="🧪",
    layout="wide",
    initial_sidebar_state="expanded",
)

# --- 主标题 ---
st.title("🤖 AI-TGA: 智能测试用例生成平台")





# --- 初始化 session state ---
# 用于在页面刷新间保持状态
if 'model_settings' not in st.session_state:
    st.session_state.model_settings = {
        'provider': 'Ollama',
        'api_key': '',
        'base_url': 'http://127.0.0.1:11434'
    }

# --- 常量定义 ---
KNOWLEDGE_BASE_DIR = "knowledge_files"
CHROMA_DB_DIR = os.path.join("db", "chroma_db")

# 确保目录存在
os.makedirs(KNOWLEDGE_BASE_DIR, exist_ok=True)
os.makedirs(CHROMA_DB_DIR, exist_ok=True)


# --- 核心逻辑函数 ---
@st.cache_resource
def get_kb_manager():
    """缓存知识库管理器实例"""
    return KnowledgeBaseManager(knowledge_base_dir=KNOWLEDGE_BASE_DIR, chroma_db_dir=CHROMA_DB_DIR)

# --- 清理函数 ---
def cleanup_directories():
    """清理知识库和数据库目录"""
    import shutil
    # 停止并等待文件释放
    if os.path.exists(KNOWLEDGE_BASE_DIR):
        shutil.rmtree(KNOWLEDGE_BASE_DIR, ignore_errors=True)
    if os.path.exists(CHROMA_DB_DIR):
        shutil.rmtree(CHROMA_DB_DIR, ignore_errors=True)
    
    # 短暂等待以确保文件句柄被释放
    import time
    time.sleep(1)

    os.makedirs(KNOWLEDGE_BASE_DIR, exist_ok=True)
    os.makedirs(CHROMA_DB_DIR, exist_ok=True)
    
    # 重置缓存和状态
    st.cache_resource.clear()
    st.session_state.knowledge_base_status = {
        "status": "已重置",
        "doc_count": 0,
        "chunk_count": 0,
    }

if 'generated_cases' not in st.session_state:
    st.session_state.generated_cases = None

if 'knowledge_base_status' not in st.session_state:
    kb_manager = get_kb_manager()
    status = kb_manager.get_status()
    st.session_state.knowledge_base_status = {
        "status": "已加载",
        "doc_count": status['doc_count'],
        "chunk_count": status['chunk_count'],
    }

# --- 创建主页面标签 ---
tab_generate, tab_kb, tab_settings = st.tabs([
    "📄 生成测试用例", 
    "📚 知识库管理", 
    "⚙️ 模型设置"
])


# --- 页面一：生成测试用例 ---
with tab_generate:
    st.header("在这里根据您的需求生成测试用例")
    
    # --- 侧边栏 ---
    with st.sidebar:
        st.header("生成参数配置")
        
        user_requirement = st.text_area("1. 输入您的需求描述", height=200, placeholder="例如：设计一个用户登录功能的测试用例，需要考虑正常登录、异常密码、锁定策略等场景。")
        
        uploaded_file = st.file_uploader("2. (可选) 上传需求文档", type=['txt', 'pdf', 'docx', 'xlsx'])
        
        st.write("---")
        
        num_cases = st.slider("3. 生成最大用例数量", min_value=1, max_value=20, value=5)
        
        temperature = st.slider("4. AI 创造性 (Temperature)", min_value=0.0, max_value=1.0, value=0.7, step=0.1, help="值越高，生成结果越随机和有创意；值越低，结果越确定和保守。")
        
        use_kb = st.checkbox("5. 使用知识库增强生成", value=True)
        
        st.write("---")
        
        # 从 session_state 获取当前模型用于显示
        current_model_provider = st.session_state.model_settings.get('provider', 'N/A')
        st.info(f"当前LLM: **{st.session_state.model_settings.get('model_name', 'qwen3:4b')}**")
        st.info(f"当前Embedding: **dengcao/Qwen3-Embedding-0.6B:Q8_0**")

        if st.button("🚀 生成测试用例", type="primary", use_container_width=True):
            if not user_requirement and not uploaded_file:
                st.warning("请输入需求描述或上传需求文档。")
            else:
                with st.spinner("正在初始化模型和知识库..."):
                    try:
                        # 1. 初始化LLM
                        model_settings = st.session_state.model_settings
                        llm_integrator = LLMIntegrator(
                            model_provider=model_settings['provider'],
                            model_name=st.session_state.model_settings.get('model_name', 'qwen3:4b'),
                            api_key=model_settings['api_key'],
                            base_url=model_settings['base_url']
                        )
                        llm = llm_integrator.get_llm()

                        # 2. 初始化Retriever
                        kb_manager = get_kb_manager()
                        retriever = kb_manager.get_retriever()
                        
                        # 3. 处理用户输入
                        full_requirement = user_requirement
                        if uploaded_file:
                            # 读取上传的临时文件内容
                            from io import StringIO
                            stringio = StringIO(uploaded_file.getvalue().decode("utf-8"))
                            full_requirement += "\n\n--- [附加文档内容] ---\n" + stringio.read()
                        
                        st.info("模型和知识库已就绪，开始生成测试用例...")

                        # 4. 创建并调用RAG链
                        rag_chain = create_rag_chain(retriever, llm, num_cases=num_cases)
                        
                        # 在新的spinner中执行invoke
                        with st.spinner("🧠 AI 正在思考中，请稍候..."):
                            response = rag_chain.invoke(full_requirement)
                            st.session_state.generated_cases = response
                            st.rerun()

                    except Exception as e:
                        st.error(f"生成过程中发生错误: {e}")
                        # 可以在这里添加更详细的错误诊断，比如检查Ollama服务是否在运行

    # --- 右侧主区域 ---
    st.subheader("生成结果")

    if st.session_state.generated_cases:
        # 这里将用于显示生成结果的视图
        st.success("测试用例已生成！")
        
        results = st.session_state.generated_cases
        
        # 将Pydantic模型转换为Pandas DataFrame
        try:
            test_cases_data = results.get('test_cases', [])
            if test_cases_data:
                df = pd.DataFrame([case for case in test_cases_data])
                
                # 创建结果展示的Tabs
                result_tabs = st.tabs(["表格视图", "JSON 数据", "导出文件"])
                
                with result_tabs[0]:
                    st.dataframe(df, use_container_width=True)
                
                with result_tabs[1]:
                    st.json(results)
                    
                with result_tabs[2]:
                    # 导出为Excel
                    excel_buffer = io.BytesIO()
                    df.to_excel(excel_buffer, index=False, engine='openpyxl')
                    st.download_button(
                        label="下载为 Excel",
                        data=excel_buffer,
                        file_name="test_cases.xlsx",
                        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                    )

                    # 导出为JSON
                    st.download_button(
                        label="下载为 JSON",
                        data=pd.io.json.dumps(results, indent=2),
                        file_name="test_cases.json",
                        mime="application/json",
                    )
            else:
                st.warning("AI模型返回了空结果，请尝试调整您的需求或模型参数。")
                st.json(results) # 显示原始返回内容以便调试

        except Exception as e:
            st.error("解析返回结果时出错，请检查LLM的输出格式是否正确。")
            st.write("原始输出:")
            st.write(results)

    else:
        st.info("请在左侧输入需求并点击生成按钮。")


# --- 页面二：知识库管理 ---
with tab_kb:
    st.header("管理您的知识库文档")
    
    col1, col2 = st.columns([3, 1])
    with col1:
        st.info(f"当前状态: **{st.session_state.knowledge_base_status['status']}** | "
                f"文档数量: **{st.session_state.knowledge_base_status['doc_count']}** | "
                f"知识片段总数: **{st.session_state.knowledge_base_status['chunk_count']}**")
    with col2:
        if st.button("🔄 刷新状态", use_container_width=True, help="清理所有已上传的文档和数据库，重置知识库状态"):
            cleanup_directories()
            st.success("知识库已重置！")
            st.rerun()
            try:
                kb_manager = get_kb_manager()
                status = kb_manager.get_status()
                st.session_state.knowledge_base_status['status'] = "状态已刷新"
                st.session_state.knowledge_base_status['doc_count'] = status['doc_count']
                st.session_state.knowledge_base_status['chunk_count'] = status['chunk_count']
                st.rerun()
            except Exception as e:
                st.error(f"刷新状态时出错: {e}")

    uploaded_docs = st.file_uploader(
        "上传产品文档、需求文档、历史用例等 (.pdf, .docx, .txt, .xlsx)", 
        accept_multiple_files=True,
        type=['pdf', 'docx', 'txt', 'xlsx']
    )
    
    if st.button("处理上传的文档", use_container_width=True, type="primary"):
        if uploaded_docs:
            # Save files first
            for doc in uploaded_docs:
                file_path = os.path.join(KNOWLEDGE_BASE_DIR, doc.name)
                with open(file_path, "wb") as f:
                    f.write(doc.getbuffer())
            
            # Now, process them with a progress bar
            progress_bar = st.progress(0)
            status_text = st.empty()

            def progress_callback(progress, message):
                progress_bar.progress(progress / 100)
                status_text.info(message)

            try:
                kb_manager = get_kb_manager()
                kb_manager.load_and_process_documents(progress_callback=progress_callback)

                # 更新状态
                status = kb_manager.get_status()
                st.session_state.knowledge_base_status['status'] = "处理完成，已更新！"
                st.session_state.knowledge_base_status['doc_count'] = status['doc_count']
                st.session_state.knowledge_base_status['chunk_count'] = status['chunk_count']
                
                st.success("文档处理成功！知识库已更新。")
                st.rerun()
            except Exception as e:
                st.error(f"处理文档时出错: {e}")
        else:
            st.warning("请先上传文档。")
        
    st.write("---")
    st.subheader("知识库概览")
    
    # 显示知识库中的文件列表
    try:
        doc_list = os.listdir(KNOWLEDGE_BASE_DIR)
        if doc_list:
            st.write("当前知识库包含以下文档：")
            st.dataframe(pd.DataFrame(doc_list, columns=["文件名"]), use_container_width=True)
        else:
            st.write("知识库为空。")
    except Exception as e:
        st.error(f"无法读取知识库目录：{e}")


# --- 页面三：模型设置 ---
with tab_settings:
    st.header("配置您的AI模型")

    provider = st.selectbox("选择模型提供商", ['Ollama', 'Gemini', 'OpenAICompatible'], index=0)
    st.session_state.model_settings['provider'] = provider

    if provider == 'Ollama':
        model_name = st.text_input("模型名称", value=st.session_state.model_settings.get('model_name', 'qwen3:4b'))
        base_url = st.text_input("Ollama Base URL", value=st.session_state.model_settings.get('base_url', 'http://127.0.0.1:11434'))
        st.session_state.model_settings['model_name'] = model_name
        st.session_state.model_settings['base_url'] = base_url
    
    # ... (其他提供商的设置)

    st.header("配置语言模型 (LLM)")

    provider = st.selectbox(
        "选择模型提供商",
        ('Ollama', 'Gemini', 'OpenAICompatible'),
        index=0,
        key='provider_select' 
    )

    api_key = ''
    base_url = ''

    if provider == 'Ollama':
        base_url = st.text_input("API Base URL", value=st.session_state.model_settings.get('base_url', 'http://127.0.0.1:11434'))
    elif provider == 'Gemini':
        api_key = st.text_input("Gemini API Key", type="password", value=st.session_state.model_settings.get('api_key', ''))
    elif provider == 'OpenAICompatible':
        base_url = st.text_input("API Base URL", value=st.session_state.model_settings.get('base_url', ''))
        api_key = st.text_input("API Key (可选)", type="password", value=st.session_state.model_settings.get('api_key', ''))

    if st.button("保存设置", use_container_width=True):
        st.session_state.model_settings = {
            'provider': provider,
            'api_key': api_key,
            'base_url': base_url
        }
        st.success("模型设置已保存！")
        st.rerun()

    st.write("---")
    st.subheader("当前配置")
    st.json(st.session_state.model_settings)
