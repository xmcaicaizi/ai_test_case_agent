"""
Streamlit 主应用文件
"""

import streamlit as st
import os
import pandas as pd
import io
import json

from config.config import config_manager

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

# 所有已配置模型的列表
if 'models' not in st.session_state:
    # 尝试从配置加载，如果没有则使用默认值
    saved_models = config_manager.get_config('models')
    if saved_models and isinstance(saved_models, list) and len(saved_models) > 0:
        st.session_state.models = saved_models
    else:
        st.session_state.models = [{
            'name': '默认Ollama模型',
            'provider': 'Ollama',
            'model_name': 'qwen3:4b',
            'api_key': '',
            'base_url': 'http://127.0.0.1:11434'
        }]
        config_manager.set_config('models', st.session_state.models) # Save default if not present

# 当前激活模型的名称
if 'active_model_name' not in st.session_state:
    st.session_state.active_model_name = config_manager.get_config('active_model_name') or (st.session_state.models[0]['name'] if st.session_state.models else None)

def get_model_settings_by_name(model_name):
    """根据名称从模型列表中获取模型配置"""
    if not model_name: return None
    for model in st.session_state.models:
        if model.get('name') == model_name:
            return model
    return None

# 当前激活模型的详细配置，用于向后兼容
# This ensures model_settings is always in sync with the active model
st.session_state.model_settings = get_model_settings_by_name(st.session_state.active_model_name) or {}
if 'embedding_model_name' not in st.session_state:
    st.session_state.embedding_model_name = config_manager.get_config('embedding_model_name', 'nomic-embed-text')

# --- 常量定义 ---
KNOWLEDGE_BASE_DIR = "knowledge_files"
CHROMA_DB_DIR = os.path.join("db", "chroma_db")

# 确保目录存在
os.makedirs(KNOWLEDGE_BASE_DIR, exist_ok=True)
os.makedirs(CHROMA_DB_DIR, exist_ok=True)


# --- 核心逻辑函数 ---
from core.knowledge_base import KnowledgeBaseManager

@st.cache_resource
def get_kb_manager():
    """缓存知识库管理器实例"""
    return KnowledgeBaseManager(
        knowledge_base_dir=KNOWLEDGE_BASE_DIR, 
        chroma_db_dir=CHROMA_DB_DIR, 
        embedding_model_name=st.session_state.embedding_model_name
    )

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
        
        st.write("---")
        
        # 模型选择
        st.subheader("LLM 模型")
        model_names = [m['name'] for m in st.session_state.models]
        
        if model_names:
            try:
                active_model_index = model_names.index(st.session_state.active_model_name)
            except (ValueError, TypeError):
                active_model_index = 0

            selected_model_name = st.selectbox(
                "选择一个模型进行生成",
                model_names,
                index=active_model_index,
                key='model_selector'
            )
            
            # 显示当前选中模型的详细信息
            current_model = get_model_settings_by_name(selected_model_name)
            if current_model and current_model.get('provider') == 'Qwen':
                from core.qwen_config import get_qwen_model_info
                model_info = get_qwen_model_info(current_model.get('model_name', ''))
                if model_info:
                    with st.expander("📊 模型详情", expanded=False):
                        col1, col2 = st.columns(2)
                        with col1:
                            st.write(f"**描述**: {model_info.get('description', 'N/A')}")
                            st.write(f"**上下文长度**: {model_info.get('context_length', 'N/A'):,} tokens")
                        with col2:
                            cost_info = model_info.get('cost_per_1k_tokens', {})
                            st.write(f"**输入成本**: ¥{cost_info.get('input', 0)}/1K tokens")
                            st.write(f"**输出成本**: ¥{cost_info.get('output', 0)}/1K tokens")
                        
                        features = model_info.get('features', [])
                        if features:
                            st.write(f"**特性**: {', '.join(features)}")
            
            if selected_model_name != st.session_state.active_model_name:
                st.session_state.active_model_name = selected_model_name
                st.session_state.model_settings = get_model_settings_by_name(selected_model_name)
                config_manager.set_config('active_model_name', selected_model_name)
                st.rerun()
        else:
            st.warning("没有配置模型。请前往'模型设置'页面添加。")

        st.subheader("Embedding 模型")
        st.info(f"当前模型: **{st.session_state.embedding_model_name}**")

        from services.llm_service import generate_test_cases_with_llm
        from services.rag_service import generate_test_cases_with_rag

        if st.button("🚀 生成测试用例", type="primary", use_container_width=True):
            if not user_requirement and not uploaded_file:
                st.warning("请输入需求描述或上传需求文档。")
            else:
                with st.spinner("正在处理中..."):
                    try:
                        full_requirement = user_requirement
                        if uploaded_file:
                            from io import StringIO
                            stringio = StringIO(uploaded_file.getvalue().decode("utf-8"))
                            full_requirement += "\n\n--- [附加文档内容] ---\n" + stringio.read()

                        model_settings = st.session_state.model_settings
                        kb_manager = get_kb_manager()
                        
                        use_rag = use_kb and kb_manager.is_available and kb_manager.get_status()['doc_count'] > 0

                        if use_rag:
                            st.info("使用知识库增强生成...")
                            response = generate_test_cases_with_rag(model_settings, num_cases, full_requirement, kb_manager)
                        else:
                            st.info("直接使用LLM生成...")
                            response = generate_test_cases_with_llm(model_settings, num_cases, full_requirement)

                        st.session_state.generated_cases = response
                        st.rerun()

                    except Exception as e:
                        import traceback
                        error_trace = traceback.format_exc()
                        st.error(f"生成过程中发生错误: {str(e)}\n\n详细错误: {error_trace}")

    # --- 右侧主区域 ---
    st.subheader("生成结果")

    if st.session_state.generated_cases:
        # 这里将用于显示生成结果的视图
        st.success("测试用例已生成！")
        
        results = st.session_state.generated_cases
        
        # 将Pydantic模型转换为Pandas DataFrame
        try:
            # 检查返回结果的类型和结构
            if isinstance(results, dict):
                test_cases_data = results.get('test_cases', [])
            elif hasattr(results, 'test_cases'):
                test_cases_data = results.test_cases
            else:
                # 如果结果不是预期格式，尝试直接使用
                test_cases_data = results if isinstance(results, list) else []
            
            if test_cases_data:
                # 将Pydantic对象转换为字典（如果需要）
                if hasattr(test_cases_data[0], 'dict'):
                    df_data = [case.dict() if hasattr(case, 'dict') else case for case in test_cases_data]
                else:
                    df_data = test_cases_data
                
                df = pd.DataFrame(df_data)
                
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
                st.write("调试信息 - 原始返回结果:")
                st.json(results) # 显示原始返回内容以便调试

        except Exception as e:
            st.error("解析返回结果时出错，请检查LLM的输出格式是否正确。")
            st.write("错误详情:", str(e))
            st.write("调试信息 - 原始输出:")
            st.write(results)
            st.write("结果类型:", type(results))
            
            # 尝试显示部分可用信息
            try:
                if hasattr(results, '__dict__'):
                    st.write("对象属性:", list(results.__dict__.keys()))
                elif isinstance(results, dict):
                    st.write("字典键:", list(results.keys()))
            except:
                pass

    else:
        st.info("请在左侧输入需求并点击生成按钮。")


# --- 页面二：知识库管理 ---
with tab_kb:
    st.header("管理您的知识库文档")

    # --- 嵌入模型配置 ---
    st.subheader("嵌入模型配置")
    kb_manager = get_kb_manager()
    
    # 从 session_state 或默认值加载 embedding_model_name
    if 'embedding_model_name' not in st.session_state:
        st.session_state.embedding_model_name = kb_manager.embedding_model_name

    # 使用 session_state 中的值创建输入框
    embedding_model_name = st.text_input(
        "Ollama 嵌入模型名称", 
        value=st.session_state.embedding_model_name,
        key="embedding_model_input"
    )

    # 更新 session_state
    st.session_state.embedding_model_name = embedding_model_name

    col1, col2 = st.columns([1, 4])
    with col1:
        if st.button("测试并保存模型", use_container_width=True):
            with st.spinner(f"正在测试模型 '{embedding_model_name}'..."):
                # 更新 KnowledgeBaseManager 中的模型名称
                kb_manager.embedding_model_name = embedding_model_name
                # 测试连接
                if kb_manager._test_ollama_connection(model_name=embedding_model_name):
                    st.success(f"模型 '{embedding_model_name}' 连接成功！已保存为默认模型。")
                    # 清除缓存以使用新模型重新加载
                    st.cache_resource.clear()
                else:
                    st.error(f"模型 '{embedding_model_name}' 连接失败。请检查模型名称和Ollama服务是否正确。")
    
    st.markdown("---_")

    # --- 知识库操作 ---
    st.subheader("知识库操作")
    col1, col2 = st.columns([3, 1])
    with col1:
        st.info(f"当前状态: **{st.session_state.knowledge_base_status['status']}** | "
                f"文档数量: **{st.session_state.knowledge_base_status['doc_count']}** | "
                f"知识片段总数: **{st.session_state.knowledge_base_status['chunk_count']}**")
    with col2:
        if st.button("🔄 重置知识库", use_container_width=True, help="清理所有已上传的文档和数据库，重置知识库状态"):
            cleanup_directories()
            st.success("知识库已重置！")
            st.rerun()

    uploaded_docs = st.file_uploader(
        "上传产品文档、需求文档、历史用例等 (.pdf, .docx, .txt, .xlsx)", 
        accept_multiple_files=True,
        type=['pdf', 'docx', 'txt', 'xlsx']
    )
    
    if st.button("处理上传的文档", use_container_width=True, type="primary"):
        if uploaded_docs:
            kb_manager = get_kb_manager()
            kb_manager.handle_doc_upload(uploaded_docs)
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
    st.header("管理您的AI模型")

    # --- 模型列表 --- #
    st.subheader("已配置的模型")
    if not st.session_state.models:
        st.info("您还没有配置任何模型。请使用下面的表单添加一个新模型。")
    else:
        for i, model in enumerate(st.session_state.models):
            with st.expander(f"**{model.get('name', f'模型 {i+1}')}** (`{model.get('provider')}` - `{model.get('model_name')}`)"):
                st.text(f"提供商: {model.get('provider')}")
                st.text(f"模型名称: {model.get('model_name')}")
                st.text(f"Base URL: {model.get('base_url', 'N/A')}")
                api_key_display = "*" * 10 if model.get('api_key') else "未设置"
                st.text(f"API Key: {api_key_display}")
                
                col1, col2, col3 = st.columns([1,1,5])
                with col1:
                    if st.button("设为活动模型", key=f"activate_{i}", use_container_width=True):
                        st.session_state.active_model_name = model['name']
                        st.session_state.model_settings = model
                        config_manager.set_config('active_model_name', model['name'])
                        st.success(f"模型 '{model['name']}' 已被激活！")
                        st.rerun()
                with col2:
                    if st.button("删除", key=f"delete_{i}", type="secondary", use_container_width=True):
                        # If deleting the active model, reset active model to the first one if possible
                        if st.session_state.active_model_name == model['name']:
                            st.session_state.active_model_name = st.session_state.models[0]['name'] if len(st.session_state.models) > 1 else None
                            config_manager.set_config('active_model_name', st.session_state.active_model_name)
                        
                        st.session_state.models.pop(i)
                        config_manager.set_config('models', st.session_state.models)
                        st.success(f"模型 '{model.get('name')}' 已被删除！")
                        st.rerun()

    st.write("---")

    # --- 通义千问快速配置 --- #
    st.subheader("🚀 通义千问快速配置")
    with st.expander("点击展开通义千问模型配置", expanded=False):
        st.info("通义千问支持 OpenAI Compatible API，只需要您的 API Key 即可快速配置。")
        
        qwen_api_key = st.text_input(
            "通义千问 API Key", 
            type="password",
            placeholder="sk-xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx",
            help="请在阿里云百炼控制台获取您的 API Key"
        )
        
        if qwen_api_key:
            from core.qwen_config import get_qwen_model_list, validate_qwen_api_key
            
            if validate_qwen_api_key(qwen_api_key):
                st.success("✅ API Key 格式验证通过")
                
                qwen_models = get_qwen_model_list()
                selected_qwen_models = st.multiselect(
                    "选择要添加的通义千问模型",
                    options=[model["model_name"] for model in qwen_models],
                    default=["qwen-plus"],
                    format_func=lambda x: next((model["display_name"] + f" - {model['description']}" for model in qwen_models if model["model_name"] == x), x)
                )
                
                if st.button("🎯 一键添加选中的通义千问模型", type="primary"):
                    added_count = 0
                    for model_name in selected_qwen_models:
                        model_info = next((model for model in qwen_models if model["model_name"] == model_name), None)
                        if model_info:
                            # 检查是否已存在相同的模型
                            existing_model = next((m for m in st.session_state.models if m.get('provider') == 'Qwen' and m.get('model_name') == model_name), None)
                            if not existing_model:
                                new_model = {
                                    'name': model_info["display_name"],
                                    'provider': 'Qwen',
                                    'model_name': model_name,
                                    'api_key': qwen_api_key,
                                    'base_url': 'https://dashscope.aliyuncs.com/compatible-mode/v1',
                                    'description': model_info["description"],
                                    'context_length': model_info["context_length"],
                                    'cost_info': model_info["cost_info"]
                                }
                                st.session_state.models.append(new_model)
                                added_count += 1
                    
                    if added_count > 0:
                        config_manager.set_config('models', st.session_state.models)
                        
                        # 如果是第一次添加模型，设为活动模型
                        if len(st.session_state.models) == added_count:
                            st.session_state.active_model_name = st.session_state.models[0]['name']
                            config_manager.set_config('active_model_name', st.session_state.models[0]['name'])
                        
                        st.success(f"✅ 成功添加 {added_count} 个通义千问模型！")
                        st.rerun()
                    else:
                        st.warning("所选模型已存在，未添加新模型。")
            else:
                st.error("❌ API Key 格式不正确，请检查后重试")

    st.write("---")

    # --- 添加/编辑模型表单 --- #
    st.subheader("手动添加模型")
    with st.form(key="add_model_form"):
        name = st.text_input("模型别名*", placeholder="例如：我的本地Qwen模型")
        provider = st.selectbox("选择模型提供商*", ('Ollama', 'Qwen', 'Doubao', 'Gemini', 'OpenAICompatible'))
        
        # 根据选择的提供商显示不同的提示
        if provider == 'Qwen':
            model_name = st.selectbox("通义千问模型*", ['qwen-plus', 'qwen-turbo', 'qwen-max', 'qwen-plus-latest', 'qwen-turbo-latest', 'qwen-max-latest'])
            base_url = st.text_input("Base URL", value="https://dashscope.aliyuncs.com/compatible-mode/v1")
        elif provider == 'Ollama':
            model_name = st.text_input("模型名称*", placeholder="例如：qwen3:4b, llama3:8b")
            base_url = st.text_input("Base URL", value="http://127.0.0.1:11434")
        else:
            model_name = st.text_input("模型名称*", placeholder="例如：gpt-4, claude-3")
            base_url = st.text_input("Base URL", placeholder="例如：https://api.openai.com/v1")
        
        api_key = st.text_input("API Key", type="password")

        submitted = st.form_submit_button("添加模型")
        if submitted:
            if not name or not provider or not model_name:
                st.error("模型别名、提供商和模型名称是必填项。")
            else:
                new_model = {
                    'name': name,
                    'provider': provider,
                    'model_name': model_name,
                    'api_key': api_key,
                    'base_url': base_url
                }
                st.session_state.models.append(new_model)
                config_manager.set_config('models', st.session_state.models)
                
                # If it's the first model added, set it as active
                if len(st.session_state.models) == 1:
                    st.session_state.active_model_name = name
                    config_manager.set_config('active_model_name', name)

                st.success(f"模型 '{name}' 已成功添加！")
                st.rerun()
