import os
import streamlit as st

def handle_doc_upload(kb_manager, uploaded_docs, knowledge_base_dir):
    if not kb_manager.is_available:
        st.error("⚠️ Ollama embedding服务不可用，无法处理文档。")
        return

    for doc in uploaded_docs:
        file_path = os.path.join(knowledge_base_dir, doc.name)
        with open(file_path, "wb") as f:
            f.write(doc.getbuffer())

    progress_bar = st.progress(0)
    status_text = st.empty()

    def progress_callback(progress, message):
        progress_bar.progress(progress / 100)
        status_text.info(message)

    try:
        kb_manager.load_and_process_documents(progress_callback=progress_callback)
        status = kb_manager.get_status()
        st.session_state.knowledge_base_status['status'] = "处理完成，已更新！"
        st.session_state.knowledge_base_status['doc_count'] = status['doc_count']
        st.session_state.knowledge_base_status['chunk_count'] = status['chunk_count']
        st.success("文档处理成功！知识库已更新。")
        st.rerun()
    except Exception as e:
        st.error(f"处理文档时出错: {e}")