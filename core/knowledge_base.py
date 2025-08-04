# core/knowledge_base.py
"""
该文件实现知识库管理，包括文档加载、文本切分、向量化和存储。
"""
import os
import pandas as pd
import glob
import time
import requests
import json
from langchain_community.document_loaders import DirectoryLoader, PyPDFLoader, Docx2txtLoader, TextLoader, UnstructuredExcelLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.embeddings import Embeddings
from langchain_chroma import Chroma
from langchain.docstore.document import Document
from typing import List

class CustomOllamaEmbeddings(Embeddings):
    def __init__(self, model: str, base_url: str = "http://localhost:11434", max_retries: int = 3, retry_delay: float = 1.0):
        self.model = model
        self.base_url = base_url
        self.max_retries = max_retries
        self.retry_delay = retry_delay

    def _call_ollama_with_retry(self, text: str) -> List[float]:
        """调用Ollama embedding API，带重试机制"""
        for attempt in range(self.max_retries):
            try:
                # 使用requests直接调用API
                # 为Qwen3 embedding模型添加特殊标记
                processed_text = text
                if "qwen3-embedding" in self.model.lower():
                    processed_text += "<|endoftext|>"

                response = requests.post(
                    f"{self.base_url}/api/embed",
                    json={"model": self.model, "input": processed_text},
                    headers={"Content-Type": "application/json"},
                    timeout=30
                )
                response.raise_for_status()
                result = response.json()
                return result["embeddings"][0]
            except Exception as e:
                if attempt == self.max_retries - 1:
                    # 最后一次尝试失败，抛出详细错误
                    raise Exception(f"Ollama embedding failed after {self.max_retries} attempts. "
                                  f"Model: {self.model}, Error: {str(e)}. "
                                  f"Please check if Ollama service is running and the model '{self.model}' is available.")
                else:
                    print(f"Ollama embedding attempt {attempt + 1} failed: {str(e)}. Retrying in {self.retry_delay}s...")
                    time.sleep(self.retry_delay)

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        embeddings = []
        for i, text in enumerate(texts):
            try:
                embedding = self._call_ollama_with_retry(text)
                embeddings.append(embedding)
            except Exception as e:
                print(f"Failed to embed document {i+1}/{len(texts)}: {str(e)}")
                raise
        return embeddings

    def embed_query(self, text: str) -> List[float]:
        try:
            return self._call_ollama_with_retry(text)
        except Exception as e:
            print(f"Failed to embed query: {str(e)}")
            raise

class KnowledgeBaseManager:
    def __init__(self, knowledge_base_dir="knowledge_files", chroma_db_dir="db/chroma_db", embedding_model_name="nomic-embed-text"):
        """
        初始化知识库管理器。

        :param knowledge_base_dir: 存放知识库文档的目录。
        :param chroma_db_dir: ChromaDB持久化存储的目录。
        :param embedding_model_name: 用于文本向量化的嵌入模型名称。
        """
        self.knowledge_base_dir = knowledge_base_dir
        self.chroma_db_dir = chroma_db_dir
        self.embedding_model_name = embedding_model_name
        self.embedding_function = None
        self.vector_store = None
        self.is_available = False  # 标记知识库是否可用

    def set_embedding_model(self, embedding_model_name: str):
        """
        动态设置并重新初始化嵌入模型和向量数据库。
        """
        print(f"Setting embedding model to: {embedding_model_name}")
        self.embedding_model_name = embedding_model_name
        self.embedding_function = None
        self.vector_store = None
        self.is_available = False
        self._ensure_initialized()

    def _test_ollama_connection(self, model_name: str = None) -> bool:
        """测试Ollama连接是否可用，可以指定模型"""
        model_to_test = model_name if model_name else self.embedding_model_name
        try:
            # 使用requests测试embedding功能
            # 为Qwen3 embedding模型添加特殊标记
            test_input = "test"
            if "qwen3-embedding" in model_to_test.lower():
                test_input += "<|endoftext|>"

            response = requests.post(
                f"http://localhost:11434/api/embed",
                json={"model": model_to_test, "input": test_input},
                headers={"Content-Type": "application/json"},
                timeout=10
            )
            response.raise_for_status()
            return True
        except Exception as e:
            print(f"Ollama embedding test failed for model {model_to_test}: {str(e)}")
            return False

    def _ensure_initialized(self):
        """确保 embedding_function 和 vector_store 已初始化。"""
        if self.embedding_function is None:
            # 首先测试Ollama是否可用
            if self._test_ollama_connection():
                try:
                    self.embedding_function = CustomOllamaEmbeddings(model=self.embedding_model_name)
                    self.vector_store = Chroma(
                        persist_directory=self.chroma_db_dir,
                        embedding_function=self.embedding_function
                    )
                    self.is_available = True
                    print("Knowledge base initialized successfully with Ollama embeddings.")
                except Exception as e:
                    print(f"Failed to initialize knowledge base with Ollama: {str(e)}")
                    self.is_available = False
            else:
                print("Ollama embedding service is not available. Knowledge base features will be disabled.")
                self.is_available = False

    def _load_excel_documents(self):
        """
        Loads and splits Excel files into documents (one per row).
        """
        split_docs = []
        excel_files = glob.glob(os.path.join(self.knowledge_base_dir, "**/*.xlsx"), recursive=True)
        for file_path in excel_files:
            try:
                df = pd.read_excel(file_path)
                headers = "\t".join(df.columns)
                for _, row in df.iterrows():
                    row_content = "\t".join(map(str, row.values))
                    content = f"{headers}\n{row_content}"
                    metadata = {"source": file_path}
                    split_docs.append(Document(page_content=content, metadata=metadata))
            except Exception as e:
                print(f"Error processing excel file {file_path}: {e}")
        return split_docs

    def load_and_process_documents(self, chunk_size=1000, chunk_overlap=200, progress_callback=None):
        """
        加载、切分、向量化并存储知识库目录中的所有文档。
        :param progress_callback: 一个用于报告进度的回调函数。
        """
        self._ensure_initialized()
        if not self.is_available:
            if progress_callback:
                progress_callback(100, "知识库服务不可用，跳过文档处理。")
            print("Knowledge base is not available. Skipping document processing.")
            return
            
        print(f"Loading documents from {self.knowledge_base_dir}...")
        if progress_callback: progress_callback(0, "开始加载文档...")

        pdf_loader = DirectoryLoader(self.knowledge_base_dir, glob="**/*.pdf", loader_cls=PyPDFLoader, show_progress=True, use_multithreading=True)
        docx_loader = DirectoryLoader(self.knowledge_base_dir, glob="**/*.docx", loader_cls=Docx2txtLoader, show_progress=True, use_multithreading=True)
        txt_loader = DirectoryLoader(self.knowledge_base_dir, glob="**/*.txt", loader_cls=TextLoader, loader_kwargs={'encoding': 'utf-8'}, show_progress=True, use_multithreading=True)
        
        standard_docs = []
        standard_docs.extend(pdf_loader.load())
        standard_docs.extend(docx_loader.load())
        standard_docs.extend(txt_loader.load())
        if progress_callback: progress_callback(20, "标准文档加载完成，开始加载 Excel 文件...")

        excel_split_docs = self._load_excel_documents()
        if progress_callback: progress_callback(40, "Excel 文件加载完成，开始切分文档...")

        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap
        )
        standard_split_docs = text_splitter.split_documents(standard_docs)

        all_split_docs = excel_split_docs + standard_split_docs

        if not all_split_docs:
            print("No documents found to process.")
            if progress_callback: progress_callback(100, "未找到可处理的文档。")
            return

        total_chunks = len(all_split_docs)
        print(f"Split documents into {total_chunks} chunks.")
        if progress_callback: progress_callback(60, f"文档切分完成，共 {total_chunks} 个片段。开始向量化...")

        # Vectorize and store in ChromaDB
        batch_size = 500
        for i in range(0, total_chunks, batch_size):
            batch_docs = all_split_docs[i:i + batch_size]
            self.vector_store.add_documents(batch_docs)
            
            progress = 60 + int((i + len(batch_docs)) / total_chunks * 40)
            if progress_callback: 
                progress_callback(progress, f"正在处理 {i + len(batch_docs)} / {total_chunks} 个片段...")

        self.vector_store.persist()  # 强制持久化
        print("Knowledge base processing complete.")
        if progress_callback: progress_callback(100, "知识库处理完成！")

    def get_retriever(self, search_type="similarity", search_kwargs={"k": 5}):
        """
        获取一个配置好的检索器。

        :param search_type: 检索类型 (e.g., "similarity", "mmr")
        :param search_kwargs: 检索参数 (e.g., {"k": 5})
        :return: LangChain Retriever
        """
        self._ensure_initialized()
        if not self.is_available:
            print("Knowledge base is not available. Cannot create retriever.")
            return None
        return self.vector_store.as_retriever(search_type=search_type, search_kwargs=search_kwargs)

    def get_status(self):
        """
        获取知识库的状态信息。

        :return: 一个包含文档数和片段数的字典。
        """
        if not self.is_available:
            return {
                "doc_count": 0,
                "chunk_count": 0,
                "available": False,
                "message": "Knowledge base service is not available"
            }
            
        doc_count = len(os.listdir(self.knowledge_base_dir))
        try:
            self._ensure_initialized()
            # .count() is a method on the collection object in Chroma
            chunk_count = self.vector_store._collection.count()
        except Exception:
            chunk_count = 0 # 如果初始化失败或没有集合，则为0
            
        return {
            "doc_count": doc_count,
            "chunk_count": chunk_count,
            "available": True
        }

    def delete_documents(self, file_paths: List[str]):
        """
        Deletes all chunks associated with the given file paths from the vector store.
        """
        self._ensure_initialized()
        if not self.is_available:
            print("Knowledge base is not available. Cannot delete documents.")
            return

        collection_data = self.vector_store._collection.get(where={"source": {"$in": file_paths}})
        ids_to_delete = collection_data.get('ids', [])

        if self.vector_store and ids_to_delete:
            self.vector_store.delete(ids=ids_to_delete)
            self.vector_store.persist()  # 强制持久化
            print(f"Deleted {len(ids_to_delete)} chunks and persisted changes.")
        else:
            print(f"No chunks found to delete for files: {file_paths}")

    def handle_doc_upload(self, uploaded_docs):
        """
        处理上传的文档，保存到知识库目录并处理
        """
        import streamlit as st
        
        if not self.is_available:
            st.error("⚠️ Ollama embedding服务不可用，无法处理文档。")
            return

        for doc in uploaded_docs:
            file_path = os.path.join(self.knowledge_base_dir, doc.name)
            with open(file_path, "wb") as f:
                f.write(doc.getbuffer())

        progress_bar = st.progress(0)
        status_text = st.empty()

        def progress_callback(progress, message):
            progress_bar.progress(progress / 100)
            status_text.info(message)

        try:
            self.load_and_process_documents(progress_callback=progress_callback)
            status = self.get_status()
            st.session_state.knowledge_base_status['status'] = "处理完成，已更新！"
            st.session_state.knowledge_base_status['doc_count'] = status['doc_count']
            st.session_state.knowledge_base_status['chunk_count'] = status['chunk_count']
            st.success("文档处理成功！知识库已更新。")
            st.rerun()
        except Exception as e:
            st.error(f"处理文档时出错: {e}")

# Example usage (for testing purposes)
if __name__ == '__main__':
    # 创建临时目录和文件用于测试
    KB_DIR = "temp_kb"
    DB_DIR = "temp_db"
    
    # Ensure directories are clean before test
    import shutil
    import gc

    def cleanup():
        # Force garbage collection to release file handles
        gc.collect()
        if os.path.exists(KB_DIR):
            shutil.rmtree(KB_DIR, ignore_errors=True)
        if os.path.exists(DB_DIR):
            shutil.rmtree(DB_DIR, ignore_errors=True)

    cleanup()
    os.makedirs(KB_DIR)
    
    with open(os.path.join(KB_DIR, "test.txt"), "w", encoding='utf-8') as f:
        f.write("这是一个关于登录功能的测试文档。\n用户需要输入用户名和密码进行登录。")
    
    with open(os.path.join(KB_DIR, "test2.txt"), "w", encoding='utf-8') as f:
        f.write("这是一个关于注册功能的测试文档。\n用户需要提供邮箱和设置密码来创建账户。")
        
    try:
        # 1. 初始化知识库管理器
        kb_manager = KnowledgeBaseManager(knowledge_base_dir=KB_DIR, chroma_db_dir=DB_DIR)
        
        # 2. 处理文档
        kb_manager.load_and_process_documents()
        
        # 3. 获取检索器并测试
        retriever = kb_manager.get_retriever()
        
        query = "如何测试登录功能？"
        results = retriever.invoke(query)
        
        print(f"\n--- Test Query: '{query}' ---")
        print(f"Found {len(results)} relevant chunks:")
        for doc in results:
            print(f"  Source: {doc.metadata.get('source', 'N/A')}")
            print(f"  Content: {doc.page_content[:100]}...")
            
        # 4. 获取状态
        status = kb_manager.get_status()
        print("\n--- Knowledge Base Status ---")
        print(f"Documents: {status['doc_count']}")
        print(f"Chunks: {status['chunk_count']}")
    finally:
        # 清理临时文件和目录
        cleanup()
