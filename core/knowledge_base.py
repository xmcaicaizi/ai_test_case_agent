# core/knowledge_base.py
"""
该文件实现知识库管理，包括文档加载、文本切分、向量化和存储。
"""
import os
import pandas as pd
import glob
from langchain_community.document_loaders import DirectoryLoader, PyPDFLoader, Docx2txtLoader, TextLoader, UnstructuredExcelLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import OllamaEmbeddings
from langchain_chroma import Chroma
from langchain.docstore.document import Document

class KnowledgeBaseManager:
    def __init__(self, knowledge_base_dir="knowledge_files", chroma_db_dir="db/chroma_db", embedding_model_name="dengcao/Qwen3-Embedding-0.6B:Q8_0"):
        """
        初始化知识库管理器。

        :param knowledge_base_dir: 存放知识库文档的目录。
        :param chroma_db_dir: ChromaDB持久化存储的目录。
        :param embedding_model_name: 用于文本向量化的嵌入模型名称。
        """
        self.knowledge_base_dir = knowledge_base_dir
        self.chroma_db_dir = chroma_db_dir
        self.embedding_model_name = embedding_model_name
        self.embedding_function = OllamaEmbeddings(model=self.embedding_model_name)
        self.vector_store = self._get_or_initialize_vector_store()

    def _get_or_initialize_vector_store(self):
        """
        加载或初始化向量数据库。
        """
        # Chroma will automatically create the directory if it doesn't exist
        return Chroma(
            persist_directory=self.chroma_db_dir,
            embedding_function=self.embedding_function
        )

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

        print("Knowledge base processing complete.")
        if progress_callback: progress_callback(100, "知识库处理完成！")

    def get_retriever(self, search_type="similarity", search_kwargs={"k": 5}):
        """
        获取一个配置好的检索器。

        :param search_type: 检索类型 (e.g., "similarity", "mmr")
        :param search_kwargs: 检索参数 (e.g., {"k": 5})
        :return: LangChain Retriever
        """
        if not self.vector_store:
            # This case should ideally not be hit with the new logic
            raise ValueError("Vector store is not initialized.")
        return self.vector_store.as_retriever(search_type=search_type, search_kwargs=search_kwargs)

    def get_status(self):
        """
        获取知识库的状态信息。

        :return: 一个包含文档数和片段数的字典。
        """
        doc_count = len(os.listdir(self.knowledge_base_dir))
        try:
            # .count() is a method on the collection object in Chroma
            chunk_count = self.vector_store._collection.count()
        except Exception:
            chunk_count = "N/A"
            
        return {
            "doc_count": doc_count,
            "chunk_count": chunk_count
        }

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
