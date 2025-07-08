# core/knowledge_base.py
"""
该文件实现知识库管理，包括文档加载、文本切分、向量化和存储。
"""
import os
from langchain_community.document_loaders import DirectoryLoader, PyPDFLoader, Docx2txtLoader, TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma

class KnowledgeBaseManager:
    def __init__(self, knowledge_base_dir, chroma_db_dir, embedding_model_name='sentence-transformers/all-MiniLM-L6-v2'):
        """
        初始化知识库管理器。

        :param knowledge_base_dir: 存放知识库文档的目录。
        :param chroma_db_dir: ChromaDB持久化存储的目录。
        :param embedding_model_name: 用于文本向量化的嵌入模型名称。
        """
        self.knowledge_base_dir = knowledge_base_dir
        self.chroma_db_dir = chroma_db_dir
        self.embedding_model_name = embedding_model_name
        self.embedding_function = HuggingFaceEmbeddings(model_name=self.embedding_model_name)
        self.vector_store = self._get_or_initialize_vector_store()

    def _get_or_initialize_vector_store(self):
        """
        加载或初始化向量数据库。
        """
        if os.path.exists(self.chroma_db_dir) and os.listdir(self.chroma_db_dir):
            print(f"Loading existing ChromaDB from {self.chroma_db_dir}")
            return Chroma(
                persist_directory=self.chroma_db_dir,
                embedding_function=self.embedding_function
            )
        else:
            print(f"Initializing new, empty ChromaDB at {self.chroma_db_dir}")
            os.makedirs(self.chroma_db_dir, exist_ok=True)
            # The vector store is created empty and documents will be added later.
            return Chroma(
                persist_directory=self.chroma_db_dir,
                embedding_function=self.embedding_function
            )

    def load_and_process_documents(self, chunk_size=1000, chunk_overlap=200):
        """
        加载、切分、向量化并存储知识库目录中的所有文档。
        """
        print(f"Loading documents from {self.knowledge_base_dir}...")
        
        # 为不同文件类型创建加载器列表
        loaders = []
        for ext in ["**/*.pdf", "**/*.docx", "**/*.txt"]:
            loader_cls = None
            if ext.endswith(".pdf"):
                loader_cls = PyPDFLoader
            elif ext.endswith(".docx"):
                loader_cls = Docx2txtLoader
            elif ext.endswith(".txt"):
                loader_cls = TextLoader
            
            # 使用glob来查找文件并为每个文件创建加载器
            # DirectoryLoader now works on a directory path and a glob pattern.
            # We can create a loader for each file type we want to support.
            
        # Updated DirectoryLoader usage
        # Instead of loader_map, we can use different loaders and combine docs.
        # A simpler approach is to load them separately.
        
        pdf_loader = DirectoryLoader(self.knowledge_base_dir, glob="**/*.pdf", loader_cls=PyPDFLoader, show_progress=True, use_multithreading=True)
        docx_loader = DirectoryLoader(self.knowledge_base_dir, glob="**/*.docx", loader_cls=Docx2txtLoader, show_progress=True, use_multithreading=True)
        txt_loader = DirectoryLoader(self.knowledge_base_dir, glob="**/*.txt", loader_cls=TextLoader, loader_kwargs={'encoding': 'utf-8'}, show_progress=True, use_multithreading=True)
        
        documents = []
        documents.extend(pdf_loader.load())
        documents.extend(docx_loader.load())
        documents.extend(txt_loader.load())

        if not documents:
            print("No documents found to process.")
            return

        print(f"Loaded {len(documents)} documents.")

        # 切分文档
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap
        )
        split_docs = text_splitter.split_documents(documents)
        print(f"Split documents into {len(split_docs)} chunks.")

        # 向量化并存入ChromaDB
        print("Embedding documents and adding to ChromaDB...")
        self.vector_store.add_documents(split_docs)
        self.vector_store.persist()
        print("Knowledge base processing complete.")

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
