from .knowledge_base import KnowledgeBaseManager
from .llm_integrator import LLMIntegrator
from .rag_chain import create_rag_chain

class AppFactory:
    @staticmethod
    def create_kb_manager(knowledge_base_dir, chroma_db_dir, embedding_model_name):
        return KnowledgeBaseManager(knowledge_base_dir, chroma_db_dir, embedding_model_name)

    @staticmethod
    def create_llm_integrator(model_provider, model_name, api_key=None, base_url=None):
        return LLMIntegrator(model_provider, model_name, api_key, base_url)

    @staticmethod
    def create_rag_chain(retriever, llm, num_cases):
        return create_rag_chain(retriever, llm, num_cases=num_cases)