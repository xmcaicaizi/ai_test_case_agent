# core/llm_integrator.py
"""
该文件实现LLM服务集成，支持多种模型提供商。
"""
import os
from langchain_community.chat_models import ChatOllama
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_openai import ChatOpenAI

class LLMIntegrator:
    def __init__(self, model_provider, model_name, api_key=None, base_url=None):
        """
        初始化LLM集成器。

        :param model_provider: 模型提供商 ('Ollama', 'Gemini', 'OpenAICompatible')
        :param model_name: 模型的具体名称
        :param api_key: API密钥 (Gemini 和 OpenAI兼容模型需要)
        :param base_url: API基础URL (Ollama 和 OpenAI兼容模型需要)
        """
        self.model_provider = model_provider
        self.model_name = model_name
        self.api_key = api_key
        self.base_url = base_url
        self.llm = self._create_llm_instance()

    def _create_llm_instance(self):
        """
        根据提供商创建并返回一个LLM实例。
        """
        if self.model_provider == 'Ollama':
            return ChatOllama(
                model=self.model_name,
                base_url=self.base_url or "http://127.0.0.1:11434"
            )
        elif self.model_provider == 'Gemini':
            if not self.api_key:
                raise ValueError("Gemini API key is required.")
            # Configure the API key
            os.environ["GOOGLE_API_KEY"] = self.api_key
            return ChatGoogleGenerativeAI(model=self.model_name)
        
        elif self.model_provider == 'Qwen':
            base_url = self.base_url or "https://dashscope.aliyuncs.com/compatible-mode/v1"
            if not self.api_key:
                raise ValueError("API key is required for Qwen.")
            return ChatOpenAI(
                model=self.model_name,
                api_key=self.api_key,
                base_url=base_url
            )
        elif self.model_provider == 'Doubao':
            base_url = self.base_url or "https://ark.cn-beijing.volces.com/api/v3"
            if not self.api_key:
                raise ValueError("API key is required for Doubao.")
            return ChatOpenAI(
                model=self.model_name,
                api_key=self.api_key,
                base_url=base_url
            )
        elif self.model_provider == 'OpenAICompatible':
            if not self.api_key:
                raise ValueError("API key is required for OpenAI compatible models.")
            if not self.base_url:
                raise ValueError("Base URL is required for OpenAI compatible models.")
            return ChatOpenAI(
                model=self.model_name,
                api_key=self.api_key,
                base_url=self.base_url
            )
        else:
            raise ValueError(f"Unsupported model provider: {self.model_provider}")

    def get_llm(self):
        """
        获取LLM实例。
        """
        return self.llm

# Example usage (for testing purposes)
if __name__ == '__main__':
    # Test Ollama
    try:
        ollama_integrator = LLMIntegrator(model_provider='Ollama', model_name='qwen3:4b')
        ollama_llm = ollama_integrator.get_llm()
        print("Ollama LLM instance created successfully:", ollama_llm)
        # response = ollama_llm.invoke("Hello, who are you?")
        # print("Ollama response:", response.content)
    except Exception as e:
        print(f"Error creating Ollama instance: {e}")

    # Test Gemini (requires API Key)
    # Make sure to set the GOOGLE_API_KEY environment variable or pass it here
    GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")
    if GEMINI_API_KEY:
        try:
            gemini_integrator = LLMIntegrator(model_provider='Gemini', model_name='gemini-1.5-pro-latest', api_key=GEMINI_API_KEY)
            gemini_llm = gemini_integrator.get_llm()
            print("Gemini LLM instance created successfully:", gemini_llm)
        except Exception as e:
            print(f"Error creating Gemini instance: {e}")
    else:
        print("Skipping Gemini test: GEMINI_API_KEY not found in environment variables.")

    # Test OpenAI Compatible (e.g. local Qwen model served with OpenAI API)
    # Make sure to set relevant environment variables or pass them here
    OPENAI_API_KEY = "none" # Often not needed for local servers
    OPENAI_BASE_URL = "http://localhost:8000/v1" 
    try:
        openai_compat_integrator = LLMIntegrator(
            model_provider='OpenAICompatible', 
            model_name='Qwen/Qwen2-7B-Instruct-GGUF', 
            api_key=OPENAI_API_KEY, 
            base_url=OPENAI_BASE_URL
        )
        openai_compat_llm = openai_compat_integrator.get_llm()
        print("OpenAI Compatible LLM instance created successfully:", openai_compat_llm)
    except Exception as e:
        print(f"Error creating OpenAI Compatible instance: {e}")
