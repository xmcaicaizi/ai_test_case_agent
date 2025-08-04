from langchain_ollama import OllamaEmbeddings

emb = OllamaEmbeddings(model="dengcao/Qwen3-Embedding-0.6B:Q8_0")
print(emb.embed_query("设计一个登录页面的测试用例"))