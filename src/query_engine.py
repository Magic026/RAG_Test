# src/query_engine.py
from llama_index.core import StorageContext, load_index_from_storage
from llama_index.multi_modal import MultiModalVectorStoreIndex
from llama_index.llms.huggingface import HuggingFaceLLM
from llama_index.embeddings.open_clip import OpenCLIPEmbedding
import torch


class MultimodalQueryEngine:
    def __init__(self, index_dir="index", llm_model="Qwen/Qwen-VL-Chat"):
        self.index_dir = index_dir
        self.llm_model = llm_model
        self.engine = None
        self._initialize_engine()

    def _initialize_engine(self):
        print("🔄 加载多模态索引...")
        storage_context = StorageContext.from_defaults(persist_dir=self.index_dir)
        index = load_index_from_storage(
            storage_context,
            embed_model=OpenCLIPEmbedding(model_name="ViT-B-32", device="cuda"),
        )

        print("🚀 加载 Qwen2.5-VL-72b 模型...")
        llm = HuggingFaceLLM(
            model_name=self.llm_model,
            tokenizer_name=self.llm_model,
            device_map="cuda",
            model_kwargs={
                "trust_remote_code": True,
                "torch_dtype": torch.bfloat16,
                "max_memory": {0: "20GiB", "cpu": "32GiB"} if torch.cuda.is_available() else None,
            },
            generate_kwargs={"max_new_tokens": 256, "temperature": 0.3},
        )

        self.engine = index.as_query_engine(
            llm=llm,
            similarity_top_k=3,
            text_qa_template=(
                "请根据以下上下文回答问题。"
                "如果信息不足，请回答“无法确定”。"
                "请尽量引用图表或文本中的具体信息，并说明来源页码。\n\n"
                "上下文：{context_str}\n\n"
                "问题：{query_str}\n\n"
                "答案："
            ),
        )

    def query_with_source(self, question):
        response = self.engine.query(question)

        # 提取来源信息
        source_pages = []
        source_files = []
        for node in response.source_nodes:
            meta = node.node.metadata
            source_pages.append(meta.get("page_label", "未知"))
            source_files.append(meta.get("file_name", "未知"))

        # 推测最可能的来源（取第一个）
        filename = source_files[0] if source_files else "未知"
        page = int(source_pages[0]) if source_pages and source_pages[0].isdigit() else 1

        return {
            "answer": str(response),
            "filename": filename,
            "page": page,
            "raw_response": response,
            "sources": [{"filename": f, "page": p} for f, p in zip(source_files, source_pages)]
        }
