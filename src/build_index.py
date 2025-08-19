# src/build_index.py
import os
from llama_index.core import SimpleDirectoryReader
from llama_index.multi_modal import MultiModalVectorStoreIndex
from llama_index.embeddings.open_clip import OpenCLIPEmbedding
from llama_index.core import Settings


def build_index(data_dir="data/财报数据库", index_dir="index"):
    print("🔧 正在加载 PDF 文档...")
    reader = SimpleDirectoryReader(
        input_dir=data_dir,
        file_extractor={".pdf": "unstructured"},
        recursive=True,
    )
    docs = reader.load_data()

    print(f"📄 共加载 {len(docs)} 个文档块（文本+图像）")

    # 设置嵌入模型
    Settings.embed_model = OpenCLIPEmbedding(
        model_name="ViT-B-32",
        device="cuda" if os.getenv("USE_GPU", "true").lower() == "true" else "cpu"
    )

    # 构建多模态索引
    index = MultiModalVectorStoreIndex.from_documents(
        docs,
        show_progress=True,
    )

    # 保存索引
    os.makedirs(index_dir, exist_ok=True)
    index.storage_context.persist(persist_dir=index_dir)
    print(f"✅ 索引已保存至: {index_dir}")


if __name__ == "__main__":
    build_index()
