# src/submit.py
import json
import os
from query_engine import MultimodalQueryEngine


def generate_submit(test_file="data/test.json", output_file="../output/submit.json"):
    # 创建输出目录
    os.makedirs(os.path.dirname(output_file), exist_ok=True)

    # 初始化引擎
    engine = MultimodalQueryEngine()

    # 读取测试集
    with open(test_file, 'r', encoding='utf-8') as f:
        test_data = json.load(f)

    results = []
    for item in test_data:
        q_id = item["id"]
        question = item["question"]

        print(f"🔍 处理问题 {q_id}: {question}")
        try:
            result = engine.query_with_source(question)
            results.append({
                "id": q_id,
                "filename": result["filename"],
                "page": result["page"],
                "answer": result["answer"].strip()
            })
        except Exception as e:
            print(f"❌ 问题 {q_id} 处理失败: {e}")
            results.append({
                "id": q_id,
                "filename": "unknown.pdf",
                "page": 1,
                "answer": "无法确定"
            })

    # 保存提交文件
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    print(f"✅ 提交文件已生成: {output_file}")


if __name__ == "__main__":
    generate_submit()
