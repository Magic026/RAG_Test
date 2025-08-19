```shell
multimodal-rag-challenge/
│
├── data/
│   ├── 财报数据库/                ← 放入提供的财报数据库.zip 解压后的内容
│   ├── train.json
│   ├── test.json
│   └── example.json
│
├── src/
│   ├── build_index.py           ← 构建多模态索引
│   ├── query_engine.py          ← 查询与推理
│   └── submit.py                ← 生成最终提交文件
│
├── index/                       ← 生成的索引文件（运行后自动创建）
├── output/
│   └── submit.json              ← 最终提交结果
│
├── requirements.txt
└── README.md
```