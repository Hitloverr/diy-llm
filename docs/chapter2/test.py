import re

def btp_hex_list(text):
    """
    UTF-8字节级预分词，返回：
        1. tokens: 每个字符的字节序列+位置信息
        2. t: 所有字节的十六进制字符串列表
    """
    tokens = []
    t = []
    for idx, char in enumerate(text):
        utf8_bytes = char.encode('utf-8')
        hex_bytes = ' '.join(f"{b:02X}" for b in utf8_bytes)
        tokens.append({
            'char': char,
            'bytes': hex_bytes, # 单个字符对应的UTF-8字节序列
            'start': idx,   # 文本信息起始位置
            'end': idx + 1   # 文本信息结束位置
        })
        t.extend([f"{b:02X}" for b in utf8_bytes])
    return tokens, t

def test_deepseek_tokenizer():
    from transformers import AutoTokenizer
    model_name = 'deepseek-ai/deepseek-coder-6.7b-instruct'
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    print(f'成功加载{model_name}的分词器')
    print(f"分词器大小V:{len(tokenizer.get_vocab())}")
    chinese_test = '注意力机制是AI的核心技术。 🚀 🚀'
    encoded_ids = tokenizer.encode(chinese_test, add_special_tokens=False)
    tokens = tokenizer.convert_ids_to_tokens(encoded_ids)
    print(f"\n原文{chinese_test}")
    print(f"编码：{tokens}")
    print(f"IDS: {encoded_ids}")

    print(encoded_ids)


if __name__ == "__main__":
    text = "Hi，你好🐋"
    # tokens, t = btp_hex_list(text)
    # for i in tokens:
    #     print(i)
    # print(t)
    # 分词器训练的核心：迭代更新候选子词 ->控制词表大小或者收敛指标-> 监控质量指标。
    test_deepseek_tokenizer()
