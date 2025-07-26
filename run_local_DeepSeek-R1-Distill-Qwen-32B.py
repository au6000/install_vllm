import os
from vllm import LLM, SamplingParams
from transformers import AutoTokenizer

# 1. モデルへのパスを指定
# パスは /home/P09/P09U010 ではなく、共有ディレクトリを指しているため修正
model_path = "/home/Competition2025/P09/shareP09/models/DeepSeek-R1-Distill-Qwen-32B"

try:
    print(f"ローカルモデルをロードします: {model_path}")
    
    tokenizer = AutoTokenizer.from_pretrained(
        model_path,
        trust_remote_code=True
    )

    # 2. LLMの初期化時に max_model_len を追加
    llm = LLM(
        model=model_path,
        tensor_parallel_size=1,
        trust_remote_code=True,
        max_model_len=8192  # ★★★ 修正点: 最大トークン長を8192に制限
    )

except (ValueError, RuntimeError, OSError) as e:
    print(f"\nモデルのロード中にエラーが発生しました: {e}")
    exit()


# チャット形式のプロンプトを定義
messages = [
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": "vLLMの主な利点を3つ教えてください。"},
]

# モデルが要求するチャット形式にプロンプトを変換
prompt = tokenizer.apply_chat_template(
    messages,
    tokenize=False,
    add_generation_prompt=True
)

# サンプリングパラメータの設定
sampling_params = SamplingParams(
    temperature=0.7,
    max_tokens=512,
    stop=["<|im_end|>"]
)

# 推論を実行
print("\n推論を開始します...")
outputs = llm.generate([prompt], sampling_params)
print("推論が完了しました。")

# 結果を表示
for output in outputs:
    generated_text = output.outputs[0].text
    print("\n===== AIの応答 =====")
    print(generated_text.strip())
    print("====================")