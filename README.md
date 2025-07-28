# 🚀 クイックスタートガイド

自分の環境ではすでに構築が終わっているため、vllm_install.shのテストを行えていません。すでにcondaの環境がある方は気を付けてください。miniforgeで行っていましたが、minicondaに変更しました。おそらく環境構築は皆さん終わっていると思うので、特に気にしないでください。

## 🔧 前提条件

🔑 SSHキーが存在すること

🧑‍💻 接続に使用するユーザー名は以下のスプレッドシートで確認：

https://docs.google.com/spreadsheets/d/1oqHeVwCXc6oPF1zQNwK40mahgqGCvJFJq7KeO4dIqeQ/edit?gid=0

## 1️⃣ 計算ノードに接続
```bash
ssh -i ~/.ssh/id_ed25519 <YOUR_USERNAME>@10.255.255.101
```
<YOUR_USERNAME> はスプレッドシートの内容に置き換えてください。

## 2️⃣ インストーラリポジトリをクローン
```bash
git clone https://github.com/au6000/install_vllm.git
```
🧹 kan.hataジョブを削除します：
```bash
bash cancel_kan_hata_jobs.sh
```

## 4️⃣ 対話的GPUセッションを起動

srun --partition=P09 \
     --nodes=1 --nodelist=osk-gpu77 \
     --cpus-per-task=10 \
     --gpus-per-node=1 \
     --time=03:00:00 \
     --pty bash -i

## 5️⃣ インストーラを実行
```bash
bash vllm_install.sh
```

このスクリプトは以下を実行します：

📦 CUDA と cuDNN モジュールの読み込み

🐍 conda のインストールと vllm 環境の作成

⚙️ PyTorch、vLLM、FlashAttention、および必要なPythonパッケージのインストール

✅ 推論テストによるインストール確認

## 6️⃣ Distill Qwen32Bで推論テストの実行
vllm 環境をアクティベートします：
```bash
conda activate vllm
```

Pythonスクリプトを実行して推論を行います
```python
python run_local_DeepSeek-R1-Distill-Qwen-32B.py
```