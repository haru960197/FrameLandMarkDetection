# FrameLandMarkDetection

MediaPipe FaceMesh を用いた，撮影済み MP4 動画に対する顔ランドマーク検出システムです．

## ディレクトリ構成

```
FrameLandMarkDetection/
├── config.yaml          # 設定ファイル（解像度・FPS 等をここで変更）
├── main.py              # メインスクリプト
├── requirements.txt     # 依存パッケージ
├── src/
│   ├── __init__.py
│   ├── config.py        # 設定読み込み
│   ├── detector.py      # MediaPipe FaceMesh 検出器
│   ├── video_io.py      # 動画 I/O
│   └── csv_logger.py    # CSV ログ出力
├── input/               # 入力 MP4 ファイルをここに配置
└── output/              # 出力先（自動生成）
```

## セットアップ

```bash
python3 -m venv .venv

source .venv/bin/activate

pip install -r requirements.txt
```

## 使い方

1. `input/` ディレクトリに処理したい `.mp4` ファイルを配置する
2. 必要であれば `config.yaml` を編集してパラメータを調整する
3. スクリプトを実行する

```bash
python main.py
```

別の設定ファイルを使う場合:

```bash
python main.py --config path/to/other_config.yaml
```

## 出力

| ファイル | 説明 |
|---|---|
| `output/<name>_landmarks.mp4` | ランドマーク描画済み動画 |
| `output/<name>_landmarks.csv` | ランドマーク座標（`save_csv: true` の場合） |

CSV フォーマット:

```
frame_index, face_index, landmark_index, x_norm, y_norm, z_norm
```

- `x_norm`, `y_norm` : 画像幅・高さで正規化した座標 [0, 1]
- `z_norm` : 深度（正規化済み，相対値）

## 設定ファイル (`config.yaml`) の主なパラメータ

| セクション | パラメータ | 説明 |
|---|---|---|
| `video` | `width`, `height` | 出力解像度 (px) |
| `video` | `fps` | 出力フレームレート |
| `face_mesh` | `max_num_faces` | 同時検出する最大顔数 |
| `face_mesh` | `refine_landmarks` | 虹彩・唇の詳細ランドマーク |
| `face_mesh` | `min_detection_confidence` | 検出信頼度閾値 |
| `drawing` | `draw_connections` | メッシュ接続線を描画するか |
| `logging` | `save_csv` | ランドマーク座標を CSV 保存するか |

## 動作確認環境

- Python 3.10+
- MediaPipe 0.10+
- OpenCV 4.8+
