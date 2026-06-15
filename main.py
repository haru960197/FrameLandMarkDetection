#!/usr/bin/env python3
"""
main.py
FrameLandMarkDetection メインエントリポイント

使い方:
    python main.py [--config CONFIG_PATH]

    --config : 設定ファイルのパス（デフォルト: config.yaml）

入力ディレクトリ (config.yaml の input_dir) に .mp4 ファイルを置くと、
出力ディレクトリ (config.yaml の output_dir) に
ランドマーク描画済み動画 (*_landmarks.mp4) を出力します。
save_csv: true の場合、同じ出力ディレクトリに CSV も保存されます。
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path

from src import (
    FaceLandmarkDetector,
    LandmarkCSVLogger,
    VideoReader,
    VideoWriter,
    collect_video_files,
    load_config,
    make_output_path,
)


# ---------------------------------------------------------------------------
# ロガーセットアップ
# ---------------------------------------------------------------------------

def setup_logger(level: str) -> logging.Logger:
    numeric_level = getattr(logging, level.upper(), logging.INFO)
    logging.basicConfig(
        level=numeric_level,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    return logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# 1動画の処理
# ---------------------------------------------------------------------------

def process_video(
    input_path: Path,
    output_path: Path,
    cfg,
    logger: logging.Logger,
) -> None:
    """
    単一動画ファイルに対してランドマーク検出を行い、結果を書き出す。

    Parameters
    ----------
    input_path : Path
        入力動画ファイルパス
    output_path : Path
        出力動画ファイルパス
    cfg : AppConfig
        アプリケーション設定
    logger : logging.Logger
        ロガーインスタンス
    """
    logger.info(f"処理開始: {input_path.name}")

    # CSV ロガーの準備
    csv_logger = None
    if cfg.logging.save_csv:
        csv_path = output_path.with_suffix(".csv")
        csv_logger = LandmarkCSVLogger(csv_path)
        logger.info(f"  CSV 出力先: {csv_path}")

    try:
        with VideoReader(str(input_path), cfg.video) as reader:
            logger.info(
                f"  入力動画: {reader.src_width}x{reader.src_height} "
                f"@ {reader.src_fps:.1f}fps, "
                f"{reader.total_frames}フレーム"
            )
            if reader.needs_resize:
                logger.info(
                    f"  リサイズ: {reader.src_width}x{reader.src_height}"
                    f" → {cfg.video.width}x{cfg.video.height}"
                )

            with VideoWriter(str(output_path), cfg.video) as writer:
                with FaceLandmarkDetector(cfg) as detector:
                    start_time = time.perf_counter()
                    n_detected = 0

                    for frame_idx, frame in reader.read_frames():
                        # ランドマーク検出 & 描画
                        annotated, landmarks_list = detector.process_frame(frame)

                        # 検出フレームカウント
                        if landmarks_list:
                            n_detected += 1

                        # CSV ログ
                        if csv_logger is not None:
                            csv_logger.log(frame_idx, landmarks_list)

                        # 書き込み
                        writer.write(annotated)

                        # 進捗表示（100フレームごと）
                        if (frame_idx + 1) % 100 == 0:
                            elapsed = time.perf_counter() - start_time
                            fps_proc = (frame_idx + 1) / elapsed
                            logger.info(
                                f"  [{frame_idx + 1}/{reader.total_frames}] "
                                f"処理速度: {fps_proc:.1f} fps"
                            )

                    elapsed_total = time.perf_counter() - start_time
                    total_frames = frame_idx + 1 if reader.total_frames > 0 else 0
                    detection_rate = (
                        n_detected / total_frames * 100
                        if total_frames > 0 else 0.0
                    )

                    logger.info(
                        f"  完了: {total_frames}フレーム処理, "
                        f"検出率: {detection_rate:.1f}%, "
                        f"所要時間: {elapsed_total:.1f}秒"
                    )
                    logger.info(f"  出力: {output_path}")

    finally:
        if csv_logger is not None:
            csv_logger.close()


# ---------------------------------------------------------------------------
# メイン
# ---------------------------------------------------------------------------

def main() -> int:
    parser = argparse.ArgumentParser(
        description="MediaPipe による顔ランドマーク検出システム"
    )
    parser.add_argument(
        "--config",
        default="config.yaml",
        help="設定ファイルのパス (デフォルト: config.yaml)",
    )
    args = parser.parse_args()

    # 設定読み込み
    cfg = load_config(args.config)
    logger = setup_logger(cfg.logging.level)

    logger.info("=" * 60)
    logger.info("FrameLandMarkDetection 起動")
    logger.info(f"  設定ファイル: {args.config}")
    logger.info(
        f"  動画設定: {cfg.video.width}x{cfg.video.height} @ {cfg.video.fps}fps"
    )
    logger.info("=" * 60)

    # 出力ディレクトリを作成
    Path(cfg.output_dir).mkdir(parents=True, exist_ok=True)

    # 入力動画ファイルを収集
    try:
        video_files = collect_video_files(cfg.input_dir)
    except FileNotFoundError as e:
        logger.error(str(e))
        return 1

    logger.info(f"処理対象: {len(video_files)} ファイル")

    # 各動画を処理
    errors = []
    for input_path in video_files:
        output_path = make_output_path(input_path, cfg.output_dir)
        try:
            process_video(input_path, output_path, cfg, logger)
        except Exception as e:
            logger.error(f"エラー [{input_path.name}]: {e}")
            errors.append(input_path.name)

    logger.info("=" * 60)
    if errors:
        logger.warning(f"エラーが発生したファイル: {errors}")
        return 1

    logger.info("全ての処理が完了しました。")
    return 0


if __name__ == "__main__":
    sys.exit(main())
