"""
src/video_io.py
動画の読み込みと書き込みを担当するクラス群
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Iterator

import cv2
import numpy as np

from .config import AppConfig, VideoConfig


class VideoReader:
    """
    OpenCV を用いた動画読み込みラッパー。

    Parameters
    ----------
    path : str | Path
        入力動画ファイルのパス
    cfg : VideoConfig
        動画設定（フレームサイズのリサイズに使用）
    """

    def __init__(self, path: str | Path, cfg: VideoConfig) -> None:
        self.path = str(path)
        self.cfg = cfg
        self.cap = cv2.VideoCapture(self.path)

        if not self.cap.isOpened():
            raise IOError(f"動画ファイルを開けません: {self.path}")

        # 実際の動画メタデータを取得
        self.src_width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.src_height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.src_fps = self.cap.get(cv2.CAP_PROP_FPS)
        self.total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))

    @property
    def needs_resize(self) -> bool:
        """config の解像度と実際のサイズが異なる場合 True。"""
        return (
            self.src_width != self.cfg.width
            or self.src_height != self.cfg.height
        )

    def read_frames(self) -> Iterator[tuple[int, np.ndarray]]:
        """
        フレームをイテレータとして返す。

        Yields
        ------
        (frame_index, frame_bgr) : tuple[int, np.ndarray]
        """
        frame_idx = 0
        while True:
            ret, frame = self.cap.read()
            if not ret:
                break
            if self.needs_resize:
                frame = cv2.resize(
                    frame,
                    (self.cfg.width, self.cfg.height),
                    interpolation=cv2.INTER_LINEAR,
                )
            yield frame_idx, frame
            frame_idx += 1

    def close(self) -> None:
        self.cap.release()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()


class VideoWriter:
    """
    OpenCV を用いた動画書き込みラッパー。

    Parameters
    ----------
    path : str | Path
        出力動画ファイルのパス
    cfg : VideoConfig
        動画設定（FPS・解像度・コーデック）
    """

    def __init__(self, path: str | Path, cfg: VideoConfig) -> None:
        self.path = str(path)
        self.cfg = cfg

        fourcc = cv2.VideoWriter_fourcc(*cfg.fourcc)
        os.makedirs(os.path.dirname(os.path.abspath(self.path)), exist_ok=True)

        self.writer = cv2.VideoWriter(
            self.path,
            fourcc,
            cfg.fps,
            (cfg.width, cfg.height),
        )

        if not self.writer.isOpened():
            raise IOError(
                f"動画ライターを初期化できません: {self.path}\n"
                f"  コーデック: {cfg.fourcc}, fps: {cfg.fps}, "
                f"  サイズ: {cfg.width}x{cfg.height}"
            )

    def write(self, frame: np.ndarray) -> None:
        """1フレームを書き込む。"""
        self.writer.write(frame)

    def close(self) -> None:
        self.writer.release()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()


# ---------------------------------------------------------------------------
# ユーティリティ関数
# ---------------------------------------------------------------------------

def collect_video_files(input_dir: str) -> list[Path]:
    """
    入力ディレクトリ内の .mp4 ファイルを収集して返す。

    Parameters
    ----------
    input_dir : str
        入力ディレクトリのパス

    Returns
    -------
    list[Path]
        ソート済みの動画ファイルパスリスト
    """
    extensions = {".mp4", ".MP4"}
    p = Path(input_dir)
    if not p.exists():
        raise FileNotFoundError(f"入力ディレクトリが存在しません: {input_dir}")

    files = sorted([f for f in p.iterdir() if f.suffix in extensions])
    if not files:
        raise FileNotFoundError(
            f"入力ディレクトリに .mp4 ファイルが見つかりません: {input_dir}"
        )
    return files


def make_output_path(input_path: Path, output_dir: str) -> Path:
    """
    入力ファイルパスから出力ファイルパスを生成する。
    例: input/test.mp4 → output/test_landmarks.mp4

    Parameters
    ----------
    input_path : Path
        入力動画のパス
    output_dir : str
        出力ディレクトリのパス

    Returns
    -------
    Path
        出力ファイルのパス
    """
    stem = input_path.stem
    return Path(output_dir) / f"{stem}_landmarks.mp4"
