"""
src/csv_logger.py
ランドマーク座標を CSV に保存するロガー
"""

from __future__ import annotations

import csv
from pathlib import Path
from typing import List, Tuple


class LandmarkCSVLogger:
    """
    検出されたランドマーク座標を CSV ファイルに書き出すロガー。

    CSV フォーマット:
    frame_index, face_index, landmark_index, x_norm, y_norm, z_norm

    Parameters
    ----------
    output_path : str | Path
        出力 CSV ファイルのパス
    """

    HEADER = ["frame_index", "face_index", "landmark_index",
               "x_norm", "y_norm", "z_norm"]

    def __init__(self, output_path: str | Path) -> None:
        self.output_path = Path(output_path)
        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        self._file = open(self.output_path, "w", newline="", encoding="utf-8")
        self._writer = csv.writer(self._file)
        self._writer.writerow(self.HEADER)

    def log(
        self,
        frame_index: int,
        landmarks_list: list[list[tuple[float, float, float]]],
    ) -> None:
        """
        1フレーム分のランドマーク座標をログに書き出す。

        Parameters
        ----------
        frame_index : int
            現在のフレームインデックス
        landmarks_list : list[list[tuple[float, float, float]]]
            各顔のランドマーク座標リスト [(x, y, z), ...]
        """
        for face_idx, face_lm in enumerate(landmarks_list):
            for lm_idx, (x, y, z) in enumerate(face_lm):
                self._writer.writerow(
                    [frame_index, face_idx, lm_idx,
                     f"{x:.6f}", f"{y:.6f}", f"{z:.6f}"]
                )

    def close(self) -> None:
        """ファイルをクローズする。"""
        self._file.close()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()
