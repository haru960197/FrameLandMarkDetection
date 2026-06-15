"""
src/config.py
設定ファイル (config.yaml) の読み込みとデータクラスへの変換
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import List, Tuple

import yaml


# ---------------------------------------------------------------------------
# データクラス群
# ---------------------------------------------------------------------------

@dataclass
class VideoConfig:
    width: int = 640
    height: int = 480
    fps: float = 90.0
    fourcc: str = "mp4v"


@dataclass
class FaceMeshConfig:
    static_image_mode: bool = False
    max_num_faces: int = 1
    refine_landmarks: bool = True
    min_detection_confidence: float = 0.5
    min_tracking_confidence: float = 0.5


@dataclass
class DrawingConfig:
    landmark_color: Tuple[int, int, int] = (0, 255, 0)
    connection_color: Tuple[int, int, int] = (0, 200, 100)
    landmark_radius: int = 1
    landmark_thickness: int = -1
    connection_thickness: int = 1
    draw_connections: bool = True
    overlay_alpha: float = 0.8


@dataclass
class LoggingConfig:
    level: str = "INFO"
    save_csv: bool = True


@dataclass
class AppConfig:
    input_dir: str = "input"
    output_dir: str = "output"
    video: VideoConfig = field(default_factory=VideoConfig)
    face_mesh: FaceMeshConfig = field(default_factory=FaceMeshConfig)
    drawing: DrawingConfig = field(default_factory=DrawingConfig)
    logging: LoggingConfig = field(default_factory=LoggingConfig)


# ---------------------------------------------------------------------------
# ローダ
# ---------------------------------------------------------------------------

def _to_tuple(lst) -> tuple:
    """YAMLのリストをタプルに変換する。"""
    return tuple(lst) if lst is not None else None


def load_config(config_path: str = "config.yaml") -> AppConfig:
    """
    指定されたYAMLファイルを読み込み、AppConfigを返す。

    Parameters
    ----------
    config_path : str
        設定ファイルのパス（デフォルト: 'config.yaml'）

    Returns
    -------
    AppConfig
        読み込まれた設定オブジェクト
    """
    if not os.path.exists(config_path):
        raise FileNotFoundError(
            f"設定ファイルが見つかりません: {config_path}\n"
            "プロジェクトルートから実行しているか確認してください。"
        )

    with open(config_path, "r", encoding="utf-8") as f:
        raw = yaml.safe_load(f)

    # --- video ---
    v = raw.get("video", {})
    video = VideoConfig(
        width=int(v.get("width", 640)),
        height=int(v.get("height", 480)),
        fps=float(v.get("fps", 90.0)),
        fourcc=str(v.get("fourcc", "mp4v")),
    )

    # --- face_mesh ---
    fm = raw.get("face_mesh", {})
    face_mesh = FaceMeshConfig(
        static_image_mode=bool(fm.get("static_image_mode", False)),
        max_num_faces=int(fm.get("max_num_faces", 1)),
        refine_landmarks=bool(fm.get("refine_landmarks", True)),
        min_detection_confidence=float(fm.get("min_detection_confidence", 0.5)),
        min_tracking_confidence=float(fm.get("min_tracking_confidence", 0.5)),
    )

    # --- drawing ---
    d = raw.get("drawing", {})
    drawing = DrawingConfig(
        landmark_color=_to_tuple(d.get("landmark_color", [0, 255, 0])),
        connection_color=_to_tuple(d.get("connection_color", [0, 200, 100])),
        landmark_radius=int(d.get("landmark_radius", 1)),
        landmark_thickness=int(d.get("landmark_thickness", -1)),
        connection_thickness=int(d.get("connection_thickness", 1)),
        draw_connections=bool(d.get("draw_connections", True)),
        overlay_alpha=float(d.get("overlay_alpha", 0.8)),
    )

    # --- logging ---
    lg = raw.get("logging", {})
    logging_cfg = LoggingConfig(
        level=str(lg.get("level", "INFO")),
        save_csv=bool(lg.get("save_csv", True)),
    )

    return AppConfig(
        input_dir=str(raw.get("input_dir", "input")),
        output_dir=str(raw.get("output_dir", "output")),
        video=video,
        face_mesh=face_mesh,
        drawing=drawing,
        logging=logging_cfg,
    )
