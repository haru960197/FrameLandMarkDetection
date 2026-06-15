"""
src/detector.py
MediaPipe FaceLandmarker (Tasks API) を用いた顔ランドマーク検出クラス

MediaPipe 0.10 以降は mp.solutions が廃止されており、
Tasks API (mediapipe.tasks) を使用する必要があります。
"""

from __future__ import annotations

import cv2
import mediapipe as mp
import numpy as np
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from mediapipe.tasks.python.vision import drawing_utils as mp_drawing
from mediapipe.tasks.python.vision import drawing_styles as mp_drawing_styles
from mediapipe.tasks.python.vision.face_landmarker import FaceLandmarksConnections

from .config import AppConfig, DrawingConfig, FaceMeshConfig


class FaceLandmarkDetector:
    """
    MediaPipe FaceLandmarker (Tasks API) を用いた顔ランドマーク検出器。

    Attributes
    ----------
    cfg : AppConfig
        アプリケーション全体の設定
    landmarker : vision.FaceLandmarker
        MediaPipe FaceLandmarker インスタンス
    """

    def __init__(self, cfg: AppConfig) -> None:
        self.cfg = cfg
        fmc: FaceMeshConfig = cfg.face_mesh

        base_options = python.BaseOptions(
            model_asset_path=fmc.model_path,
        )
        options = vision.FaceLandmarkerOptions(
            base_options=base_options,
            num_faces=fmc.max_num_faces,
            min_face_detection_confidence=fmc.min_detection_confidence,
            min_face_presence_confidence=fmc.min_presence_confidence,
            min_tracking_confidence=fmc.min_tracking_confidence,
            output_face_blendshapes=False,
            output_facial_transformation_matrixes=False,
        )
        self.landmarker = vision.FaceLandmarker.create_from_options(options)

    def close(self) -> None:
        """リソースを解放する。"""
        self.landmarker.close()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()

    # ------------------------------------------------------------------
    # メイン処理
    # ------------------------------------------------------------------

    def process_frame(
        self,
        frame_bgr: np.ndarray,
    ) -> tuple[np.ndarray, list[list[tuple[float, float, float]]]]:
        """
        1フレームに対してランドマーク検出と描画を行う。

        Parameters
        ----------
        frame_bgr : np.ndarray
            OpenCV BGR 形式の入力フレーム

        Returns
        -------
        annotated : np.ndarray
            ランドマーク描画済みフレーム（BGR）
        landmarks_list : list[list[tuple[float, float, float]]]
            検出された各顔のランドマーク一覧 [(x_norm, y_norm, z_norm), ...]
            x_norm, y_norm は [0, 1] に正規化された座標
        """
        dc: DrawingConfig = self.cfg.drawing

        # BGR → RGB（MediaPipe は RGB を受け付ける）
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)

        # MediaPipe Image オブジェクトに変換
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)

        # ランドマーク検出
        detection_result = self.landmarker.detect(mp_image)

        annotated = frame_bgr.copy()
        landmarks_list: list[list[tuple[float, float, float]]] = []

        if not detection_result.face_landmarks:
            return annotated, landmarks_list

        h, w = frame_bgr.shape[:2]

        for face_landmarks in detection_result.face_landmarks:
            # ランドマーク座標を収集
            face_lm = [
                (lm.x, lm.y, lm.z)
                for lm in face_landmarks
            ]
            landmarks_list.append(face_lm)

            if dc.draw_connections:
                # テッセレーション（メッシュ全体）描画
                mp_drawing.draw_landmarks(
                    image=annotated,
                    landmark_list=face_landmarks,
                    connections=FaceLandmarksConnections.FACE_LANDMARKS_TESSELATION,
                    landmark_drawing_spec=None,
                    connection_drawing_spec=mp_drawing_styles
                    .get_default_face_mesh_tesselation_style(),
                )
                # 輪郭描画
                mp_drawing.draw_landmarks(
                    image=annotated,
                    landmark_list=face_landmarks,
                    connections=FaceLandmarksConnections.FACE_LANDMARKS_CONTOURS,
                    landmark_drawing_spec=None,
                    connection_drawing_spec=mp_drawing_styles
                    .get_default_face_mesh_contours_style(),
                )
                # 虹彩描画
                mp_drawing.draw_landmarks(
                    image=annotated,
                    landmark_list=face_landmarks,
                    connections=FaceLandmarksConnections.FACE_LANDMARKS_LEFT_IRIS,
                    landmark_drawing_spec=None,
                    connection_drawing_spec=mp_drawing_styles
                    .get_default_face_mesh_iris_connections_style(),
                )
                mp_drawing.draw_landmarks(
                    image=annotated,
                    landmark_list=face_landmarks,
                    connections=FaceLandmarksConnections.FACE_LANDMARKS_RIGHT_IRIS,
                    landmark_drawing_spec=None,
                    connection_drawing_spec=mp_drawing_styles
                    .get_default_face_mesh_iris_connections_style(),
                )

            # 個別ランドマーク点の描画
            lm_color = tuple(int(c) for c in dc.landmark_color)
            for lm in face_landmarks:
                cx = int(lm.x * w)
                cy = int(lm.y * h)
                cv2.circle(
                    annotated,
                    (cx, cy),
                    dc.landmark_radius,
                    lm_color,
                    dc.landmark_thickness,
                )

        # オーバーレイ合成（元フレームとブレンド）
        if dc.overlay_alpha < 1.0:
            annotated = cv2.addWeighted(
                annotated, dc.overlay_alpha,
                frame_bgr, 1.0 - dc.overlay_alpha,
                0,
            )

        return annotated, landmarks_list
