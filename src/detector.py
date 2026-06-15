"""
src/detector.py
MediaPipe FaceMesh を用いた顔ランドマーク検出クラス
"""

from __future__ import annotations

import cv2
import mediapipe as mp
import numpy as np

from .config import AppConfig, DrawingConfig, FaceMeshConfig


class FaceLandmarkDetector:
    """
    MediaPipe FaceMesh を用いた顔ランドマーク検出器。

    Attributes
    ----------
    cfg : AppConfig
        アプリケーション全体の設定
    face_mesh : mp.solutions.face_mesh.FaceMesh
        MediaPipe FaceMesh インスタンス
    mp_drawing : module
        MediaPipe 描画ユーティリティ
    mp_face_mesh : module
        MediaPipe FaceMesh モジュール
    """

    def __init__(self, cfg: AppConfig) -> None:
        self.cfg = cfg
        fmc: FaceMeshConfig = cfg.face_mesh

        self.mp_face_mesh = mp.solutions.face_mesh
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_drawing_styles = mp.solutions.drawing_styles

        self.face_mesh = self.mp_face_mesh.FaceMesh(
            static_image_mode=fmc.static_image_mode,
            max_num_faces=fmc.max_num_faces,
            refine_landmarks=fmc.refine_landmarks,
            min_detection_confidence=fmc.min_detection_confidence,
            min_tracking_confidence=fmc.min_tracking_confidence,
        )

    def close(self) -> None:
        """リソースを解放する。"""
        self.face_mesh.close()

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

        # MediaPipe の書き込み保護を外す
        frame_rgb.flags.writeable = False
        results = self.face_mesh.process(frame_rgb)
        frame_rgb.flags.writeable = True

        annotated = frame_bgr.copy()
        landmarks_list: list[list[tuple[float, float, float]]] = []

        if results.multi_face_landmarks is None:
            return annotated, landmarks_list

        h, w = frame_bgr.shape[:2]

        for face_landmarks in results.multi_face_landmarks:
            # ランドマーク座標を収集
            face_lm = [
                (lm.x, lm.y, lm.z)
                for lm in face_landmarks.landmark
            ]
            landmarks_list.append(face_lm)

            if dc.draw_connections:
                # テッセレーション（メッシュ）描画
                self.mp_drawing.draw_landmarks(
                    image=annotated,
                    landmark_list=face_landmarks,
                    connections=self.mp_face_mesh.FACEMESH_TESSELATION,
                    landmark_drawing_spec=None,
                    connection_drawing_spec=self.mp_drawing_styles
                    .get_default_face_mesh_tesselation_style(),
                )
                # 輪郭描画
                self.mp_drawing.draw_landmarks(
                    image=annotated,
                    landmark_list=face_landmarks,
                    connections=self.mp_face_mesh.FACEMESH_CONTOURS,
                    landmark_drawing_spec=None,
                    connection_drawing_spec=self.mp_drawing_styles
                    .get_default_face_mesh_contours_style(),
                )
                # 虹彩描画（refine_landmarks=True の場合のみ有効）
                if self.cfg.face_mesh.refine_landmarks:
                    self.mp_drawing.draw_landmarks(
                        image=annotated,
                        landmark_list=face_landmarks,
                        connections=self.mp_face_mesh.FACEMESH_IRISES,
                        landmark_drawing_spec=None,
                        connection_drawing_spec=self.mp_drawing_styles
                        .get_default_face_mesh_iris_connections_style(),
                    )

            # 個別ランドマーク点の描画
            lm_color = tuple(int(c) for c in dc.landmark_color)
            for lm in face_landmarks.landmark:
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
