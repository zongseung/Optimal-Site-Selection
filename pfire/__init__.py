"""pfire — 강원 가상 전주 산불 위험 추정 파이프라인.

설계: 물리식이 본체(I) + 체제별 전문가(MoE) 가중 보정 + 발화/노출 취약(S) +
그날 격자 기상(W) 를 곱셈 결합한 비지도 우선 위험 추정.
학습용 정답 라벨이 없으므로 발화점은 검증·임계값 앵커로만 사용한다.

모든 경로·상수는 :mod:`pfire.config` 에서만 가져온다(하드코딩 금지).
"""
from __future__ import annotations

import os as _os

from . import config as _config

# Rust exposure 커널(rayon)의 스레드 상한을 패키지 import 시점에 설정한다.
# rayon 전역풀은 첫 simulate_exposure 호출 때 초기화되며 그때 RAYON_NUM_THREADS 를
# 읽는다. 커널은 항상 pfire 를 거쳐 호출되므로 여기서 먼저 설정하면 반드시 적용된다.
# setdefault: 외부에서 명시적으로 지정한 값(예: 벤치마크 단일스레드)은 존중한다.
_os.environ.setdefault("RAYON_NUM_THREADS", str(_config.N_THREADS))

__all__ = [
    "config",
    "geo",
    "io",
    "regimes",
    "experts",
    "weather",
    "hazard",
    "hierarchy",
    "posterior",
    "exposure",
    "exposure_engine",
    "calibrate",
    "validate",
    "submit",
]
