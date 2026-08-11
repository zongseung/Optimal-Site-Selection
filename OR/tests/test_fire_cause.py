"""pfire.fire_cause 단위 테스트 — 원인 분류기(classify_cause).

검증 항목:
  - 각 범주 대표 문구가 올바른 라벨로 분류된다(전선→grid_electric, 입산자→human,
    용접→work_spark, 낙뢰→natural, 원인미상→unknown).
  - 우선순위(grid_electric > work_spark > ...): "벌채...전선줄 스파크" 처럼 설비·작업
    신호가 섞이면 config.CAUSE_PRIORITY 상 grid_electric 우선(과소계수 방지).
  - None/빈 문자열 → unknown.

데이터 의존 없이 합성 문자열로 결정적으로 검증한다(used_dataset 불필요).
"""
from __future__ import annotations

import numpy as np

from pfire import config, fire_cause


def test_classify_cause_categories():
    """각 범주 대표 문구가 기대 라벨로 분류된다."""
    resn = ["전선혼촉", "입산자실화", "용접불티", "낙뢰", "원인미상"]
    expected = ["grid_electric", "human", "work_spark", "natural", "unknown"]
    got = fire_cause.classify_cause(resn)
    assert list(got) == expected


def test_classify_cause_priority_grid_over_work():
    """혼합 문구 '벌채작업에 전선줄 스파크' → 우선순위상 grid_electric.

    '벌채'·'스파크'(work_spark) 와 '전선'(grid_electric) 키워드가 모두 있으나
    config.CAUSE_PRIORITY 가 grid_electric 을 먼저 검사하므로 grid_electric.
    """
    got = fire_cause.classify_cause(["벌채작업에 전선줄 스파크"])
    assert got[0] == "grid_electric"
    # 우선순위 사전 조건이 실제로 grid_electric 을 work_spark 보다 앞에 둠을 확인.
    pr = config.CAUSE_PRIORITY
    assert pr.index("grid_electric") < pr.index("work_spark")


def test_classify_cause_none_and_empty():
    """None·빈 문자열·공백 → unknown."""
    got = fire_cause.classify_cause([None, "", "   "])
    assert list(got) == ["unknown", "unknown", "unknown"]


def test_classify_cause_output_dtype_and_values():
    """출력 길이=입력, 값은 항상 config.CAUSE_PRIORITY 안의 라벨."""
    resn = ["특고압 전선 아크", "쓰레기소각", None, "예초기 스파크", "벼락"]
    got = fire_cause.classify_cause(resn)
    assert len(got) == len(resn)
    assert set(np.unique(got.astype(str))).issubset(set(config.CAUSE_PRIORITY))
