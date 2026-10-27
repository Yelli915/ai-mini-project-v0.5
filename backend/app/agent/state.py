"""
그래프 상태(State)
──────────────────────────────────────────────────────────────
모든 노드가 함께 읽고 쓰는 공유 데이터입니다.
노드는 상태 전체를 돌려주지 않고, "바꾸고 싶은 항목만" 담은 dict를 돌려줍니다.
LangGraph가 그 dict를 기존 상태에 반영합니다.
"""
from __future__ import annotations

import operator
from typing import Annotated, TypedDict


class AgentState(TypedDict, total=False):
    # ── 입력 ─────────────────────────────
    run_id: str
    item_id: str
    item_name: str
    competitors: list[str]      # 분석 대상 기업 (items.yaml)
    query: str                  # 사용자 질문
    test_mode: bool             # 점검용 테스트 모드 (True면 writer가 부실한 결과를 만듦)

    # ── Supervisor ──────────────────────
    next: str                   # 다음에 실행할 노드 이름
    decision_count: int         # Supervisor 판단 횟수 (무한 반복 방지)
    research_count: int         # researcher 실행 횟수 (2회째는 뉴스 재검색)

    # ── 작업 결과 ────────────────────────
    evidence: list[dict]                # researcher 가 모은 근거 자료
    competitor: str                     # Send 로 갈래 하나에 넘기는 분석 대상 기업
    # 갈래(기업)마다 결과를 하나씩 돌려주면 덮어쓰지 않고 이어 붙입니다. (Fan-in 규칙)
    competitor_findings: Annotated[list[dict], operator.add]
    draft: str                          # writer 가 작성한 보고서 초안

    # ── 평가 ─────────────────────────────
    eval_passed: bool
    eval_reason: str
    eval_attempts: int
