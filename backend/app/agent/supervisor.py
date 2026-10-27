"""
Supervisor
──────────────────────────────────────────────────────────────
작업자(worker) 노드 중 다음에 무엇을 실행할지 결정합니다.
고정 순서표 대신 현재 state를 보고 매번 다음 작업을 고릅니다. (위에서부터 처음 맞는 규칙 적용)

  0. 비용 상한 도달               → writer (LLM 없이 정리본)
  1. 판단 횟수 상한 도달          → writer
  2. 아직 검색한 적 없음          → researcher
  3. 최신 정보 질문 + 날짜 있는 웹 근거 없음 + 재검색 여지 → researcher (뉴스 재검색)
  4. 기업 비교 질문 + 분석 결과 없음 → competitor_analyst
  5. 그 외                        → writer

작업자 목록
  researcher          질문에 필요한 근거 자료 수집 (문서 + 웹)
  competitor_analyst  분석 대상 기업별 현황 분석
  writer              보고서 작성
"""
from __future__ import annotations

import re

from langgraph.types import Send

from app import budget
from app.agent.state import AgentState
from app.agent.tracing import traced
from app.events import Route, emit
from app.retrieval import web_search_enabled

WORKERS = ["researcher", "competitor_analyst", "writer"]

# 정상 경로 중 가장 긴 것(researcher → researcher → competitor_analyst → writer)의 판단 수
MAX_DECISIONS = 4
# 검색 상한 : 일반 검색 1회 + 뉴스 재검색 1회 (같은 뉴스 검색을 반복해도 결과가 같음 — days=90 측정)
MAX_RESEARCH = 2

# 부분 문자열 오인 방지 : "새로운"(형용사) ≠ "새로 발표", "대비책·대비하다"(준비) ≠ "A 대비 B", "vs"는 단어 단독일 때만
RECENT_PATTERN = re.compile(r"최근|최신|새로(?!운)|신규|소식")
COMPARE_PATTERN = re.compile(r"비교|대비(?!책|하|해)|(?<![a-z])vs(?![a-z])", re.IGNORECASE)


def is_recent_question(query: str) -> bool:
    return bool(RECENT_PATTERN.search(query))


def is_compare_question(query: str, competitors: list[str]) -> bool:
    q = query.lower()
    mentioned = sum(c.lower() in q for c in competitors)
    return mentioned >= 2 or bool(COMPARE_PATTERN.search(query))


def decide(state: AgentState) -> tuple[str, str]:
    """(다음 작업, 이유) 를 돌려줍니다."""
    query = state["query"]
    research_count = state.get("research_count", 0)
    evidence = state.get("evidence", [])
    dated_web = sum(1 for e in evidence if e["kind"] == "web" and e.get("date"))

    if budget.exceeded():
        b = budget.current()
        return "writer", f"비용 상한 도달(누적 ${b.spent_usd:.5f} / 상한 ${b.limit_usd:.5f}) — 남은 작업 생략, 모은 결과로 정리"
    if state.get("decision_count", 0) >= MAX_DECISIONS:
        return "writer", f"판단 횟수 상한({MAX_DECISIONS}회) 도달 — 모은 근거로 바로 작성"
    if research_count == 0:
        return "researcher", "아직 모은 근거가 없어 먼저 검색"

    recent = is_recent_question(query)
    if recent and dated_web == 0:
        if not web_search_enabled():
            recent_note = "최신 정보 질문이지만 웹 검색 불가(TAVILY_API_KEY 없음)라 재검색 생략"
        elif research_count < MAX_RESEARCH:
            return "researcher", f"최신 정보 질문인데 날짜 있는 웹 근거가 0건이라 뉴스로 재검색 ({research_count + 1}/{MAX_RESEARCH}회)"
        else:
            recent_note = f"재검색 상한({MAX_RESEARCH}회) 도달 — 날짜 있는 웹 근거 없이 진행"
    elif recent:
        recent_note = f"최신 정보 질문 — 날짜 있는 웹 근거 {dated_web}건 확보"
    else:
        recent_note = ""

    if state.get("competitors") and is_compare_question(query, state["competitors"]):
        findings = state.get("competitor_findings")
        if not findings:
            return "competitor_analyst", f"기업 비교 질문인데 경쟁사 분석 결과가 아직 없음 — {len(state['competitors'])}개 기업 동시 분석"
        ok = sum(1 for f in findings if not f.get("error"))
        return "writer", f"기업 비교 질문 — 경쟁사 분석 {ok}/{len(findings)}건 성공, 작성"

    if recent_note:
        return "writer", f"{recent_note}, 기업 비교 요청이 없어 경쟁사 분석 생략"
    return "writer", "단일 사실 질문 — 기업 비교·최신 정보 요청이 없어 경쟁사 분석 생략"


@traced("supervisor")
async def supervisor(state: AgentState) -> dict:
    next_worker, reason = decide(state)
    emit(Route(source="supervisor", target=next_worker, reason=reason))
    return {"next": next_worker, "decision_count": state.get("decision_count", 0) + 1}


def route_from_supervisor(state: AgentState):
    """
    조건부 엣지: Supervisor가 정한 다음 노드 이름을 돌려줍니다.
    competitor_analyst 는 기업마다 Send 로 갈래를 나눠 동시에 실행합니다. (Fan-out)
    """
    if state["next"] == "competitor_analyst":
        return [Send("competitor_analyst", {**state, "competitor": c}) for c in state["competitors"]]
    return state["next"]
