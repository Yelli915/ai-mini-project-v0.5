"""
작업자(worker) 노드와 평가 노드
──────────────────────────────────────────────────────────────
  researcher          근거 자료 수집
  competitor_analyst  분석 대상 기업별 현황 분석
  writer              보고서 작성 (답변 스트리밍)
  step_evaluator      작성된 보고서 평가
"""
from __future__ import annotations

import asyncio
import time

from langchain_core.messages import HumanMessage, SystemMessage

from app.agent.prompts import (
    COMPETITOR_ANALYST_SYSTEM,
    WRITER_SYSTEM,
    format_evidence,
    format_findings,
    merge_findings_evidence,
)
from app import budget
from app.agent.state import AgentState
from app.agent.tracing import traced
from app.budget import BudgetExceeded
from app.config import settings
from app.events import BranchEnd, BranchStart, Evaluation, Progress, Token, emit
from app.llm import ainvoke_tracked, astream_tracked
from app.retrieval import index_exists, search_docs, search_web, web_search_enabled


# ──────────────────────────────────────────────────────────────
# researcher : 근거 자료 수집
# ──────────────────────────────────────────────────────────────
@traced("researcher")
async def researcher(state: AgentState) -> dict:
    item_id, query = state["item_id"], state["query"]
    count = state.get("research_count", 0) + 1

    # 재검색 : 문서는 같은 결과라 다시 찾지 않고, 웹을 뉴스로 찾아 기존 근거에 이어 붙입니다.
    if count > 1:
        web = await search_web(f"{state['item_name']} {query}", max_results=3, topic="news")
        seen = {e["source"] for e in state.get("evidence", [])}
        web = [e for e in web if e["source"] not in seen]      # 1차 검색과 같은 기사는 번호가 둘로 나뉘지 않게 제외
        dated = sum(1 for e in web if e.get("date"))
        emit(Progress(node="researcher", message=f"뉴스 재검색 {len(web)}건 (날짜 있는 근거 {dated}건)"))
        return {"evidence": state.get("evidence", []) + web, "research_count": count}

    if not index_exists(item_id):
        emit(Progress(node="researcher", message=f"'{item_id}' 문서 인덱스가 없습니다. scripts/ingest.py 를 먼저 실행하세요."))
    docs = await search_docs(item_id, query, k=4)
    emit(Progress(node="researcher", message=f"문서 검색 {len(docs)}건"))

    web = []
    if web_search_enabled():
        web = await search_web(f"{state['item_name']} {query}", max_results=3)
        emit(Progress(node="researcher", message=f"웹 검색 {len(web)}건"))
    else:
        emit(Progress(node="researcher", message="TAVILY_API_KEY 가 없어 웹 검색을 건너뜁니다."))

    return {"evidence": docs + web, "research_count": count}


# ──────────────────────────────────────────────────────────────
# competitor_analyst : 분석 대상 기업별 현황 분석
#  - 기업마다 근거를 추가로 찾고, LLM으로 현황과 TRL을 추정합니다.
#  - 기업 하나의 분석을 "갈래(branch)"라고 부르며, 갈래마다 시작/종료 이벤트를 남깁니다.
# ──────────────────────────────────────────────────────────────
async def analyze_competitor(state: AgentState, competitor: str) -> dict:
    """기업 하나를 분석해 {"competitor": 기업명, "analysis": 분석 내용, "sources": [...]} 를 돌려줍니다."""
    item_name = state["item_name"]
    search_query = f"{competitor} {item_name}"

    docs = await search_docs(state["item_id"], search_query, k=3)
    web = await search_web(search_query, max_results=2)
    evidence = docs + web

    messages = [
        SystemMessage(content=COMPETITOR_ANALYST_SYSTEM),
        HumanMessage(content=(
            f"분석 아이템: {item_name}\n분석 대상 기업: {competitor}\n\n"
            f"근거 자료:\n{format_evidence(evidence)}"
        )),
    ]
    analysis = await ainvoke_tracked(messages, node="competitor_analyst")
    return {
        "competitor": competitor,
        "analysis": analysis,
        "sources": [e["source"] for e in evidence],
        "evidence": evidence,      # writer 가 분석 속 [n] 번호를 같은 자료로 이어 받기 위해 보관
    }


@traced("competitor_analyst")
async def competitor_analyst(state: AgentState) -> dict:
    """
    갈래 하나 = 기업 하나. Supervisor 가 Send 로 기업 수만큼 동시에 실행합니다.
    시간 제한을 넘기거나 실패한 갈래는 error 를 기록하고, 나머지 갈래는 그대로 진행합니다.
    """
    competitor = state["competitor"]
    emit(BranchStart(node="competitor_analyst", branch=competitor))
    started = time.perf_counter()

    try:
        finding = await asyncio.wait_for(analyze_competitor(state, competitor), timeout=settings.branch_timeout_sec)
    except asyncio.TimeoutError:
        finding = {"competitor": competitor, "error": f"시간 제한 {settings.branch_timeout_sec:g}초 초과"}
    except Exception as e:      # 한 갈래의 실패가 실행 전체를 멈추지 않도록 기록만 남깁니다.
        finding = {"competitor": competitor, "error": f"{type(e).__name__}: {e}"}

    elapsed = int((time.perf_counter() - started) * 1000)
    if finding.get("error"):
        emit(Progress(node="competitor_analyst", message=f"{competitor} 분석 실패 — {finding['error']} (나머지 기업은 계속 진행)"))
    emit(BranchEnd(node="competitor_analyst", branch=competitor, duration_ms=elapsed))
    return {"competitor_findings": [finding]}


# ──────────────────────────────────────────────────────────────
# writer : 보고서 작성 (작성 중인 내용을 Frontend로 스트리밍)
# ──────────────────────────────────────────────────────────────
POOR_OUTPUT_SAMPLE = (
    "## 1. 요약\n해당 기술은 업계에서 많은 관심을 받고 있으며 향후 성장이 기대됩니다. "
    "여러 기업이 개발을 진행하고 있고 시장 상황에 따라 경쟁이 치열해질 것으로 보입니다.\n\n"
    "## 2. 기술 개요\n이 기술은 성능과 효율을 높이기 위한 차세대 기술입니다. "
    "다양한 분야에서 활용될 수 있으며 중요성이 계속 커지고 있습니다.\n\n"
    "## 3. 경쟁사별 현황\n주요 기업들은 각자의 전략에 따라 기술 개발에 힘쓰고 있습니다. "
    "각 기업의 기술 수준은 상당히 높은 편으로 판단됩니다.\n\n"
    "## 4. 시사점\n지속적인 투자와 연구 개발이 필요합니다.\n\n"
    "## 5. 한계와 불확실성\n시장 상황은 변할 수 있습니다.\n"
)


@traced("writer")
async def writer(state: AgentState) -> dict:
    # 점검용 테스트 모드: 근거 없이 일반론만 담긴 "부실한 결과"를 만듭니다.
    if state.get("test_mode"):
        emit(Progress(node="writer", message="[테스트 모드] 부실한 중간 결과를 생성합니다."))
        emit(Token(node="writer", text=POOR_OUTPUT_SAMPLE))
        return {"draft": POOR_OUTPUT_SAMPLE}

    evidence, findings = merge_findings_evidence(state.get("evidence", []), state.get("competitor_findings", []))
    messages = [
        SystemMessage(content=WRITER_SYSTEM),
        HumanMessage(content=(
            f"사용자 질문: {state['query']}\n"
            f"분석 아이템: {state['item_name']}\n"
            f"분석 대상 기업: {', '.join(state.get('competitors', []))}\n\n"
            f"근거 자료:\n{format_evidence(evidence)}\n\n"
            f"경쟁사 분석 결과:\n{format_findings(findings)}"
        )),
    ]
    try:
        draft = await astream_tracked(messages, node="writer")
    except BudgetExceeded as e:
        emit(Progress(node="writer", message=f"{e} — 모은 근거와 분석으로 정리본을 냅니다."))
        draft = budget_fallback_report(state["query"], evidence, findings)
        emit(Token(node="writer", text=draft))
    return {"draft": draft}


def budget_fallback_report(query: str, evidence: list[dict], findings: list[dict]) -> str:
    """비용 상한 도달 시 LLM 없이 지금까지 모은 분석과 근거 목록을 그대로 정리합니다."""
    lines = [
        "> ⚠️ 실행 1회의 LLM 비용 상한에 도달해, LLM 작성 없이 지금까지 모은 결과로 정리한 보고서입니다.",
        f"\n**질문** : {query}",
        "\n## 경쟁사 분석 결과",
        # format_findings 는 LLM용 지시문이 섞여 있어 사용자용 문구로 따로 정리합니다.
        "\n\n".join(
            f"### {f['competitor']}\n" + (f"(분석 실패: {f['error']})" if f.get("error") else f["analysis"])
            for f in findings
        ) or "(경쟁사 분석 결과 없음)",
        "\n## 근거 자료",
        *(f"- [{i}] ({e['kind']}) {e['title']} | {e['source']}" for i, e in enumerate(evidence, start=1)),
    ]
    return "\n".join(lines)


# ──────────────────────────────────────────────────────────────
# step_evaluator : 보고서 평가
#  - 통과하면 종료, 통과하지 못하면 writer 가 다시 작성합니다.
# ──────────────────────────────────────────────────────────────
MIN_REPORT_CHARS = 300


@traced("step_evaluator")
async def step_evaluator(state: AgentState) -> dict:
    draft = state.get("draft", "")
    attempt = state.get("eval_attempts", 0) + 1

    passed = len(draft.strip()) >= MIN_REPORT_CHARS
    reason = "보고서 분량 충족" if passed else f"보고서 분량 부족 ({len(draft.strip())}자)"

    emit(Evaluation(stage="report", passed=passed, reason=reason, attempt=attempt))
    return {"eval_passed": passed, "eval_reason": reason, "eval_attempts": attempt}


def route_after_evaluation(state: AgentState) -> str:
    # 비용 상한 도달 후에는 다시 써도 LLM을 부를 수 없으므로 재작성하지 않고 종료합니다.
    if not state.get("eval_passed") and budget.exceeded():
        emit(Progress(node="step_evaluator", message="비용 상한 도달로 재작성 없이 종료합니다."))
        return "end"
    return "end" if state.get("eval_passed") else "writer"
