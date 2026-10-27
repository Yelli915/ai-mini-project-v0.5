"""LLM 프롬프트 모음"""
import re

COMPETITOR_ANALYST_SYSTEM = """당신은 반도체 기술 전략 분석가입니다.
주어진 근거 자료만 사용하여 특정 기업의 기술 현황을 분석합니다.
근거 자료에 없는 내용은 추측하지 말고 "근거 부족"이라고 쓰세요.

다음 형식으로 답하세요.
- 현황 요약: (3~5문장)
- TRL 추정: (1~9 중 하나, 또는 "판단 불가")
- 추정 근거: (어떤 자료의 어떤 내용 때문인지, 자료 번호 [n] 표시)
- 불확실한 점: (1~2문장)
"""

WRITER_SYSTEM = """당신은 SK hynix R&D 조직을 위한 기술 전략 보고서 작성자입니다.
주어진 근거 자료와 경쟁사 분석 결과만 사용하여 사용자 질문에 답하는 보고서를 Markdown으로 작성하세요.

보고서 구성
## 1. 요약
## 2. 기술 개요
## 3. 경쟁사별 현황 (기업별 TRL 추정과 근거)
## 4. 시사점
## 5. 한계와 불확실성

규칙
- 모든 주장 뒤에 근거 자료 번호 [n]를 붙입니다.
- 근거가 없는 내용은 쓰지 않습니다.
- 경쟁사 분석 결과가 없으면 3장에 기업별 TRL을 추정하지 말고, 이번 질문에서는 경쟁사 분석을 생략했다고만 씁니다.
"""


def format_evidence(evidence: list[dict]) -> str:
    """근거 자료를 번호를 붙인 텍스트로 바꿉니다. (LLM 입력용)"""
    if not evidence:
        return "(근거 자료 없음)"
    lines = []
    for i, e in enumerate(evidence, start=1):
        lines.append(f"[{i}] ({e['kind']}) {e['title']} | {e['source']}\n{e['content'][:1200]}")
    return "\n\n".join(lines)


def format_findings(findings: list[dict]) -> str:
    if not findings:
        return "(경쟁사 분석 결과 없음 — 이번 질문에서는 경쟁사 분석을 생략함)"
    return "\n\n".join(
        f"### {f['competitor']}\n" + (f"(분석 실패: {f['error']} — 한계와 불확실성에 누락 사실을 적을 것)" if f.get("error") else f["analysis"])
        for f in findings
    )


_CITATION = re.compile(r"\[(\d+(?:\s*,\s*\d+)*)\]")


def _shift_citation(m: re.Match, offset: int, count: int) -> str:
    """갈래 근거 수(count) 밖의 번호(없는 자료 인용)는 버리고 offset 만큼 옮깁니다."""
    nums = [int(n) for n in m.group(1).split(",")]
    if any(n >= 1000 for n in nums):      # [2027] 같은 연도는 인용이 아님 (갈래 근거는 많아야 5건)
        return m.group(0)
    kept = [str(n + offset) for n in nums if 1 <= n <= count]
    return "[" + ", ".join(kept) + "]" if kept else ""


def merge_findings_evidence(evidence: list[dict], findings: list[dict]) -> tuple[list[dict], list[dict]]:
    """
    경쟁사 분석은 갈래마다 자기 근거를 [1]부터 번호 매겨 인용합니다.
    writer 근거 목록 뒤에 갈래 근거를 이어 붙이고, 분석 속 [n]을 이어 붙인 위치의 번호로 바꿔
    writer 가 보는 번호와 실제 자료가 일치하도록 합니다.
    갈래 근거의 본문은 분석에 이미 요약돼 있어 제목·출처만 넘깁니다. (본문까지 넘기면 writer 입력 2.2k → 6.6k 토큰)
    """
    merged, renumbered = list(evidence), []
    for f in findings:
        offset, count = len(merged), len(f.get("evidence", []))
        merged += [{**e, "content": ""} for e in f.get("evidence", [])]
        if f.get("analysis"):
            f = {**f, "analysis": _CITATION.sub(lambda m: _shift_citation(m, offset, count), f["analysis"])}
        renumbered.append(f)
    return merged, renumbered
