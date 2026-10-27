"""
LLM 호출 비용 상한
──────────────────────────────────────────────────────────────
실행(요청) 1회의 누적 비용을 ContextVar 에 담아, llm.py 와 노드가 함께 봅니다.
  - main.py 가 실행을 시작할 때 start() 로 추적기를 만듭니다.
  - llm.py 는 호출 "전"에 상한 도달 여부를 확인하고(도달했으면 BudgetExceeded),
    호출 "후"에 실제 비용을 더합니다.
  - 비용은 호출이 끝나야 알 수 있으므로, 상한 직전의 호출 1회만큼은 넘을 수 있습니다.
    병렬 갈래가 동시에 확인을 통과하면 그 갈래 수만큼 넘을 수 있습니다.
ContextVar 는 요청마다 따로 존재하고, LangGraph 가 노드를 실행하는 task 에도 그대로 전달됩니다.
(추적기 객체 자체를 공유하므로 병렬 갈래가 더한 비용도 한곳에 모입니다.)
"""
from __future__ import annotations

from contextvars import ContextVar
from dataclasses import dataclass


class BudgetExceeded(Exception):
    pass


@dataclass
class Budget:
    limit_usd: float          # 0 이하이면 상한 없음
    spent_usd: float = 0.0
    notified: bool = False    # 상한 도달 알림을 한 번만 보내기 위한 표시

    def exceeded(self) -> bool:
        return self.limit_usd > 0 and self.spent_usd >= self.limit_usd


_current: ContextVar[Budget | None] = ContextVar("budget", default=None)


def start(limit_usd: float) -> Budget:
    budget = Budget(limit_usd)
    _current.set(budget)
    return budget


def current() -> Budget | None:
    return _current.get()


def exceeded() -> bool:
    budget = current()
    return budget is not None and budget.exceeded()
