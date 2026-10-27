"""Coverage of an intent's investigative goals and relevant evidence (paper Section V-A).

* **IC(p)** -- investigative coverage: fraction of the intent's goals (``intent_goals.json``)
  addressed by at least one node.
* **EA(p)** -- evidence alignment: fraction of the intent's relevant evidence categories
  (``intent_taxonomy.json``) consumed by at least one node.

Both are deterministic lookups. The same coverage profile (which goals / categories are covered)
is the signal CADE uses at initialization, variation, selection and termination.

Implementation choices: destructive actions never count towards coverage (so the search is never
rewarded for adding them), and an intent with no relevant categories (``unknown``) has EA = 1.
"""

from __future__ import annotations

from dataclasses import dataclass

from app.playbook.vocabulary import IntentGoal, Vocabulary
from app.schemas.enums import EvidenceCategory
from app.schemas.playbook import Playbook
from app.schemas.taxonomy import IntentTaxonomy


@dataclass(frozen=True)
class Gap:
    kind: str  # "goal" | "category"
    key: str  # goal id or evidence category value


@dataclass(frozen=True)
class CoverageProfile:
    goal_ids: tuple[str, ...]
    goals_covered: tuple[bool, ...]
    categories: tuple[EvidenceCategory, ...]
    categories_covered: tuple[bool, ...]

    @property
    def ic(self) -> float:
        return sum(self.goals_covered) / len(self.goals_covered) if self.goals_covered else 1.0

    @property
    def ea(self) -> float:
        return sum(self.categories_covered) / len(self.categories_covered) if self.categories_covered else 1.0

    @property
    def vector(self) -> tuple[int, ...]:
        """The coverage-feature vector (goal bits then category bits)."""
        return tuple(int(b) for b in self.goals_covered + self.categories_covered)

    @property
    def complete(self) -> bool:
        return all(self.goals_covered) and all(self.categories_covered)

    @property
    def covered_count(self) -> int:
        """|c(p)|: how many goals and relevant categories the playbook addresses."""
        return sum(self.vector)

    def gaps(self) -> list[Gap]:
        """U(p): the goals and relevant evidence categories left unaddressed."""
        return [Gap("goal", g) for g, ok in zip(self.goal_ids, self.goals_covered) if not ok] + [
            Gap("category", c.value) for c, ok in zip(self.categories, self.categories_covered) if not ok
        ]

    def covered(self) -> list[Gap]:
        return [Gap("goal", g) for g, ok in zip(self.goal_ids, self.goals_covered) if ok] + [
            Gap("category", c.value) for c, ok in zip(self.categories, self.categories_covered) if ok
        ]


class CoverageModel:
    def __init__(self, vocabulary: Vocabulary, taxonomy: IntentTaxonomy) -> None:
        self.vocabulary = vocabulary
        self.taxonomy = taxonomy

    def goals(self, intent_label: str) -> list[IntentGoal]:
        return self.vocabulary.goals_for(intent_label)

    def relevant_categories(self, intent_label: str) -> tuple[EvidenceCategory, ...]:
        definition = self.taxonomy.get(intent_label)
        return tuple(definition.relevant_evidence) if definition else ()

    def profile(self, playbook: Playbook, intent_label: str) -> CoverageProfile:
        actions = [a for a in (self.vocabulary.get(n.action) for n in playbook.nodes) if a and not a.destructive]
        goals = self.goals(intent_label)
        categories = self.relevant_categories(intent_label)
        consumed = {c for a in actions for c in a.categories}
        return CoverageProfile(
            goal_ids=tuple(g.id for g in goals),
            goals_covered=tuple(any(self.vocabulary.addresses(a, g) for a in actions) for g in goals),
            categories=categories,
            categories_covered=tuple(c in consumed for c in categories),
        )

    def closers(self, gap: Gap, intent_label: str):
        """Non-destructive actions that would close ``gap``."""
        if gap.kind == "goal":
            goal = next(g for g in self.goals(intent_label) if g.id == gap.key)
            return self.vocabulary.actions_for_goal(goal)
        return self.vocabulary.actions_consuming(EvidenceCategory(gap.key))

    def closes(self, action, gap: Gap, intent_label: str) -> bool:
        """Whether ``action`` addresses ``gap`` (destructive actions never count, as in :meth:`profile`)."""
        if action is None or action.destructive:
            return False
        if gap.kind == "goal":
            goal = next((g for g in self.goals(intent_label) if g.id == gap.key), None)
            return goal is not None and self.vocabulary.addresses(action, goal)
        return EvidenceCategory(gap.key) in action.categories

    def all_gaps(self, intent_label: str) -> list[Gap]:
        """Every goal and relevant category of the intent (the gaps of an empty playbook)."""
        return [Gap("goal", g.id) for g in self.goals(intent_label)] + [
            Gap("category", c.value) for c in self.relevant_categories(intent_label)
        ]


def hamming(a: tuple[int, ...], b: tuple[int, ...]) -> float:
    """Normalised Hamming distance between two coverage vectors (0 = identical coverage)."""
    if not a:
        return 0.0
    return sum(x != y for x, y in zip(a, b)) / len(a)


def jaccard(a: tuple[int, ...], b: tuple[int, ...]) -> float:
    """Jaccard distance 1 - |a n b| / |a u b| between two coverage vectors (paper Section 5.2).

    Two vectors that cover nothing are identical (distance 0).
    """
    union = sum(1 for x, y in zip(a, b) if x or y)
    if union == 0:
        return 0.0
    return 1 - sum(1 for x, y in zip(a, b) if x and y) / union
