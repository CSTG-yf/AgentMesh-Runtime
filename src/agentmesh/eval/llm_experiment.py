from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Literal

import orjson
import yaml
from pydantic import BaseModel, ConfigDict, Field


class ProtocolOptimizationProfile(BaseModel):
    model_config = ConfigDict(frozen=True)

    planner_max_chars: int = Field(default=1200, gt=0)
    retriever_max_chars: int = Field(default=1600, gt=0)
    executor_max_chars: int = Field(default=2400, gt=0)
    summarizer_max_chars: int = Field(default=2400, gt=0)
    evidence_max_items: int = Field(default=3, gt=0)
    evidence_min_score: float = Field(default=0.0, ge=0.0, le=1.0)
    deterministic_retrieval_evidence: bool = False


class LLMExperimentProfile(BaseModel):
    model_config = ConfigDict(frozen=True)

    profile_id: str
    artifact_label: str = Field(pattern=r"^[A-Za-z0-9][A-Za-z0-9_.-]*$")
    repeat: int = Field(default=3, gt=0)
    prompt_version: str = "p2"
    route_policy_version: Literal["legacy", "quality_safe_v1"] = "legacy"
    optimization_enabled: bool = False
    protocol: ProtocolOptimizationProfile = Field(
        default_factory=ProtocolOptimizationProfile
    )

    @property
    def sha256(self) -> str:
        payload = orjson.dumps(self.model_dump(mode="json"), option=orjson.OPT_SORT_KEYS)
        return hashlib.sha256(payload).hexdigest()


def load_llm_experiment_profile(path: Path) -> LLMExperimentProfile:
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError("LLM experiment profile must be a YAML mapping")
    return LLMExperimentProfile.model_validate(raw)
