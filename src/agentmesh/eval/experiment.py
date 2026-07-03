from __future__ import annotations

import hashlib
import platform
import sys
from enum import StrEnum
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict

BENCHMARK_SCHEMA_VERSION = "2.0"
_SECRET_KEYS = {"api_key", "token", "secret", "password", "authorization"}


class BenchmarkTrack(StrEnum):
    DETERMINISTIC = "deterministic"
    LLM = "llm"


class ExperimentManifest(BaseModel):
    model_config = ConfigDict(frozen=True)

    schema_version: str = BENCHMARK_SCHEMA_VERSION
    experiment_id: str
    suite_name: str
    suite_sha256: str
    track: BenchmarkTrack
    repeat_count: int
    seed: int
    python_version: str
    platform: str
    environment: dict[str, str]
    profile_id: str = ""
    profile_sha256: str = ""
    prompt_version: str = ""
    route_policy_version: str = ""
    model_fingerprint: str = ""
    prompt_tree_sha256: str = ""
    quality_rules_sha256: str = ""


def build_experiment_manifest(
    *,
    suite_path: Path,
    suite_name: str,
    track: BenchmarkTrack,
    repeat_count: int,
    seed: int,
    environment: dict[str, Any] | None = None,
    profile_id: str = "",
    profile_sha256: str = "",
    prompt_version: str = "",
    route_policy_version: str = "",
    model_fingerprint: str = "",
    prompt_tree_sha256: str = "",
    quality_rules_sha256: str = "",
) -> ExperimentManifest:
    suite_sha256 = hashlib.sha256(suite_path.read_bytes()).hexdigest()
    safe_environment = {
        str(key): str(value)
        for key, value in sorted((environment or {}).items())
        if not any(secret_key in str(key).lower() for secret_key in _SECRET_KEYS)
        and value is not None
    }
    identity = "|".join(
        [
            BENCHMARK_SCHEMA_VERSION,
            suite_name,
            suite_sha256,
            track.value,
            str(repeat_count),
            str(seed),
            profile_id,
            profile_sha256,
            prompt_version,
            route_policy_version,
            model_fingerprint,
            prompt_tree_sha256,
            quality_rules_sha256,
        ]
    )
    experiment_id = f"exp-{hashlib.sha256(identity.encode()).hexdigest()[:16]}"
    return ExperimentManifest(
        experiment_id=experiment_id,
        suite_name=suite_name,
        suite_sha256=suite_sha256,
        track=track,
        repeat_count=repeat_count,
        seed=seed,
        python_version=sys.version.split()[0],
        platform=platform.platform(),
        environment=safe_environment,
        profile_id=profile_id,
        profile_sha256=profile_sha256,
        prompt_version=prompt_version,
        route_policy_version=route_policy_version,
        model_fingerprint=model_fingerprint,
        prompt_tree_sha256=prompt_tree_sha256,
        quality_rules_sha256=quality_rules_sha256,
    )


def model_fingerprint(base_url: str, model: str) -> str:
    return hashlib.sha256(
        f"{base_url.rstrip('/')}|{model}".encode()
    ).hexdigest()[:16]


def prompt_tree_sha256(prompt_dir: Path) -> str:
    digest = hashlib.sha256()
    if prompt_dir.exists():
        for path in sorted(item for item in prompt_dir.rglob("*") if item.is_file()):
            digest.update(path.relative_to(prompt_dir).as_posix().encode())
            digest.update(b"\0")
            digest.update(path.read_bytes())
            digest.update(b"\0")
    return digest.hexdigest()


def paired_mode_order(*, seed: int, pair_index: int) -> tuple[str, str]:
    return (
        ("text", "protocol")
        if (seed + pair_index) % 2 == 0
        else ("protocol", "text")
    )
