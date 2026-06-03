from pydantic import BaseModel


class SandboxLimits(BaseModel):
    timeout_seconds: float = 5.0
    max_output_chars: int = 8_000
