from pydantic import BaseModel, Field


class CapabilityDescriptor(BaseModel):
    agent_name: str
    capabilities: list[str] = Field(default_factory=list)
