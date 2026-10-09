"""Validated controls supported by the installed Chatterbox Turbo model."""
from pydantic import BaseModel, Field

class ChatterboxSettings(BaseModel):
    temperature: float = Field(default=0.8, ge=0.1, le=1.5)
    top_p: float = Field(default=0.95, ge=0.1, le=1.0)
    top_k: int = Field(default=1000, ge=1, le=2000)
    repetition_penalty: float = Field(default=1.2, ge=1.0, le=2.0)
    max_chars: int = Field(default=280, ge=100, le=500)
    seed: int | None = Field(default=None, ge=0, le=2147483647)

CONFIG_PRESETS = {
    "balanced": {"label": "Balanced", "settings": ChatterboxSettings().model_dump()},
    "steady": {"label": "Lower variation", "settings": ChatterboxSettings(temperature=0.6, top_p=0.9).model_dump()},
    "expressive": {"label": "Higher variation", "settings": ChatterboxSettings(temperature=1.0, top_p=0.98).model_dump()},
    "longer_chunks": {"label": "Longer phrases", "settings": ChatterboxSettings(max_chars=420).model_dump()},
}
