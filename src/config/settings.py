from typing import Literal

from pydantic import Field, model_validator
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    # Provider selection
    llm_provider: Literal["anthropic", "gemini"] = "anthropic"

    # API keys
    anthropic_api_key: str = ""
    gemini_api_key: str = ""

    # Model tiers — leave as "" to use provider defaults (filled by validator below)
    # model_fast:      lightweight tasks — field summarization, keyword extraction,
    #                  query fixing, pairwise selection, Generator B1
    # model_powerful:  complex reasoning — schema linking, Generator B2, Generator C (ICL)
    # model_reasoning: extended thinking — Generator A
    model_fast: str = ""
    model_powerful: str = ""
    model_reasoning: str = ""

    # Paths
    bird_data_dir: str = "./data/bird"
    preprocessed_dir: str = "./data/preprocessed"
    cache_dir: str = "./data/cache"

    # Generation
    max_candidates: int = 11
    query_fix_iterations: int = 2
    icl_examples_count: int = 8

    # Schema linking
    faiss_top_k: int = 30
    lsh_top_k: int = 5

    # Selection
    fast_path_threshold: int = 1

    # Caching
    cache_llm_responses: bool = Field(default=False, alias="CACHE_LLM_RESPONSES")

    # Logging
    log_level: str = "INFO"

    @model_validator(mode="after")
    def _apply_model_defaults(self) -> "Settings":
        """Fill in provider-appropriate model defaults when tiers are left blank."""
        _DEFAULTS = {
            "anthropic": {
                "model_fast":      "claude-haiku-4-5-20251001",
                "model_powerful":  "claude-sonnet-4-6",
                "model_reasoning": "claude-sonnet-4-6",
            },
            "gemini": {
                "model_fast":      "gemini-2.5-flash",
                "model_powerful":  "gemini-2.5-pro",
                "model_reasoning": "gemini-2.5-flash",
            },
        }
        defaults = _DEFAULTS[self.llm_provider]
        for field_name, default_val in defaults.items():
            if not getattr(self, field_name):
                setattr(self, field_name, default_val)
        return self

    model_config = {
        "env_file": ".env",
        "populate_by_name": True,
    }


settings = Settings()
