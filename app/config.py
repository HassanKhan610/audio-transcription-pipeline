import os
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Settings:
    model_size: str = os.getenv("WHISPER_MODEL", "small")
    device: str = os.getenv("WHISPER_DEVICE", "cpu")
    compute_type: str = os.getenv("WHISPER_COMPUTE_TYPE", "int8")
    data_dir: Path = Path(os.getenv("DATA_DIR", "data"))
    max_upload_mb: int = int(os.getenv("MAX_UPLOAD_MB", "200"))
    max_duration_s: float = float(os.getenv("MAX_DURATION_S", str(4 * 60 * 60)))
    # Long-audio chunking
    chunk_target_s: float = float(os.getenv("CHUNK_TARGET_S", "30"))
    chunk_search_window_s: float = float(os.getenv("CHUNK_SEARCH_WINDOW_S", "5"))
    chunk_overlap_s: float = float(os.getenv("CHUNK_OVERLAP_S", "1.0"))
    # Files at or under this length are transcribed in one pass, no chunking
    single_pass_max_s: float = float(os.getenv("SINGLE_PASS_MAX_S", "45"))


settings = Settings()
