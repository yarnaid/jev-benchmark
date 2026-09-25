"""Long-lived services shared by the web app and the CLI.

Classes:
    Services: stores, OpenRouter client, catalog, job registry and config loaders from Settings.
        `api_key` prefers a key supplied by the browser over the server's OPENROUTER_API_KEY;
        `sweep_interrupted` marks runs, generations and analyses left `running` as interrupted.
"""

import httpx2

from jev_bench.analysis.config import AnalysisConfig, load_analysis_config
from jev_bench.analysis.job import mark_interrupted_analyses
from jev_bench.benchmark_config import BenchmarkConfig, load_benchmark_config
from jev_bench.catalog import Catalog
from jev_bench.generation.config import GenerationConfig, load_generation_config
from jev_bench.generation.generator import mark_interrupted_generations
from jev_bench.jobs import JobRegistry
from jev_bench.openrouter import OpenRouterClient
from jev_bench.questions import QuestionSet, load_question_set
from jev_bench.runner import mark_interrupted_runs
from jev_bench.settings import Settings
from jev_bench.store.analyses import AnalysisStore
from jev_bench.store.embeddings import EmbeddingCaches
from jev_bench.store.generations import GenerationStore
from jev_bench.store.labels import LabelStore
from jev_bench.store.runs import RunStore

__all__ = [
    "Services",
]


class Services:
    def __init__(self, settings: Settings, http: httpx2.AsyncClient) -> None:
        self.settings = settings
        self.client = OpenRouterClient(
            http, max_retries=settings.max_retries, retry_base_delay_s=settings.retry_base_delay_s
        )
        self.catalog = Catalog(self.client)
        self.jobs = JobRegistry()
        self.generations = GenerationStore(settings.data_dir / "generations")
        self.runs = RunStore(settings.data_dir / "runs")
        self.labels = LabelStore(settings.data_dir / "labels")
        self.analyses = AnalysisStore(settings.data_dir / "analyses")
        self.embedding_caches = EmbeddingCaches(settings.data_dir / "embeddings")

    def question_set(self) -> QuestionSet:
        return load_question_set(self.settings.config_dir / "questions.toml")

    def benchmark_config(self) -> BenchmarkConfig:
        return load_benchmark_config(self.settings.config_dir / "benchmark.toml")

    def generation_config(self) -> GenerationConfig:
        return load_generation_config(self.settings.config_dir / "generation.toml")

    def analysis_config(self) -> AnalysisConfig:
        return load_analysis_config(self.settings.config_dir / "analysis.toml")

    def api_key(self, supplied: str | None) -> str | None:
        cleaned = (supplied or "").strip()
        return cleaned or self.settings.server_api_key() or None

    def sweep_interrupted(self) -> list[str]:
        runs = mark_interrupted_runs(self.runs, self.jobs.is_running)
        generations = mark_interrupted_generations(self.generations, self.jobs.is_running)
        analyses = mark_interrupted_analyses(self.analyses, self.jobs.is_running)
        return [*runs, *generations, *analyses]
