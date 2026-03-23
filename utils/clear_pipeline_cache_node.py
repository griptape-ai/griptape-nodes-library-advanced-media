import logging

from griptape_nodes.exe_types.node_types import AsyncResult, SuccessFailureNode

from diffusers_nodes_library.common.utils.huggingface_utils import model_cache

logger = logging.getLogger("griptape_nodes")


class ClearPipelineCacheNode(SuccessFailureNode):
    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        self._create_status_parameters(
            result_details_tooltip="Details about the cache clearing operation",
            result_details_placeholder="Cache clearing details will appear here when the node executes.",
        )

    def process(self) -> AsyncResult | None:
        yield lambda: self._process()

    def _process(self) -> None:
        self._clear_execution_status()
        try:
            stats = model_cache.get_cache_stats()
            pipeline_count = stats["cached_pipelines"]
            model_cache.clear_pipeline_cache()
            self._set_status_results(
                was_successful=True,
                result_details=f"Cleared {pipeline_count} pipeline(s) from cache.",
            )
        except Exception as e:
            self._set_status_results(
                was_successful=False,
                result_details=str(e),
            )
            self._handle_failure_exception(e)
