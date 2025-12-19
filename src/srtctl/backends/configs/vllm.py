# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
vLLM backend configuration (placeholder).

Implements BackendProtocol for vLLM inference serving.
"""

from dataclasses import field
from pathlib import Path
from typing import (
    TYPE_CHECKING,
    Any,
    ClassVar,
    Dict,
    List,
    Literal,
    Optional,
    Sequence,
    Type,
)

from marshmallow import Schema
from marshmallow_dataclass import dataclass

if TYPE_CHECKING:
    from srtctl.core.endpoints import Endpoint, Process
    from srtctl.core.runtime import RuntimeContext


@dataclass(frozen=True)
class VLLMBackendConfig:
    """vLLM backend configuration - implements BackendProtocol.

    Placeholder for future vLLM support.

    Example YAML:
        backend:
          type: vllm
          tensor_parallel_size: 4
          pipeline_parallel_size: 1
    """

    type: Literal["vllm"] = "vllm"

    # vLLM-specific settings
    tensor_parallel_size: int = 1
    pipeline_parallel_size: int = 1
    max_model_len: Optional[int] = None
    gpu_memory_utilization: float = 0.9

    # Environment variables
    environment: Dict[str, str] = field(default_factory=dict)

    Schema: ClassVar[Type[Schema]] = Schema

    # =========================================================================
    # BackendProtocol Implementation (stubs)
    # =========================================================================

    def get_config_for_mode(self, mode: str) -> Dict[str, Any]:
        """vLLM doesn't have prefill/decode modes."""
        return {
            "tensor_parallel_size": self.tensor_parallel_size,
            "pipeline_parallel_size": self.pipeline_parallel_size,
            "max_model_len": self.max_model_len,
            "gpu_memory_utilization": self.gpu_memory_utilization,
        }

    def get_environment_for_mode(self, mode: str) -> Dict[str, str]:
        """Get environment variables."""
        return dict(self.environment)

    def allocate_endpoints(
        self,
        num_prefill: int,
        num_decode: int,
        num_agg: int,
        gpus_per_prefill: int,
        gpus_per_decode: int,
        gpus_per_agg: int,
        gpus_per_node: int,
        available_nodes: Sequence[str],
    ) -> List["Endpoint"]:
        """Allocate endpoints for vLLM (aggregated mode only)."""
        from srtctl.core.endpoints import allocate_endpoints

        # vLLM uses aggregated mode
        return allocate_endpoints(
            num_prefill=0,
            num_decode=0,
            num_agg=num_agg or 1,
            gpus_per_prefill=0,
            gpus_per_decode=0,
            gpus_per_agg=gpus_per_agg or (self.tensor_parallel_size * self.pipeline_parallel_size),
            gpus_per_node=gpus_per_node,
            available_nodes=available_nodes,
        )

    def endpoints_to_processes(
        self,
        endpoints: List["Endpoint"],
        base_port: int = 8081,
    ) -> List["Process"]:
        """Convert endpoints to processes."""
        from srtctl.core.endpoints import endpoints_to_processes

        return endpoints_to_processes(endpoints, base_port)

    def build_worker_command(
        self,
        process: "Process",
        endpoint_processes: List["Process"],
        runtime: "RuntimeContext",
        use_sglang_router: bool = False,
        dump_config_path: Optional[Path] = None,
    ) -> List[str]:
        """Build vLLM server command."""
        cmd = [
            "python3",
            "-m",
            "vllm.entrypoints.openai.api_server",
            "--model",
            str(runtime.model_path),
            "--host",
            "0.0.0.0",
            "--port",
            str(process.sys_port),
            "--tensor-parallel-size",
            str(self.tensor_parallel_size),
            "--pipeline-parallel-size",
            str(self.pipeline_parallel_size),
            "--gpu-memory-utilization",
            str(self.gpu_memory_utilization),
        ]

        if self.max_model_len:
            cmd.extend(["--max-model-len", str(self.max_model_len)])

        return cmd

