from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import diffusers  # type: ignore[reportMissingImports]

import json
import logging

from griptape_nodes.exe_types.core_types import Parameter, ParameterMode
from griptape_nodes.exe_types.node_types import BaseNode
from griptape_nodes.traits.options import Options

logger = logging.getLogger("griptape_nodes")


class SchedulerParameters:
    # Takes the scheduler class NAMES rather than the classes: the choices populate a dropdown while
    # the node is being edited, so they have to be available on the orchestrator, which has no
    # diffusers. `get_scheduler_class` resolves a name against the real module in the worker.
    def __init__(self, node: BaseNode, scheduler_type_names: list[str]):
        self._node = node
        self._scheduler_type_parameter_name = "scheduler_type"
        self._scheduler_config_parameter_name = "scheduler_config"

        self._scheduler_type_names = scheduler_type_names

    def add_input_parameters(self) -> None:
        self._node.add_parameter(
            Parameter(
                name=self._scheduler_type_parameter_name,
                default_value=self._scheduler_type_names[0],
                input_types=["str"],
                type="str",
                traits={Options(choices=self._scheduler_type_names)},
                tooltip=self._scheduler_type_parameter_name,
                allowed_modes={ParameterMode.PROPERTY},
                ui_options={
                    "display_name": self._scheduler_type_parameter_name,
                    "show_search": True,
                },
            )
        )

        self._node.add_parameter(
            Parameter(
                name=self._scheduler_config_parameter_name,
                default_value=None,
                input_types=["json", "str", "dict"],
                type="json",
                tooltip=self._scheduler_config_parameter_name,
                allowed_modes={ParameterMode.INPUT, ParameterMode.PROPERTY},
                ui_options={
                    "display_name": self._scheduler_config_parameter_name,
                    "show_search": True,
                    "hide": True,
                },
            )
        )

    def remove_input_parameters(self) -> None:
        self._node.remove_parameter_element_by_name(self._scheduler_type_parameter_name)
        self._node.remove_parameter_element_by_name(self._scheduler_config_parameter_name)

    def validate_before_node_run(self) -> list[Exception] | None:
        errors = []
        scheduler_type_name = self._node.get_parameter_value(self._scheduler_type_parameter_name)
        if scheduler_type_name not in self._scheduler_type_names:
            errors.append(
                ValueError(
                    f"Attempted to use scheduler {scheduler_type_name!r}. "
                    f"Failed because this pipeline only supports: {', '.join(self._scheduler_type_names)}."
                )
            )
        try:
            self.get_scheduler_config()
        except (ValueError, TypeError) as e:
            errors.append(e)
        return errors or None

    def validate_in_execution_environment(self) -> list[Exception] | None:
        try:
            self.get_scheduler()
        except Exception as e:
            return [e]
        return None

    def get_config_kwargs(self) -> dict:
        return {
            self._scheduler_type_parameter_name: self._node.get_parameter_value(self._scheduler_type_parameter_name),
            self._scheduler_config_parameter_name: self._node.get_parameter_value(
                self._scheduler_config_parameter_name
            ),
        }

    def get_scheduler_class(self) -> type[diffusers.SchedulerMixin]:
        import diffusers  # type: ignore[reportMissingImports]

        scheduler_type_name = self._node.get_parameter_value(self._scheduler_type_parameter_name)
        if scheduler_type_name not in self._scheduler_type_names:
            msg = (
                f"Attempted to use scheduler {scheduler_type_name!r}. "
                f"Failed because this pipeline only supports: {', '.join(self._scheduler_type_names)}."
            )
            raise ValueError(msg)
        return getattr(diffusers, scheduler_type_name)

    def get_scheduler_config(self) -> dict:
        scheduler_config = self._node.get_parameter_value(self._scheduler_config_parameter_name)
        if scheduler_config is None:
            return {}
        if isinstance(scheduler_config, dict):
            return scheduler_config
        if isinstance(scheduler_config, str):
            try:
                return json.loads(scheduler_config)
            except json.JSONDecodeError as e:
                msg = f"Invalid JSON string provided. Failed to parse JSON: {e}. Input was: {scheduler_config[:200]!r}"
                raise ValueError(msg) from e
        else:
            msg = f"Invalid {self._scheduler_config_parameter_name} provided. Must be json, str, or dict"
            raise TypeError(msg)

    def get_scheduler(self) -> diffusers.SchedulerMixin:
        scheduler_class = self.get_scheduler_class()
        scheduler_config = self.get_scheduler_config()
        return scheduler_class.from_config(scheduler_config)
