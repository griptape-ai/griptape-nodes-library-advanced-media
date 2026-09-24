"""Pins the edit-time / execution boundary that `griptape_nodes_library.json` declares.

The orchestrator imports every node module and builds every node class with only
`pip_dependencies` installed, so anything listed in `pip_dependencies_exec` has to stay
unimported until `process()` runs. A module-scope import of one -- directly, or through a helper
several hops away -- stops the whole library from registering, and the failure names the helper
rather than the node.
"""

import ast
import importlib
import inspect
import json
import re
import subprocess
import sys
from pathlib import Path

LIBRARY_ROOT = Path(__file__).parents[1]
MANIFEST = json.loads((LIBRARY_ROOT / "griptape_nodes_library.json").read_text())

# Distribution name -> the name it is imported under, where the two differ.
DIST_TO_IMPORT = {
    "beautifulsoup4": "bs4",
    "controlnet-aux": "controlnet_aux",
    "opencv-python": "cv2",
    "optimum-quanto": "optimum",
    "scikit-image": "skimage",
    "static-ffmpeg": "static_ffmpeg",
}

# Build tooling and packages whose import name belongs to something else entirely.
NOT_IMPORTED = {"cmake", "ninja", "protobuf"}

# Runs in a fresh interpreter, because importing the library from the test process would leave the
# node modules in `sys.modules` and every import below would then be a silent no-op.
#
# The watcher attributes each import to the innermost frame inside this repository rather than
# diffing `sys.modules`. The engine reaches several execution-set packages on its own account --
# `griptape_nodes.utils.ffmpeg_cache` imports `static_ffmpeg`, and the engine's HuggingFace repo
# parameter imports `huggingface_hub`, which imports `tqdm` -- so a diff blames this library for
# the engine's closure, while attribution ignores it and names the offending file and line.
PROBE = """
import importlib, json, sys, traceback

plan = json.loads(sys.argv[1])
root = plan["root"]
exec_imports = set(plan["exec_imports"])
sys.path.insert(0, root)
report = {"import_failures": [], "build_failures": [], "imported_from_library": []}


class Watcher:
    def find_spec(self, fullname, path=None, target=None):
        top = fullname.split(".")[0]
        if top not in exec_imports:
            return None
        # The frames above the import machinery belong to whoever ran the import statement. Only
        # the innermost one is the direct importer: this library appears further out on every
        # chain that starts with it importing an engine module.
        for frame in reversed(traceback.extract_stack()):
            if frame.filename.startswith("<"):
                continue
            if frame.filename.startswith(root) and ".venv" not in frame.filename:
                site = f"{frame.filename[len(root):].lstrip('/')}:{frame.lineno}"
                report["imported_from_library"].append(f"{top} from {site}")
            break
        return None


sys.meta_path.insert(0, Watcher())

for name in plan["node_modules"]:
    try:
        importlib.import_module(name)
    except Exception as e:
        report["import_failures"].append(f"{name}: {type(e).__name__}: {e}")

for module_name, class_name in plan["node_classes"]:
    try:
        getattr(importlib.import_module(module_name), class_name)(name=class_name)
    except Exception as e:
        report["build_failures"].append(f"{class_name}: {type(e).__name__}: {e}")

report["imported_from_library"] = sorted(set(report["imported_from_library"]))
print("PROBE_REPORT " + json.dumps(report))
"""


def exec_import_names() -> set[str]:
    """Import names of the execution set, which the orchestrator must never pull in."""
    names = set()
    for spec in MANIFEST["metadata"]["dependencies"]["pip_dependencies_exec"]:
        dist = re.split(r"[<>=!\[~; ]", spec, maxsplit=1)[0].lower()
        if dist in NOT_IMPORTED:
            continue
        names.add(DIST_TO_IMPORT.get(dist, dist.replace("-", "_")))
    return names


def node_modules() -> list[str]:
    modules = []
    for node in MANIFEST["nodes"]:
        module_name = node["file_path"].removesuffix(".py").replace("/", ".")
        if module_name not in modules:
            modules.append(module_name)
    return modules


def module_path(module_name: str) -> Path | None:
    """The file backing a module name inside this repository, or None if it is not ours."""
    for candidate in (
        LIBRARY_ROOT / (module_name.replace(".", "/") + ".py"),
        LIBRARY_ROOT / module_name.replace(".", "/") / "__init__.py",
    ):
        if candidate.is_file():
            return candidate
    return None


def is_type_checking_block(node: ast.stmt) -> bool:
    if not isinstance(node, ast.If):
        return False
    test = node.test
    if isinstance(test, ast.Name):
        return test.id == "TYPE_CHECKING"
    return isinstance(test, ast.Attribute) and test.attr == "TYPE_CHECKING"


def module_scope_imports(path: Path) -> tuple[set[str], set[str]]:
    """Third-party top-level names and local module names that importing `path` evaluates.

    A module-scope `with` or `try` block runs on import just like a bare statement, so this
    descends into them. `if TYPE_CHECKING:` never runs, so it is skipped.
    """
    package_parts = path.relative_to(LIBRARY_ROOT).parts[:-1]
    third_party: set[str] = set()
    local: set[str] = set()

    def visit(body: list[ast.stmt]) -> None:
        for node in body:
            if is_type_checking_block(node):
                continue
            if isinstance(node, (ast.If, ast.Try, ast.With, ast.AsyncWith, ast.For, ast.While)):
                visit(node.body)
                visit(getattr(node, "orelse", []))
                visit(getattr(node, "finalbody", []))
                for handler in getattr(node, "handlers", []):
                    visit(handler.body)
            elif isinstance(node, ast.Import):
                for alias in node.names:
                    if module_path(alias.name) is not None:
                        local.add(alias.name)
                    else:
                        third_party.add(alias.name.split(".")[0])
            elif isinstance(node, ast.ImportFrom):
                if node.level:
                    base = list(package_parts[: len(package_parts) - (node.level - 1)])
                    target = ".".join([*base, node.module] if node.module else base)
                elif node.module:
                    target = node.module
                else:
                    continue
                if module_path(target) is None and not node.level:
                    third_party.add(target.split(".")[0])
                    continue
                local.add(target)
                for alias in node.names:
                    local.add(f"{target}.{alias.name}")

    visit(ast.parse(path.read_text(encoding="utf-8")).body)
    return third_party, local


def import_closure(start: Path) -> dict[str, set[str]]:
    """Third-party top-level name -> the files in this repository importing it at module scope."""
    reached: dict[str, set[str]] = {}
    seen: set[Path] = set()
    pending = [start]
    while pending:
        path = pending.pop()
        if path in seen:
            continue
        seen.add(path)
        third_party, local = module_scope_imports(path)
        for name in third_party:
            reached.setdefault(name, set()).add(path.relative_to(LIBRARY_ROOT).as_posix())
        for target in local:
            # `from pkg.mod import symbol` records both; only one of them is a module.
            next_path = module_path(target) or module_path(target.rsplit(".", 1)[0])
            if next_path is not None:
                pending.append(next_path)
    return reached


def test_no_node_module_reaches_an_execution_dependency_at_module_scope() -> None:
    """The static half of the boundary, and the half that does not depend on what is installed.

    A runtime probe only sees an import that misses the module cache, so a package the engine
    already imported is invisible to it. Following the module-scope import graph sees every one.
    """
    execution_only = exec_import_names()
    offenders: dict[str, set[str]] = {}
    for module_name in node_modules():
        path = module_path(module_name)
        assert path is not None, f"{module_name} is declared in the manifest but has no file"
        for name, files in import_closure(path).items():
            if name in execution_only:
                offenders.setdefault(name, set()).update(files)

    assert not offenders, "\n".join(
        f"{name} is execution-only but imported at module scope by: {', '.join(sorted(files))}"
        for name, files in sorted(offenders.items())
    )


def test_orchestrator_builds_every_node_without_the_execution_set() -> None:
    plan = {
        "root": str(LIBRARY_ROOT),
        "exec_imports": sorted(exec_import_names()),
        "node_modules": node_modules(),
        "node_classes": [
            [node["file_path"].removesuffix(".py").replace("/", "."), node["class_name"]] for node in MANIFEST["nodes"]
        ],
    }
    completed = subprocess.run(  # noqa: S603
        [sys.executable, "-c", PROBE, json.dumps(plan)],
        capture_output=True,
        text=True,
        check=False,
    )
    assert completed.returncode == 0, completed.stderr

    lines = [line for line in completed.stdout.splitlines() if line.startswith("PROBE_REPORT ")]
    assert lines, f"probe produced no report:\n{completed.stdout}\n{completed.stderr}"
    report = json.loads(lines[-1].removeprefix("PROBE_REPORT "))

    assert report["import_failures"] == []
    assert report["build_failures"] == []
    assert report["imported_from_library"] == [], (
        "this library imported an execution-only package while the orchestrator was importing node "
        "modules or building nodes; move the import inside the function that needs it"
    )


def pipeline_type_classes() -> list[type]:
    """Every concrete pipeline-type parameter class reachable from the declared node modules."""
    for module_name in node_modules():
        importlib.import_module(module_name)
    from diffusers_nodes_library.common.parameters.diffusion.pipeline_type_parameters import (
        DiffusionPipelineTypePipelineParameters,
    )

    def descendants(cls: type) -> set[type]:
        found = set()
        for sub in cls.__subclasses__():
            found.add(sub)
            found |= descendants(sub)
        return found

    return [cls for cls in descendants(DiffusionPipelineTypePipelineParameters) if not inspect.isabstract(cls)]


def returned_attribute_name(cls: type) -> str:
    """The diffusers class name that `cls.pipeline_class` returns, read off the source."""
    tree = ast.parse(inspect.getsource(inspect.getmodule(cls)))  # type: ignore[arg-type]
    class_def = next(n for n in ast.walk(tree) if isinstance(n, ast.ClassDef) and n.name == cls.__name__)
    property_def = next(n for n in class_def.body if isinstance(n, ast.FunctionDef) and n.name == "pipeline_class")
    returned = next(n.value for n in reversed(list(ast.walk(property_def))) if isinstance(n, ast.Return) and n.value)
    if isinstance(returned, ast.Attribute):
        return returned.attr
    if isinstance(returned, ast.Name):
        return returned.id
    msg = f"{cls.__name__}.pipeline_class returns an expression this test cannot read: {ast.dump(returned)}"
    raise AssertionError(msg)


def test_every_pipeline_class_declares_its_name() -> None:
    """`PIPELINE_NAME` is the orchestrator's only route to the name, so it has to match.

    The builder node stamps a config hash built from `pipeline_name` into its output during
    `after_value_set`, which runs on the orchestrator, where `pipeline_class.__name__` cannot be
    read because there is no diffusers to read it from.
    """
    classes = pipeline_type_classes()
    assert classes, "no concrete pipeline-type parameter classes were reachable"

    for cls in classes:
        declares_name = "PIPELINE_NAME" in vars(cls)
        declares_class = "pipeline_class" in vars(cls)
        assert declares_name == declares_class, (
            f"{cls.__name__} declares only one of PIPELINE_NAME and pipeline_class; they name the same pipeline "
            f"and have to be written together"
        )
        if declares_class:
            assert cls.PIPELINE_NAME == returned_attribute_name(cls)  # type: ignore[attr-defined]
        assert getattr(cls, "PIPELINE_NAME", "")
