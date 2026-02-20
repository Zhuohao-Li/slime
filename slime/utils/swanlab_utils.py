import logging
import os
import uuid
from copy import deepcopy
from typing import Any

logger = logging.getLogger(__name__)


def _import_swanlab():
    try:
        import swanlab

        return swanlab
    except ImportError as e:
        raise RuntimeError(
            "SwanLab logging is enabled but `swanlab` is not installed. Please install it with `pip install swanlab`."
        ) from e


def _compute_config_for_logging(args):
    output = deepcopy(args.__dict__)

    whitelist_env_vars = [
        "SLURM_JOB_ID",
    ]
    output["env_vars"] = {k: v for k, v in os.environ.items() if k in whitelist_env_vars}

    return output


def _resolve_project(args) -> str:
    if args.swanlab_project:
        return args.swanlab_project
    if args.wandb_project:
        return args.wandb_project
    return "slime"


def _resolve_experiment_name(args) -> str:
    # Keep generated experiment name stable across distributed workers by caching on args.
    cached_name = getattr(args, "swanlab_experiment_name_resolved", None)
    if cached_name:
        return cached_name

    base_name = args.swanlab_experiment_name or args.wandb_group or "slime-run"
    if args.swanlab_random_suffix and (not args.swanlab_run_id):
        base_name = f"{base_name}_{uuid.uuid4().hex[:6]}"

    args.swanlab_experiment_name_resolved = base_name
    return base_name


def _extract_run_id(swanlab) -> str | None:
    for attr in ("run", "experiment", "session"):
        obj = getattr(swanlab, attr, None)
        if obj is not None and hasattr(obj, "id"):
            run_id = getattr(obj, "id")
            if run_id:
                return str(run_id)
    return None


def _maybe_login(swanlab, args):
    if args.swanlab_key:
        login_kwargs: dict[str, Any] = {}
        if args.swanlab_host:
            login_kwargs["host"] = args.swanlab_host
        swanlab.login(args.swanlab_key, **login_kwargs)


def _build_init_kwargs(args, config):
    init_kwargs = {
        "project": _resolve_project(args),
        "experiment_name": _resolve_experiment_name(args),
        "config": config,
    }

    if args.swanlab_mode:
        init_kwargs["mode"] = args.swanlab_mode
    if args.swanlab_dir:
        os.makedirs(args.swanlab_dir, exist_ok=True)
        init_kwargs["logdir"] = args.swanlab_dir
    if args.swanlab_description:
        init_kwargs["description"] = args.swanlab_description
    if args.swanlab_run_id:
        init_kwargs["id"] = args.swanlab_run_id
    if args.swanlab_resume:
        init_kwargs["resume"] = args.swanlab_resume

    return init_kwargs


def init_swanlab_primary(args):
    if not args.use_swanlab:
        args.swanlab_run_id = None
        return

    swanlab = _import_swanlab()
    _maybe_login(swanlab, args)

    init_kwargs = _build_init_kwargs(
        args=args,
        config={"FRAMEWORK": "slime", **_compute_config_for_logging(args)},
    )
    swanlab.init(**init_kwargs)

    if args.swanlab_run_id is None:
        args.swanlab_run_id = _extract_run_id(swanlab)
        if args.swanlab_run_id is None:
            logger.warning("Unable to infer SwanLab run id; distributed workers may create separate runs.")


def init_swanlab_secondary(args, router_addr=None):
    if not args.use_swanlab:
        return

    del router_addr  # not used for SwanLab yet

    swanlab = _import_swanlab()
    _maybe_login(swanlab, args)

    init_kwargs = _build_init_kwargs(
        args=args,
        config={"FRAMEWORK": "slime", **args.__dict__},
    )

    # Default to allow-style resume semantics for secondary workers when a run id is provided.
    if ("id" in init_kwargs) and ("resume" not in init_kwargs):
        init_kwargs["resume"] = "allow"

    swanlab.init(**init_kwargs)


def log(metrics, step=None):
    swanlab = _import_swanlab()
    if step is None:
        swanlab.log(metrics)
    else:
        swanlab.log(metrics, step=step)
