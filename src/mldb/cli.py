import click

from mldb.store import ArtifactInfo, RunInfo, RunStore


def _format_hparams(
    r: RunInfo, hparams_filter: list[str] | None, values_only: bool = False
) -> str:
    if not r.hparams:
        return ""
    hparams = r.hparams
    if hparams_filter is not None:
        hparams = [h for h in hparams if h.split("=", 1)[0] in hparams_filter]
    if values_only:
        return ", ".join(h.split("=", 1)[1] for h in hparams)
    return ", ".join(hparams)


_RUN_COLUMNS = {
    "rid": ("run_id", lambda r, hp, vo: r.run_id),
    "rn": ("run_name", lambda r, hp, vo: r.run_name),
    "rts": ("run_timestamp", lambda r, hp, vo: r.run_timestamp),
    "t": ("tags", lambda r, hp, vo: ", ".join(r.tags) if r.tags else ""),
    "hp": ("hparams", lambda r, hp, vo: _format_hparams(r, hp, vo)),
}

_DEFAULT_RUN_FORMAT = "rid,rn,rts,t,hp"


def _parse_run_format(fmt: str) -> list[str]:
    cols = [c.strip() for c in fmt.split(",")]
    for c in cols:
        if c not in _RUN_COLUMNS:
            valid = ", ".join(_RUN_COLUMNS)
            raise click.UsageError(
                f"Unknown format column '{c}'. Valid columns: {valid}"
            )
    return cols


def _print_runs(
    runs: list[RunInfo],
    fmt: str = _DEFAULT_RUN_FORMAT,
    hparams_filter: list[str] | None = None,
    hparams_values_only: bool = False,
) -> None:
    if not runs:
        return
    cols = _parse_run_format(fmt)
    values = [
        [_RUN_COLUMNS[c][1](r, hparams_filter, hparams_values_only) for c in cols]
        for r in runs
    ]
    widths = [max(len(row[i]) for row in values) for i in range(len(cols))]
    for row in values:
        click.echo("  ".join(v.ljust(w) for v, w in zip(row, widths)))


def _print_artifacts(artifacts: list[ArtifactInfo], show_run_id: bool) -> None:
    if not artifacts:
        return
    if show_run_id:
        id_w = max(len(a.run_id) for a in artifacts)
        for a in artifacts:
            click.echo(f"{a.run_id:<{id_w}}  {a.artifact_name}")
    else:
        for a in artifacts:
            click.echo(a.artifact_name)


def _get_store(data: str | None) -> RunStore:
    if data is None:
        return RunStore.from_env()
    return RunStore(root_dir=data)


def _parse_tags(tags: tuple[str, ...]) -> tuple[list[str], list[str]]:
    include, exclude = [], []
    for t in tags:
        if t.startswith("+"):
            include.append(t[1:])
        elif t.startswith("-"):
            exclude.append(t[1:])
        else:
            raise click.UsageError(f"Tag '{t}' must be prefixed with '+' or '-'")
    return include, exclude


def _split_tags_and_hparams(args: tuple[str, ...]) -> tuple[list[str], list[str]]:
    tags, hparams = [], []
    for a in args:
        stripped = a[1:] if a and a[0] in "+-" else a
        if "=" in stripped:
            hparams.append(a)
        else:
            tags.append(a)
    return tags, hparams


@click.group()
def main() -> None:
    pass


def _parse_hparams(args: tuple[str, ...]) -> dict[str, list[str]]:
    hparams: dict[str, list[str]] = {}
    for a in args:
        if "=" not in a:
            raise click.UsageError(
                f"Hyperparameter '{a}' must be in the form name=value"
            )
        name, value = a.split("=", 1)
        hparams.setdefault(name, []).append(value)
    return hparams


@main.command("list", context_settings={"ignore_unknown_options": True})
@click.argument("args", nargs=-1, type=click.UNPROCESSED)
@click.option(
    "--data",
    "-d",
    default=None,
    help="Root directory to use instead of DATA_ROOT env var.",
)
@click.option(
    "--columns",
    "-o",
    "fmt",
    default=_DEFAULT_RUN_FORMAT,
    help=f"Comma-separated columns to display: {', '.join(_RUN_COLUMNS)}.",
)
@click.option(
    "--hparams",
    "-p",
    "hparams_display",
    default=None,
    help=(
        "Comma-separated hyperparameter names to display in the hparams column "
        "(default: show all)."
    ),
)
@click.option(
    "--values-only",
    "-b",
    "values_only",
    is_flag=True,
    default=False,
    help="Print hyperparameters as values only, without the name= prefix.",
)
def list_runs(
    args: tuple[str, ...],
    data: str | None,
    fmt: str,
    hparams_display: str | None,
    values_only: bool,
) -> None:
    """List runs, optionally filtered by tags and hyperparameters (name=value)."""
    tags, hparams = _split_tags_and_hparams(args)
    store = _get_store(data)
    try:
        runs = store.list_runs(tags=tags, hparams=_parse_hparams(tuple(hparams)))
    finally:
        store.close()
    hparams_filter = (
        [h.strip() for h in hparams_display.split(",")]
        if hparams_display is not None
        else None
    )
    _print_runs(runs, fmt, hparams_filter, values_only)


@main.command("tag", context_settings={"ignore_unknown_options": True})
@click.argument("run_id")
@click.argument("tags", nargs=-1, type=click.UNPROCESSED)
@click.option(
    "--data", default=None, help="Root directory to use instead of DATA_ROOT env var."
)
def tag_run(run_id: str, tags: tuple[str, ...], data: str | None) -> None:
    """Add (+) or remove (-) tags on a run."""
    if not tags:
        raise click.UsageError("Provide at least one tag prefixed with '+' or '-'")
    include_tags, exclude_tags = _parse_tags(tags)
    store = _get_store(data)
    try:
        if include_tags:
            store.add_tags(run_id, include_tags)
        if exclude_tags:
            store.remove_tags(run_id, exclude_tags)
    finally:
        store.close()


@main.command("artifacts")
@click.argument("run_id", required=False, default=None)
@click.option(
    "--data", default=None, help="Root directory to use instead of DATA_ROOT env var."
)
def list_artifacts(run_id: str | None, data: str | None) -> None:
    """List stored artifacts, optionally filtered to a single run."""
    store = _get_store(data)
    try:
        artifacts = store.list_artifacts_by_run(run_id)
    finally:
        store.close()
    _print_artifacts(artifacts, show_run_id=run_id is None)


@main.command("hparams")
@click.argument("run_id")
@click.option(
    "--data", default=None, help="Root directory to use instead of DATA_ROOT env var."
)
def show_hparams(run_id: str, data: str | None) -> None:
    """Show hyperparameters stored for a run."""
    store = _get_store(data)
    try:
        rows = store.get_hparams([run_id])
    finally:
        store.close()
    if not rows:
        return
    name_w = max(len(r["name"]) for r in rows)
    for r in rows:
        click.echo(f"{r['name']:<{name_w}}  {r['value']}")


@main.command("merge")
@click.argument("source_dir")
@click.option(
    "--data", default=None, help="Root directory to use instead of DATA_ROOT env var."
)
def merge_store(source_dir: str, data: str | None) -> None:
    """Merge runs, artifacts, and blobs from another store's directory into this one."""
    store = _get_store(data)
    try:
        store.merge(source_dir)
    finally:
        store.close()


@main.command("delete", context_settings={"ignore_unknown_options": True})
@click.argument("args", nargs=-1, type=click.UNPROCESSED, required=True)
@click.option(
    "--data", default=None, help="Root directory to use instead of DATA_ROOT env var."
)
def delete_runs(args: tuple[str, ...], data: str | None) -> None:
    """Delete a run by id, or all runs matching a tag query (+include / -exclude)."""
    is_tag_query = not any(a.startswith("run_") for a in args)
    store = _get_store(data)
    try:
        if is_tag_query:
            runs = store.list_runs(tags=list(args))
            for run in runs:
                store.delete_run(run.run_id)
                click.echo(f"Deleted {run.run_id}")
        else:
            if len(args) != 1:
                raise click.UsageError(
                    "Provide exactly one run_id, or use tags to delete by query"
                )
            store.delete_run(args[0])
            click.echo(f"Deleted {args[0]}")
    finally:
        store.close()
