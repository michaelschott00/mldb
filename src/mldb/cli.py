import click

from mldb.store import ArtifactInfo, RunInfo, RunStore


def _print_runs(runs: list[RunInfo]) -> None:
    if not runs:
        return
    id_w = max(len(r.run_id) for r in runs)
    name_w = max(len(r.run_name) for r in runs)
    ts_w = max(len(r.run_timestamp) for r in runs)
    for r in runs:
        tags_str = ", ".join(r.tags) if r.tags else ""
        click.echo(
            f"{r.run_id:<{id_w}}  {r.run_name:<{name_w}}  {r.run_timestamp:<{ts_w}}  {tags_str}"
        )


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


@click.group()
def main() -> None:
    pass


@main.command("list", context_settings={"ignore_unknown_options": True})
@click.argument("tags", nargs=-1, type=click.UNPROCESSED)
def list_runs(tags: tuple[str, ...]) -> None:
    """List runs, optionally filtered by tags."""
    store = RunStore.from_env()
    try:
        runs = store.list_runs(tags=list(tags))
    finally:
        store.close()
    _print_runs(runs)


@main.command("tag", context_settings={"ignore_unknown_options": True})
@click.argument("run_id")
@click.argument("tags", nargs=-1, type=click.UNPROCESSED)
def tag_run(run_id: str, tags: tuple[str, ...]) -> None:
    """Add (+) or remove (-) tags on a run."""
    if not tags:
        raise click.UsageError("Provide at least one tag prefixed with '+' or '-'")
    include_tags, exclude_tags = _parse_tags(tags)
    store = RunStore.from_env()
    try:
        if include_tags:
            store.add_tags(run_id, include_tags)
        if exclude_tags:
            store.remove_tags(run_id, exclude_tags)
    finally:
        store.close()


@main.command("artifacts")
@click.argument("run_id", required=False, default=None)
def list_artifacts(run_id: str | None) -> None:
    """List stored artifacts, optionally filtered to a single run."""
    store = RunStore.from_env()
    try:
        artifacts = store.list_artifacts_by_run(run_id)
    finally:
        store.close()
    _print_artifacts(artifacts, show_run_id=run_id is None)


@main.command("delete", context_settings={"ignore_unknown_options": True})
@click.argument("args", nargs=-1, type=click.UNPROCESSED, required=True)
def delete_runs(args: tuple[str, ...]) -> None:
    """Delete a run by id, or all runs matching a tag query (+include / -exclude)."""
    is_tag_query = not any(a.startswith("run_") for a in args)
    store = RunStore.from_env()
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
