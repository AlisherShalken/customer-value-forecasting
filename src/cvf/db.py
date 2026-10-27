"""DuckDB warehouse access and SQL template rendering."""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Any

import duckdb
from jinja2 import Environment, FileSystemLoader, StrictUndefined

from cvf.config import PROJECT_ROOT

SQL_DIR = PROJECT_ROOT / "sql"

_jinja = Environment(
    loader=FileSystemLoader(SQL_DIR),
    undefined=StrictUndefined,
    keep_trailing_newline=True,
)


def render_sql(name: str, **params: Any) -> str:
    """Render a SQL template from the sql/ directory.

    Templates only receive identifiers and integers from code; user-supplied
    values (customer ids, dates) are always bound as query parameters.
    """
    return _jinja.get_template(name).render(**params)


@contextmanager
def connect(path: Path | str, read_only: bool = False) -> Iterator[duckdb.DuckDBPyConnection]:
    if str(path) != ":memory:":
        Path(path).parent.mkdir(parents=True, exist_ok=True)
    con = duckdb.connect(str(path), read_only=read_only)
    try:
        yield con
    finally:
        con.close()


def run_script(con: duckdb.DuckDBPyConnection, name: str, **params: Any) -> None:
    con.execute(render_sql(name, **params))
