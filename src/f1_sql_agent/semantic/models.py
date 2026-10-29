"""Data structures of the semantic layer.

The structural part (tables, columns, keys, relationships) comes from the dbt
manifest. Business knowledge (metrics, rules, examples) is added in later steps.
Models are frozen: once loaded, the layer cannot be modified by accident.
"""

from pydantic import BaseModel, ConfigDict


class _Frozen(BaseModel):
    """Base model: immutable and strict about unknown fields."""

    model_config = ConfigDict(frozen=True, extra="forbid")


class ColumnRef(_Frozen):
    """A column in another table: the target of a relationship."""

    table: str
    column: str


class Column(_Frozen):
    """A column as the agent sees it."""

    name: str
    data_type: str
    description: str = ""
    nullable: bool = True
    # Valid values, from dbt accepted_values tests (empty = not an enum).
    accepted_values: tuple[str, ...] = ()
    # Foreign-key-like target, from dbt relationships tests.
    references: ColumnRef | None = None


class Table(_Frozen):
    """A table of the star schema with its grain (keys) and columns."""

    name: str
    columns: tuple[Column, ...]
    description: str = ""
    primary_key: tuple[str, ...] = ()
    unique_keys: tuple[tuple[str, ...], ...] = ()

    def column(self, name: str) -> Column:
        """Return a column by name (KeyError if it does not exist)."""
        for col in self.columns:
            if col.name == name:
                return col
        raise KeyError(f"{self.name} has no column {name!r}")


class Catalog(_Frozen):
    """Structural metadata of every table the agent can query."""

    tables: dict[str, Table]

    def relationships(self) -> list[tuple[str, str, ColumnRef]]:
        """Return (table, column, target) for each column with a reference."""
        return [
            (table.name, col.name, col.references)
            for table in self.tables.values()
            for col in table.columns
            if col.references is not None
        ]
