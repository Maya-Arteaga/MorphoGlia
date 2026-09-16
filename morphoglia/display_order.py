from __future__ import annotations

import re
from collections.abc import Iterable, Mapping, Sequence


_NUMBER_RE = re.compile(
    r"[-+]?(?:\d+(?:\.\d*)?|\.\d+)"
)


def normalize_metadata_field(
    field_name: object,
) -> str:
    """
    Normalize a metadata field name for semantic matching.

    Examples
    --------
    timepoint, time_point, time-point, "time point"
        -> timepoint
    """

    return re.sub(
        r"[^a-z0-9]+",
        "",
        str(field_name).casefold(),
    )


def _unique_strings(
    values: Iterable[object],
) -> list[str]:
    """
    Deduplicate display values while preserving their first spelling.
    """

    seen: set[str] = set()
    result: list[str] = []

    for raw_value in values:

        value = str(
            raw_value
        )

        identity = value.casefold()

        if identity in seen:
            continue

        seen.add(
            identity
        )

        result.append(
            value
        )

    return result


def _configured_order(
    field_name: str,
    metadata_order: Mapping[
        str,
        Sequence[object],
    ] | None,
) -> list[str]:
    """
    Recover an optional user-defined order for one metadata field.

    Field-name matching is semantic/case-insensitive, so these all match:
        timepoint
        time_point
        time-point
        Time Point
    """

    if not metadata_order:
        return []


    target = normalize_metadata_field(
        field_name
    )


    matches = [
        (
            str(
                configured_field
            ),
            configured_values,
        )
        for (
            configured_field,
            configured_values,
        )
        in metadata_order.items()
        if normalize_metadata_field(
            configured_field
        )
        == target
    ]


    if len(
        matches
    ) > 1:

        names = [
            name
            for name, _
            in matches
        ]

        raise ValueError(
            "metadata_order contains multiple keys "
            f"for the same metadata field {field_name!r}: "
            f"{names}"
        )


    if not matches:
        return []


    _, raw_values = matches[
        0
    ]


    if isinstance(
        raw_values,
        (
            str,
            bytes,
        ),
    ):

        raise TypeError(
            "metadata_order values must be sequences "
            "of category labels, not one string. "
            f"Invalid value for {field_name!r}: "
            f"{raw_values!r}"
        )


    ordered = _unique_strings(
        raw_values
    )


    if len(
        ordered
    ) != len(
        list(
            raw_values
        )
    ):

        raise ValueError(
            "metadata_order contains duplicate values "
            f"for {field_name!r}."
        )


    return ordered


def _automatic_level_key(
    value: str,
    field_name: str,
):
    """
    Automatic ordering when the user did not explicitly specify one.

    timepoint is semantic:
        DIV7, DIV14, DIV21 -> 7, 14, 21

    Other metadata remain alphabetic.
    """

    normalized_field = normalize_metadata_field(
        field_name
    )


    if normalized_field == "timepoint":

        match = _NUMBER_RE.search(
            value
        )


        if match is not None:

            return (
                0,
                float(
                    match.group(
                        0
                    )
                ),
                value.casefold(),
                value,
            )


        # Non-numeric timepoint labels remain valid but follow
        # numeric timepoints deterministically.
        return (
            1,
            float(
                "inf"
            ),
            value.casefold(),
            value,
        )


    return (
        0,
        0.0,
        value.casefold(),
        value,
    )


def ordered_levels(
    values: Iterable[object],
    field_name: str,
    metadata_order: Mapping[
        str,
        Sequence[object],
    ] | None = None,
) -> list[str]:
    """
    Canonical display order for one metadata field.

    Priority
    --------
    1. User-defined preferred order.
    2. Semantic automatic order (currently timepoint -> embedded number).
    3. Alphabetical fallback.

    A user order may be partial. Any observed values not listed by the
    user are appended using the automatic rule.
    """

    levels = _unique_strings(
        values
    )


    preferred = _configured_order(
        field_name,
        metadata_order,
    )


    preferred_rank = {
        value.casefold():
            rank
        for rank, value
        in enumerate(
            preferred
        )
    }


    return sorted(
        levels,
        key=lambda value:
            (
                (
                    0,
                    preferred_rank[
                        value.casefold()
                    ],
                    0,
                    0.0,
                    "",
                    "",
                )
                if value.casefold()
                in preferred_rank
                else (
                    1,
                    0,
                    *(
                        _automatic_level_key(
                            value,
                            field_name,
                        )
                    ),
                )
            ),
    )


def ordered_combinations(
    combinations: Iterable[
        Sequence[object]
    ],
    field_names: Sequence[str],
    metadata_order: Mapping[
        str,
        Sequence[object],
    ] | None = None,
    *,
    comparison_oriented: bool = True,
) -> list[tuple[str, ...]]:
    """
    Canonically order observed combinations of metadata values.

    For multiple fields, comparison_oriented=True preserves the existing
    MorphoGlia convention:

        category_fields = [condition, region]

    is flattened as:

        condition within region

    so comparison groups remain adjacent in stacked bars, residual
    heatmaps, profile tables, and generic >2-field grids.

    Each individual field still uses exactly the same canonical level order.
    """

    fields = tuple(
        str(
            field
        )
        for field in field_names
    )


    if not fields:
        return []


    seen: set[
        tuple[str, ...]
    ] = set()

    unique: list[
        tuple[str, ...]
    ] = []


    for raw_combination in combinations:

        combination = tuple(
            str(
                value
            )
            for value in raw_combination
        )


        if len(
            combination
        ) != len(
            fields
        ):

            raise ValueError(
                "Metadata combination width does not "
                "match field_names."
            )


        if combination in seen:
            continue


        seen.add(
            combination
        )

        unique.append(
            combination
        )


    if not unique:
        return []


    ranks: list[
        dict[str, int]
    ] = []


    for index, field_name in enumerate(
        fields
    ):

        levels = ordered_levels(
            (
                combination[
                    index
                ]
                for combination
                in unique
            ),
            field_name,
            metadata_order,
        )


        ranks.append(
            {
                level.casefold():
                    rank
                for rank, level
                in enumerate(
                    levels
                )
            }
        )


    if (
        comparison_oriented
        and len(
            fields
        )
        > 1
    ):

        precedence = (
            list(
                range(
                    1,
                    len(
                        fields
                    ),
                )
            )
            + [
                0
            ]
        )

    else:

        precedence = list(
            range(
                len(
                    fields
                )
            )
        )


    def combination_key(
        combination: tuple[
            str,
            ...
        ],
    ):

        rank_key = tuple(
            ranks[
                index
            ][
                combination[
                    index
                ].casefold()
            ]
            for index
            in precedence
        )


        text_key = tuple(
            combination[
                index
            ].casefold()
            for index
            in precedence
        )


        return (
            rank_key,
            text_key,
            combination,
        )


    return sorted(
        unique,
        key=combination_key,
    )
