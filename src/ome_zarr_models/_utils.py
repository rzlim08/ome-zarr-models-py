"""
Private utilities.
"""

from __future__ import annotations

from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import MISSING, fields, is_dataclass
from typing import TYPE_CHECKING, Any, TypeVar

import pydantic
import pydantic_zarr.v2
import pydantic_zarr.v3
from pydantic import create_model

from ome_zarr_models.base import BaseAttrsv2, BaseAttrsv3
from ome_zarr_models.common.validation import (
    check_array_path,
    check_group_path,
)

if TYPE_CHECKING:
    from collections.abc import Iterable

    import zarr
    from zarr.abc.store import Store

    from ome_zarr_models._v06.base import BaseGroupv06
    from ome_zarr_models.v04.base import BaseGroupv04
    from ome_zarr_models.v05.base import BaseGroupv05

TBaseGroupv2 = TypeVar("TBaseGroupv2", bound="BaseGroupv04[Any]")
TAttrsv2 = TypeVar("TAttrsv2", bound=BaseAttrsv2)


def _from_zarr_v2(
    group: zarr.Group,
    group_cls: type[TBaseGroupv2],
    attrs_cls: type[TAttrsv2],
) -> TBaseGroupv2:
    """
    Create a GroupSpec from a potentially unlistable Zarr group.

    This uses methods on the attribute class to get required and optional
    paths to ararys and groups, and then manually constructs the GroupSpec
    from those paths.

    Parameters
    ----------
    group :
        Zarr group to create GroupSpec from.
    group_cls :
        Class of the Group to return.
    attrs_cls :
        Attributes class.
    """
    # on unlistable storage backends, the members of this group will be {}
    group_spec_in: pydantic_zarr.v2.AnyGroupSpec
    group_spec_in = pydantic_zarr.v2.GroupSpec.from_zarr(group, depth=0)
    attributes = attrs_cls.model_validate(group_spec_in.attributes)

    members_tree_flat: dict[
        str, pydantic_zarr.v2.AnyGroupSpec | pydantic_zarr.v2.AnyArraySpec
    ] = {}

    def _resolve_required_array(array_path: str) -> tuple[str, Any]:
        return ("/" + array_path, check_array_path(group, array_path, expected_zarr_version=2))

    def _resolve_optional_array(array_path: str) -> tuple[str, Any] | None:
        try:
            return ("/" + array_path, check_array_path(group, array_path, expected_zarr_version=2))
        except ValueError:
            return None

    def _resolve_required_group(group_path: str, group_type: type) -> list[tuple[str, Any]]:
        check_group_path(group, group_path, expected_zarr_version=2)
        group_flat = group_type.from_zarr(group[group_path]).to_flat()
        return [("/" + group_path + path, group_flat[path]) for path in group_flat]

    def _resolve_optional_group(group_path: str, group_type: type) -> list[tuple[str, Any]]:
        try:
            check_group_path(group, group_path, expected_zarr_version=2)
        except FileNotFoundError:
            return []
        group_flat = group_type.from_zarr(group[group_path]).to_flat()
        return [("/" + group_path + path, group_flat[path]) for path in group_flat]

    _submit_and_collect(
        members_tree_flat,
        attrs_cls.get_array_paths(attributes),
        attrs_cls.get_optional_array_paths(attributes),
        attrs_cls.get_group_paths(attributes),
        attrs_cls.get_optional_group_paths(attributes),
        _resolve_required_array,
        _resolve_optional_array,
        _resolve_required_group,
        _resolve_optional_group,
    )

    members_normalized: pydantic_zarr.v2.AnyGroupSpec = (
        pydantic_zarr.v2.GroupSpec.from_flat(members_tree_flat)
    )
    return group_cls(members=members_normalized.members, attributes=attributes)


_MAX_WORKERS: int = 8


def _submit_and_collect(
    members_tree_flat: dict[str, Any],
    required_array_paths: list[str],
    optional_array_paths: list[str],
    required_groups: dict[str, type],
    optional_groups: dict[str, type],
    resolve_required_array: Any,
    resolve_optional_array: Any,
    resolve_required_group: Any,
    resolve_optional_group: Any,
) -> None:
    """Submit all I/O work to a thread pool and collect results."""
    with ThreadPoolExecutor(max_workers=_MAX_WORKERS) as executor:
        futures = []
        for path in required_array_paths:
            futures.append(executor.submit(resolve_required_array, path))
        for path in optional_array_paths:
            futures.append(executor.submit(resolve_optional_array, path))
        for path, gtype in required_groups.items():
            futures.append(executor.submit(resolve_required_group, path, gtype))
        for path, gtype in optional_groups.items():
            futures.append(executor.submit(resolve_optional_group, path, gtype))

        for future in as_completed(futures):
            result = future.result()
            if result is None:
                continue
            if isinstance(result, tuple):
                members_tree_flat[result[0]] = result[1]
            else:
                for key, value in result:
                    members_tree_flat[key] = value


TBaseGroupv3 = TypeVar("TBaseGroupv3", bound="BaseGroupv05[Any] | BaseGroupv06[Any]")
TAttrsv3 = TypeVar("TAttrsv3", bound=BaseAttrsv3)


def _from_zarr_v3(
    group: zarr.Group,
    group_cls: type[TBaseGroupv3],
    attrs_cls: type[TAttrsv3],
) -> TBaseGroupv3:
    """
    Create a GroupSpec from a potentially unlistable Zarr group.

    This uses methods on the attribute class to get required and optional
    paths to ararys and groups, and then manually constructs the GroupSpec
    from those paths.

    Parameters
    ----------
    group :
        Zarr group to create GroupSpec from.
    group_cls :
        Class of the Group to return.
    attrs_cls :
        Attributes class.
    """
    # on unlistable storage backends, the members of this group will be {}
    group_spec_in: pydantic_zarr.v3.AnyGroupSpec
    group_spec_in = pydantic_zarr.v3.GroupSpec.from_zarr(group, depth=0)
    attrs_dict = group.attrs.asdict()
    if "ome" not in attrs_dict:
        raise ValueError("Zarr group attributes does not contain an 'ome' key")
    ome_attributes = attrs_cls.model_validate(attrs_dict["ome"])

    members_tree_flat: dict[
        str, pydantic_zarr.v3.AnyGroupSpec | pydantic_zarr.v3.AnyArraySpec
    ] = {}

    def _resolve_required_array(array_path: str) -> tuple[str, Any]:
        return ("/" + array_path, check_array_path(group, array_path, expected_zarr_version=3))

    def _resolve_optional_array(array_path: str) -> tuple[str, Any] | None:
        try:
            return ("/" + array_path, check_array_path(group, array_path, expected_zarr_version=3))
        except ValueError:
            return None

    def _resolve_required_group(group_path: str, group_type: type) -> list[tuple[str, Any]]:
        check_group_path(group, group_path, expected_zarr_version=3)
        group_flat = group_type.from_zarr(group[group_path]).to_flat()
        return [("/" + group_path + path, group_flat[path]) for path in group_flat]

    def _resolve_optional_group(group_path: str, group_type: type) -> list[tuple[str, Any]]:
        try:
            check_group_path(group, group_path, expected_zarr_version=3)
        except FileNotFoundError:
            return []
        group_flat = group_type.from_zarr(group[group_path]).to_flat()
        return [("/" + group_path + path, group_flat[path]) for path in group_flat]

    _submit_and_collect(
        members_tree_flat,
        attrs_cls.get_array_paths(ome_attributes),
        attrs_cls.get_optional_array_paths(ome_attributes),
        attrs_cls.get_group_paths(ome_attributes),
        attrs_cls.get_optional_group_paths(ome_attributes),
        _resolve_required_array,
        _resolve_optional_array,
        _resolve_required_group,
        _resolve_optional_group,
    )

    members_normalized: pydantic_zarr.v3.AnyGroupSpec
    members_normalized = pydantic_zarr.v3.GroupSpec.from_flat(members_tree_flat)
    return group_cls(  # type: ignore[return-value]
        members=members_normalized.members, attributes=group_spec_in.attributes
    )


def get_store_path(store: Store) -> str:
    """
    Get a path from a zarr store
    """
    if hasattr(store, "path"):
        return store.path  # type: ignore[no-any-return]

    return ""


T = TypeVar("T")


def duplicates(values: Iterable[T]) -> dict[T, int]:
    """
    Takes a sequence of hashable elements and returns a dict where the keys are the
    elements of the input that occurred at least once, and the values are the
    frequencies of those elements.
    """
    counts = Counter(values)
    return {k: v for k, v in counts.items() if v > 1}


def dataclass_to_pydantic(dataclass_type: type) -> type[pydantic.BaseModel]:
    """Convert a dataclass to a Pydantic model.

    Parameters
    ----------
    dataclass_type : type
        The dataclass to convert to a Pydantic model.

    Returns
    -------
    type[pydantic.BaseModel] a Pydantic model class.
    """
    if not is_dataclass(dataclass_type):
        raise TypeError(f"{dataclass_type} is not a dataclass")

    field_definitions = {}
    for _field in fields(dataclass_type):
        if _field.default is not MISSING:
            # Default value is provided
            field_definitions[_field.name] = (_field.type, _field.default)
        elif _field.default_factory is not MISSING:
            # Default factory is provided
            field_definitions[_field.name] = (_field.type, _field.default_factory())
        else:
            # No default value
            field_definitions[_field.name] = (_field.type, Ellipsis)

    return create_model(dataclass_type.__name__, **field_definitions)  # type: ignore[no-any-return, call-overload]
