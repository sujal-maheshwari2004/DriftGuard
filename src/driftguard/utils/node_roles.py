"""
Role bookkeeping for graph nodes.

A node is keyed by its normalized text, so the same text legitimately turns up
in more than one position: "restart the server" can be the action of one event
and the outcome of another. Storing a single role meant the second write called
add_node() again and silently replaced the first node's role, frequency and
first_seen, which left the action unreachable.

Roles are therefore a set. These helpers accept the legacy single-string form
as well, so graphs written by earlier versions load unchanged.
"""

_SEPARATOR = ","


def parse_roles(value) -> tuple[str, ...]:
    """
    Normalize a stored role value into a tuple, preserving insertion order.

    Accepts the legacy `"action"` string, the serialized `"action,outcome"`
    form, an iterable of roles, or None.
    """

    if value is None:
        return ()

    if isinstance(value, str):
        candidates = value.split(_SEPARATOR)
    else:
        candidates = value

    roles = [role.strip() for role in candidates if role and role.strip()]
    return tuple(dict.fromkeys(roles))


def add_role(value, role: str) -> tuple[str, ...]:
    """
    Return the roles with `role` appended, or unchanged if already present.
    """

    roles = parse_roles(value)

    if role in roles:
        return roles

    return roles + (role,)


def has_role(value, role: str) -> bool:
    return role in parse_roles(value)


def serialize_roles(value) -> str:
    """
    Render roles for backends with a single text column.
    """

    return _SEPARATOR.join(parse_roles(value))
