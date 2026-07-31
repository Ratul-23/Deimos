"""What each expression command checks."""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Literal

from loguru import logger
from wizwalker import XYZ, MemoryReadError
from wizwalker.extensions.wizsprinter import SprintyClient  # ty: ignore[unresolved-import]
from wizwalker.memory import DynamicClientObject, Window

from ...teleport_math import calc_Distance
from ...utils import get_window_from_path, is_visible_by_path
from ..ast import (
    CommandExpression,
    Eval,
    EvalKind,
    ExprKind,
    IdentExpression,
    PlayerSelector,
    StringExpression,
    VMError,
)

if TYPE_CHECKING:
    from ..vm import VM


@dataclass
class EvalContext:
    """What one predicate check needs."""

    vm: VM
    clients: list[SprintyClient]
    selector: PlayerSelector
    expression: CommandExpression


type Predicate = Callable[[EvalContext], Awaitable[bool]]

# Filled by the @predicate decorators below.
# anyplayer holds when one client matches. Any other selector needs them all.
PREDICATES: dict[ExprKind, Predicate] = {}


def _same_text(left: str | None, right: str | None) -> bool:
    """Whether two names match, ignoring case."""
    if left is None or right is None:
        return left is right

    return left.lower() == right.lower()


# Window checks are polled. Report each bad path once.
_REPORTED_BAD_PATHS: set[str] = set()


def _no_match(ctx: EvalContext) -> bool:
    """False, dropping any earlier anyplayer matches."""
    if ctx.selector.any_player:
        ctx.vm._any_player_client = []

    return False


def _valid_window_path(value: Any) -> list[str] | None:
    """A value as a window path."""
    if isinstance(value, list) and all(isinstance(part, str) for part in value):
        return value

    described: str = repr(value)

    if described not in _REPORTED_BAD_PATHS:
        _REPORTED_BAD_PATHS.add(described)
        logger.warning(f"Expected a window path, got {described}")

    return None


def predicate(kind: ExprKind) -> Callable[[Predicate], Predicate]:
    """Register an ExprKind's check."""

    def register(fn: Predicate) -> Predicate:
        """Add the check to the registry."""
        if kind in PREDICATES:
            raise ValueError(f"{kind.name} already has a predicate")

        PREDICATES[kind] = fn
        return fn

    return register


@predicate(ExprKind.zone_changed)
async def zone_changed(ctx: EvalContext) -> bool:
    """Whether the zone changed since `logzone`."""
    # Given a zone, did it arrive. Given none, did the zone move. First look only baselines.
    expected_zone: str | None = None

    if len(ctx.expression.command.data) > 1:
        expected_zone = await ctx.vm._extract_data_info(ctx.expression.command.data[1])

    if ctx.selector.any_player:
        ctx.vm._any_player_client = []
        found_any: bool = False

        for client in ctx.vm._clients:
            current_zone: str | None = await client.zone_name()
            last_zone: str | None = ctx.vm.logged_data["zone"].get(client.title, None)

            if current_zone is None:
                continue

            if expected_zone is not None:
                if current_zone.lower() == expected_zone.lower():
                    ctx.vm._any_player_client.append(client)
                    ctx.vm.logged_data["zone"][client.title] = current_zone.lower()
                    found_any = True

            else:
                if last_zone is None:
                    ctx.vm.logged_data["zone"][client.title] = current_zone.lower()

                elif current_zone.lower() != last_zone.lower():
                    ctx.vm._any_player_client.append(client)
                    ctx.vm.logged_data["zone"][client.title] = current_zone.lower()
                    found_any = True

        return found_any

    else:
        all_valid: bool = True

        for client in ctx.clients:
            current_zone: str | None = await client.zone_name()
            last_zone: str | None = ctx.vm.logged_data["zone"].get(client.title, None)

            if current_zone is None:
                all_valid = False

                if expected_zone is None:
                    continue
                else:
                    break

            if expected_zone is not None:
                if current_zone.lower() != expected_zone.lower():
                    all_valid = False
                    break

                ctx.vm.logged_data["zone"][client.title] = current_zone.lower()

            else:
                if last_zone is None:
                    ctx.vm.logged_data["zone"][client.title] = current_zone.lower()
                    all_valid = False

                elif current_zone.lower() == last_zone.lower():
                    all_valid = False

                else:
                    ctx.vm.logged_data["zone"][client.title] = current_zone.lower()

        return all_valid


@predicate(ExprKind.goal_changed)
async def goal_changed(ctx: EvalContext) -> bool:
    """Whether the goal changed since `loggoal`."""
    # Given a goal, did it arrive. Given none, did the goal move. First look only baselines.
    expected_goal: str | None = None

    if len(ctx.expression.command.data) > 1:
        expected_goal = await ctx.vm._extract_data_info(ctx.expression.command.data[1])
        if expected_goal is None:
            logger.error("Failed to extract goal name from expression")
            return _no_match(ctx)

        expected_goal = expected_goal.lower()

    if ctx.selector.any_player:
        ctx.vm._any_player_client = []
        found_any: bool = False

        for client in ctx.vm._clients:
            current_goal: str = (await ctx.vm._fetch_tracked_goal_text(client)).lower()
            last_goal: str | None = ctx.vm.logged_data["goal"].get(client.title, None)

            if expected_goal is not None:
                if current_goal == expected_goal:
                    ctx.vm._any_player_client.append(client)
                    ctx.vm.logged_data["goal"][client.title] = current_goal
                    found_any = True

            else:
                if last_goal is None:
                    ctx.vm.logged_data["goal"][client.title] = current_goal

                elif current_goal != last_goal:
                    ctx.vm._any_player_client.append(client)
                    ctx.vm.logged_data["goal"][client.title] = current_goal
                    found_any = True

        return found_any

    else:
        all_valid: bool = True

        for client in ctx.clients:
            current_goal: str = (await ctx.vm._fetch_tracked_goal_text(client)).lower()
            last_goal: str | None = ctx.vm.logged_data["goal"].get(client.title, None)

            if expected_goal is not None:
                if current_goal != expected_goal:
                    all_valid = False
                    break

                ctx.vm.logged_data["goal"][client.title] = current_goal

            else:
                if last_goal is None:
                    ctx.vm.logged_data["goal"][client.title] = current_goal
                    all_valid = False

                elif current_goal == last_goal:
                    all_valid = False

                else:
                    ctx.vm.logged_data["goal"][client.title] = current_goal

        return all_valid


@predicate(ExprKind.quest_changed)
async def quest_changed(ctx: EvalContext) -> bool:
    """Whether the quest changed since `logquest`."""
    # Given a quest, did it start. Given none, did the quest move. First look only baselines.
    if len(ctx.expression.command.data) > 1:
        expected_quest: str | None = await ctx.vm._extract_data_info(ctx.expression.command.data[1])
        if expected_quest is None:
            logger.error("Failed to extract quest name from expression")
            return _no_match(ctx)

        expected_quest = expected_quest.lower()

        if ctx.selector.any_player:
            ctx.vm._any_player_client = []
            found_any: bool = False

            for client in ctx.vm._clients:
                current_quest: str = (await ctx.vm._fetch_tracked_quest_text(client)).lower()
                last_quest: str | None = ctx.vm.logged_data["quest"].get(client.title, None)

                if current_quest == expected_quest and (last_quest is None or current_quest != last_quest):
                    ctx.vm._any_player_client.append(client)
                    ctx.vm.logged_data["quest"][client.title] = current_quest
                    found_any = True

            return found_any

        else:
            all_match: bool = True

            for client in ctx.clients:
                current_quest: str = (await ctx.vm._fetch_tracked_quest_text(client)).lower()
                last_quest: str | None = ctx.vm.logged_data["quest"].get(client.title, None)

                if current_quest != expected_quest or (last_quest is not None and current_quest == last_quest):
                    all_match = False
                    break

                ctx.vm.logged_data["quest"][client.title] = current_quest

            return all_match

    else:
        if ctx.selector.any_player:
            ctx.vm._any_player_client = []
            found_any: bool = False

            for client in ctx.vm._clients:
                current_quest: str = (await ctx.vm._fetch_tracked_quest_text(client)).lower()
                last_quest: str | None = ctx.vm.logged_data["quest"].get(client.title, None)

                if last_quest is None:
                    ctx.vm.logged_data["quest"][client.title] = current_quest

                elif current_quest != last_quest:
                    ctx.vm._any_player_client.append(client)
                    ctx.vm.logged_data["quest"][client.title] = current_quest
                    found_any = True

            return found_any

        else:
            all_changed: bool = True

            for client in ctx.clients:
                current_quest: str = (await ctx.vm._fetch_tracked_quest_text(client)).lower()
                last_quest: str | None = ctx.vm.logged_data["quest"].get(client.title, None)

                if last_quest is None:
                    ctx.vm.logged_data["quest"][client.title] = current_quest
                    all_changed = False

                elif current_quest == last_quest:
                    all_changed = False

                else:
                    ctx.vm.logged_data["quest"][client.title] = current_quest

            return all_changed


@predicate(ExprKind.items_dropped)
async def items_dropped(ctx: EvalContext) -> bool:
    """Whether an item, or any item in a list, has dropped."""
    raw_items: Any = ctx.expression.command.data[1]
    item_names: str | list[str] = (
        raw_items if isinstance(raw_items, list) else await ctx.vm._extract_data_info(raw_items)
    )

    if ctx.selector.any_player:
        ctx.vm._any_player_client = []
        found_any: bool = False

        for client in ctx.vm._clients:
            if await ctx.vm._check_drops(client, item_names):
                ctx.vm._any_player_client.append(client)
                found_any = True

        return found_any

    else:
        for client in ctx.clients:
            if not await ctx.vm._check_drops(client, item_names):
                return False

        return True


@predicate(ExprKind.window_visible)
async def window_visible(ctx: EvalContext) -> bool:
    """Whether the window at a path is visible."""
    # Resolving hands back the path itself, with any $constant naming it already substituted.
    path: list[str] | None = _valid_window_path(await ctx.vm._extract_data_info(ctx.expression.command.data[1]))

    if path is None:
        return _no_match(ctx)

    if ctx.selector.any_player:
        ctx.vm._any_player_client = []
        found_any: bool = False

        for client in ctx.vm._clients:
            if await is_visible_by_path(client, path):
                ctx.vm._any_player_client.append(client)
                found_any = True

        return found_any

    else:
        for client in ctx.clients:
            if not await is_visible_by_path(client, path):
                return False

        return True


@predicate(ExprKind.window_disabled)
async def window_disabled(ctx: EvalContext) -> bool:
    """Whether the window at a path is greyed out."""
    # Resolving hands back the path itself, with any $constant naming it already substituted.
    path: list[str] | None = _valid_window_path(await ctx.vm._extract_data_info(ctx.expression.command.data[1]))

    if path is None:
        return _no_match(ctx)

    if ctx.selector.any_player:
        ctx.vm._any_player_client = []
        found_any: bool = False

        for client in ctx.vm._clients:
            root: Window = client.root_window
            window: Window | Literal[False] = await get_window_from_path(root, path)

            if window and await window.is_control_grayed():
                ctx.vm._any_player_client.append(client)
                found_any = True

        return found_any

    else:
        for client in ctx.clients:
            root: Window = client.root_window
            window: Window | Literal[False] = await get_window_from_path(root, path)

            if not window or not await window.is_control_grayed():
                return False

        return True


@predicate(ExprKind.in_range)
async def in_range(ctx: EvalContext) -> bool:
    """Whether a named entity is within teleport range."""
    # Our own wizards show up in the entity list. Collect ids to skip them.
    own_ids: set[int] = {await client.client_object.global_id_full() for client in ctx.vm._clients}
    target: str = str(await ctx.vm._extract_data_info(ctx.expression.command.data[1])).lower()

    if ctx.selector.any_player:
        ctx.vm._any_player_client = []
        found_any: bool = False

        for client in ctx.vm._clients:
            entities: list[DynamicClientObject] = await client.get_base_entity_list()

            for entity in entities:
                entity_gid: int = await entity.global_id_full()

                if entity_gid in own_ids:
                    continue

                entity_name: str | None = await entity.object_name()

                if not entity_name:
                    continue

                if target in entity_name.lower():
                    ctx.vm._any_player_client.append(client)
                    found_any = True
                    break

        return found_any

    else:
        for client in ctx.clients:
            entities: list[DynamicClientObject] = await client.get_base_entity_list()
            found: bool = False

            for entity in entities:
                entity_gid: int = await entity.global_id_full()

                if entity_gid in own_ids:
                    continue

                entity_name: str | None = await entity.object_name()

                if not entity_name:
                    continue

                if target in entity_name.lower():
                    found = True
                    break

            if not found:
                return False

        return True


@predicate(ExprKind.same_place)
async def same_place(ctx: EvalContext) -> bool:
    """Whether all clients stand together."""
    if len(ctx.clients) < 2:
        return True

    # Clients see each other as entities only when together. Count the sightings.
    own_ids: set[int] = {await client.client_object.global_id_full() for client in ctx.clients}
    target: int = len(own_ids)

    for client in ctx.clients:
        entities: list[DynamicClientObject] = await client.get_base_entity_list()
        found: int = 0

        for entity in entities:
            entity_gid: int = await entity.global_id_full()

            if entity_gid in own_ids:
                found += 1

        if found != target:
            return False

    return True


@predicate(ExprKind.in_zone)
async def in_zone(ctx: EvalContext) -> bool:
    """Whether the client is in a given zone."""
    # Same zone for every client. Resolve once.
    expected: str | None = await ctx.vm._extract_data_info(ctx.expression.command.data[1])

    if ctx.selector.any_player:
        ctx.vm._any_player_client = []
        found_any: bool = False

        for client in ctx.vm._clients:
            zone: str | None = await client.zone_name()

            if _same_text(zone, expected):
                ctx.vm._any_player_client.append(client)
                found_any = True

        return found_any

    else:
        for client in ctx.clients:
            zone: str | None = await client.zone_name()

            if not _same_text(zone, expected):
                return False

        return True


@predicate(ExprKind.same_zone)
async def same_zone(ctx: EvalContext) -> bool:
    """Whether all clients share a zone."""
    if len(ctx.clients) == 0:
        return True

    expected_zone: str | None = await ctx.clients[0].zone_name()

    for client in ctx.clients[1:]:
        if not _same_text(await client.zone_name(), expected_zone):
            return False

    return True


@predicate(ExprKind.same_quest)
async def same_quest(ctx: EvalContext) -> bool:
    """Whether all clients track one quest."""
    if len(ctx.clients) == 0:
        return True

    expected_quest_text: str = await ctx.vm._fetch_tracked_quest_text(ctx.clients[0])

    for client in ctx.clients[1:]:
        quest_text: str = await ctx.vm._fetch_tracked_quest_text(client)
        if expected_quest_text != quest_text:
            return False

    return True


@predicate(ExprKind.same_yaw)
async def same_yaw(ctx: EvalContext) -> bool:
    """Whether all clients face one way."""
    if len(ctx.clients) == 0:
        return True

    # Yaw reads finer than a wizard can be aimed. Compare rounded.
    expected_yaw: float = await ctx.clients[0].body.yaw()
    rounded_expected_yaw: float = round(expected_yaw, 1)

    for client in ctx.clients[1:]:
        yaw: float = await client.body.yaw()
        rounded_client_yaw: float = round(yaw, 1)

        if rounded_expected_yaw != rounded_client_yaw:
            return False

    return True


@predicate(ExprKind.same_xyz)
async def same_xyz(ctx: EvalContext) -> bool:
    """Whether all clients share a position."""
    if len(ctx.clients) == 0:
        return True

    expected_pos: XYZ = await ctx.clients[0].body.position()

    for client in ctx.clients[1:]:
        pos: XYZ = await client.body.position()
        distance: float = calc_Distance(expected_pos, pos)

        # Wizards standing together report slightly different spots. Allow slack.
        if distance > 5.0:
            return False

    return True


@predicate(ExprKind.tracking_quest)
async def tracking_quest(ctx: EvalContext) -> bool:
    """Whether the tracked quest name matches."""
    expected_text: str = str(await ctx.vm._extract_data_info(ctx.expression.command.data[1])).lower()

    if ctx.selector.any_player:
        ctx.vm._any_player_client = []
        found_any: bool = False

        for client in ctx.vm._clients:
            name: str = await ctx.vm._fetch_tracked_quest_text(client)
            if name == expected_text:
                ctx.vm._any_player_client.append(client)
                found_any = True

        return found_any

    else:
        for client in ctx.clients:
            name: str = await ctx.vm._fetch_tracked_quest_text(client)
            if name != expected_text:
                return False

        return True


@predicate(ExprKind.tracking_goal)
async def tracking_goal(ctx: EvalContext) -> bool:
    """Whether the tracked goal text matches."""
    expected_text: str = str(await ctx.vm._extract_data_info(ctx.expression.command.data[1])).lower()

    if ctx.selector.any_player:
        ctx.vm._any_player_client = []
        found_any: bool = False

        for client in ctx.vm._clients:
            text: str = await ctx.vm._fetch_tracked_goal_text(client)
            if text == expected_text:
                ctx.vm._any_player_client.append(client)
                found_any = True

        return found_any

    else:
        for client in ctx.clients:
            text: str = await ctx.vm._fetch_tracked_goal_text(client)
            if text != expected_text:
                return False

        return True


@predicate(ExprKind.loading)
async def loading(ctx: EvalContext) -> bool:
    """Whether a loading screen is up."""
    if ctx.selector.any_player:
        ctx.vm._any_player_client = []
        found_any: bool = False

        for client in ctx.vm._clients:
            if await client.is_loading():
                ctx.vm._any_player_client.append(client)
                found_any = True

        return found_any

    else:
        for client in ctx.clients:
            if not await client.is_loading():
                return False

        return True


@predicate(ExprKind.in_combat)
async def in_combat(ctx: EvalContext) -> bool:
    """Whether the client is in combat."""
    if ctx.selector.any_player:
        ctx.vm._any_player_client = []
        found_any: bool = False

        for client in ctx.vm._clients:
            if await client.in_battle():
                ctx.vm._any_player_client.append(client)
                found_any = True

        return found_any

    else:
        for client in ctx.clients:
            if not await client.in_battle():
                return False

        return True


@predicate(ExprKind.has_quest)
async def has_quest(ctx: EvalContext) -> bool:
    """Whether the quest log contains a named quest."""
    expected_text: str = str(await ctx.vm._extract_data_info(ctx.expression.command.data[1])).lower()

    if ctx.selector.any_player:
        ctx.vm._any_player_client = []
        found_any: bool = False

        for client in ctx.vm._clients:
            for _, quest in await ctx.vm._fetch_quests(client):
                if await ctx.vm._fetch_quest_text(client, quest) == expected_text:
                    ctx.vm._any_player_client.append(client)
                    found_any = True
                    break

        return found_any

    else:
        for client in ctx.clients:
            found: bool = False

            for _, quest in await ctx.vm._fetch_quests(client):
                if await ctx.vm._fetch_quest_text(client, quest) == expected_text:
                    found = True
                    break

            if not found:
                return False

        return True


@predicate(ExprKind.has_dialogue)
async def has_dialogue(ctx: EvalContext) -> bool:
    """Whether a dialogue window is open."""
    if ctx.selector.any_player:
        ctx.vm._any_player_client = []
        found_any: bool = False

        for client in ctx.vm._clients:
            if await client.is_in_dialog():
                ctx.vm._any_player_client.append(client)
                found_any = True

        return found_any

    else:
        for client in ctx.clients:
            if not await client.is_in_dialog():
                return False

        return True


@predicate(ExprKind.has_xyz)
async def has_xyz(ctx: EvalContext) -> bool:
    """Whether the client stands at a position."""
    target_pos: Any = await ctx.vm._extract_data_info(ctx.expression.command.data[1])

    if not isinstance(target_pos, XYZ):
        raise VMError(f"Expected a position, got {target_pos!r}")

    if ctx.selector.any_player:
        ctx.vm._any_player_client = []
        found_any: bool = False

        for client in ctx.vm._clients:
            # A unit of slack absorbs drift between written and reported.
            client_pos: XYZ = await client.body.position()
            if calc_Distance(target_pos, client_pos) <= 1:
                ctx.vm._any_player_client.append(client)
                found_any = True

        return found_any

    else:
        for client in ctx.clients:
            # A unit of slack absorbs drift between written and reported.
            client_pos: XYZ = await client.body.position()
            if calc_Distance(target_pos, client_pos) > 1:
                return False

        return True


@predicate(ExprKind.has_yaw)
async def has_yaw(ctx: EvalContext) -> bool:
    """Whether the client faces a heading."""
    target_yaw: Any = await ctx.vm._extract_data_info(ctx.expression.command.data[1])

    if not isinstance(target_yaw, (int, float)):
        raise VMError(f"Expected a heading, got {target_yaw!r}")

    # Yaw reads finer than a wizard can be aimed. Compare rounded.
    rounded_target_yaw: float = round(target_yaw, 1)

    if ctx.selector.any_player:
        ctx.vm._any_player_client = []
        found_any: bool = False

        for client in ctx.vm._clients:
            client_yaw: float = await client.body.yaw()
            rounded_client_yaw: float = round(client_yaw, 1)

            if rounded_client_yaw == rounded_target_yaw:
                ctx.vm._any_player_client.append(client)
                found_any = True

        return found_any

    else:
        for client in ctx.clients:
            client_yaw: float = await client.body.yaw()
            rounded_client_yaw: float = round(client_yaw, 1)

            if rounded_client_yaw != rounded_target_yaw:
                return False

        return True


@dataclass
class StatContext:
    """What a stat reader is given."""

    vm: VM
    client: SprintyClient
    eval: Eval


type StatValue = float | int | str | list[float] | list[str]

type StatEvaluator = Callable[[StatContext], Awaitable[StatValue]]

# Filled by the @stat decorators below.
STATS: dict[EvalKind, StatEvaluator] = {}


def stat(kind: EvalKind) -> Callable[[StatEvaluator], StatEvaluator]:
    """Register an EvalKind's reader."""

    def register(fn: StatEvaluator) -> StatEvaluator:
        """Add the reader to the registry."""
        if kind in STATS:
            raise ValueError(f"{kind.name} already has an evaluator")

        STATS[kind] = fn
        return fn

    return register


def _window_path(ctx: StatContext) -> list[str] | None:
    """The window path a stat reads."""
    path: Any = ctx.eval.args[0]

    if isinstance(path, IdentExpression):
        path = ctx.vm._constants.get(path.ident, path.ident)

    return _valid_window_path(path)


async def _window_text(window: Window) -> str:
    """A window's text."""
    try:
        text: str = await window.maybe_text()
        if text:
            return text

    except (ValueError, MemoryReadError):
        pass

    try:
        return await window.read_wide_string_from_offset(616)

    except (ValueError, MemoryReadError):
        return ""


def _numbers_in(text: str) -> list[float]:
    """Every number in a window's text."""
    result: list[float] = []

    for part in text.split("/"):
        numeric_text: str = "".join(char for char in part if char.isdigit() or char == "." or char == "-")
        result.append(float(numeric_text) if numeric_text else 0.0)

    return result


@stat(EvalKind.duel_round)
async def read_duel_round(ctx: StatContext) -> StatValue:
    """Current combat round, 0 outside battle."""
    if not await ctx.client.in_battle():
        return 0

    return await ctx.client.duel.round_num()


@stat(EvalKind.account_level)
async def read_account_level(ctx: StatContext) -> StatValue:
    """The wizard's level."""
    return await ctx.client.stats.reference_level()


@stat(EvalKind.health)
async def read_health(ctx: StatContext) -> StatValue:
    """Current health."""
    return await ctx.client.stats.current_hitpoints()


@stat(EvalKind.max_health)
async def read_max_health(ctx: StatContext) -> StatValue:
    """Maximum health."""
    return await ctx.client.stats.max_hitpoints()


@stat(EvalKind.mana)
async def read_mana(ctx: StatContext) -> StatValue:
    """Current mana."""
    return await ctx.client.stats.current_mana()


@stat(EvalKind.max_mana)
async def read_max_mana(ctx: StatContext) -> StatValue:
    """Maximum mana."""
    return await ctx.client.stats.max_mana()


@stat(EvalKind.energy)
async def read_energy(ctx: StatContext) -> StatValue:
    """Current energy."""
    return await ctx.client.current_energy()


@stat(EvalKind.max_energy)
async def read_max_energy(ctx: StatContext) -> StatValue:
    """Maximum energy."""
    return await ctx.client.stats.energy_max()


@stat(EvalKind.bagcount)
async def read_bagcount(ctx: StatContext) -> StatValue:
    """Items currently in the backpack."""
    return (await ctx.client.backpack_space())[0]


@stat(EvalKind.max_bagcount)
async def read_max_bagcount(ctx: StatContext) -> StatValue:
    """Backpack capacity."""
    return (await ctx.client.backpack_space())[1]


@stat(EvalKind.gold)
async def read_gold(ctx: StatContext) -> StatValue:
    """Gold carried."""
    return await ctx.client.stats.current_gold()


@stat(EvalKind.max_gold)
async def read_max_gold(ctx: StatContext) -> StatValue:
    """Gold the purse can hold."""
    return await ctx.client.stats.base_gold_pouch()


@stat(EvalKind.windowtext)
async def read_windowtext(ctx: StatContext) -> StatValue:
    """A window's text, lowercased."""
    path: list[str] | None = _window_path(ctx)

    if path is None:
        return ""

    try:
        window: Window | Literal[False] = await get_window_from_path(ctx.client.root_window, path)
        if window:
            return (await _window_text(window)).lower()

        return ""

    except (ValueError, MemoryReadError):
        return ""


@stat(EvalKind.windownum)
async def read_windownum(ctx: StatContext) -> StatValue:
    """Numbers in a window's text."""
    path: list[str] | None = _window_path(ctx)

    if path is None:
        return [0.0]

    try:
        window: Window | Literal[False] = await get_window_from_path(ctx.client.root_window, path)
        if window:
            text: str = await _window_text(window)

            if text:
                return _numbers_in(text)

    except (ValueError, MemoryReadError):
        pass

    return [0.0]


@stat(EvalKind.playercount)
async def read_playercount(ctx: StatContext) -> StatValue:
    """How many clients the VM drives."""
    return len(ctx.vm._clients)


@stat(EvalKind.counter)
async def read_counter(ctx: StatContext) -> StatValue:
    """What a named counter holds."""
    name: Any = ctx.eval.args[0]
    assert isinstance(name, StringExpression), "Counter name must be a string"

    # Raising sends the branch forward, which could kill the bot.
    return ctx.vm._counters.get(name.string, 0)


@stat(EvalKind.potioncount)
async def read_potioncount(ctx: StatContext) -> StatValue:
    """Potion charges remaining."""
    return await ctx.client.stats.potion_charge()


@stat(EvalKind.max_potioncount)
async def read_max_potioncount(ctx: StatContext) -> StatValue:
    """Potion capacity."""
    return await ctx.client.stats.potion_max()


@stat(EvalKind.any_player_list)
async def read_any_player_list(ctx: StatContext) -> StatValue:
    """Titles that satisfied the last `any`."""
    return [client.title for client in ctx.vm._any_player_client]
