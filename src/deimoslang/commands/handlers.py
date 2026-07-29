"""What each statement command does."""

from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from math import isfinite
from typing import TYPE_CHECKING, Any, Literal

from loguru import logger
from wizwalker import XYZ, Client, Keycode
from wizwalker.extensions.scripting.deck_builder import DeckBuilder
from wizwalker.extensions.wizsprinter import SprintyClient  # ty: ignore[unresolved-import]
from wizwalker.extensions.wizsprinter.wiz_navigator import toZone  # ty: ignore[unresolved-import]
from wizwalker.memory import DynamicClientObject, Window

from ...auto_pet import dancedance
from ...command_parser import teleport_to_friend_from_list
from ...config_combat import default_config, delegate_combat_configs
from ...dance_game_hook import attempt_activate_dance_hook
from ...deck_encoder import DeckEncoderDecoder
from ...teleport_math import collision_tp
from ...utils import (
    click_window_by_path,
    get_window_from_path,
    is_free,
    is_visible_by_path,
    logout_and_in,
    refill_potions,
    refill_potions_if_needed,
)
from ..ast import (
    ClickKind,
    CommandKind,
    CursorKind,
    Expression,
    IdentExpression,
    Instruction,
    PlayerSelector,
    TeleportKind,
    VMError,
    WaitforKind,
)

if TYPE_CHECKING:
    from ..vm import VM

type EvalArg = Callable[[Any, SprintyClient], Awaitable[Any]]


@dataclass
class ExecContext:
    """What one handler call needs."""

    vm: VM
    clients: list[SprintyClient]
    args: list[Any]
    eval_arg: EvalArg
    instruction: Instruction | None = None


@dataclass
class InstructionContext:
    """What an instruction handler is given."""

    vm: VM
    instruction: Instruction


type Handler = Callable[[ExecContext], Awaitable[None]]

type InstructionHandler = Callable[[InstructionContext], Awaitable[None]]

# Filled by the decorators below.
HANDLERS: dict[str, Handler] = {}

INSTRUCTION_HANDLERS: dict[str, InstructionHandler] = {}


def handler(kind: CommandKind) -> Callable[[Handler], Handler]:
    """Register a CommandKind's behaviour."""

    def register(fn: Handler) -> Handler:
        """Add the handler to the registry."""
        if kind.name in HANDLERS:
            raise ValueError(f"{kind.name} already has a handler")

        HANDLERS[kind.name] = fn
        return fn

    return register


def instruction_handler(name: str) -> Callable[[InstructionHandler], InstructionHandler]:
    """Register an InstructionKind's behaviour."""

    def register(fn: InstructionHandler) -> InstructionHandler:
        """Add the handler to the registry."""
        if name in INSTRUCTION_HANDLERS:
            raise ValueError(f"{name} already has an instruction handler")

        INSTRUCTION_HANDLERS[name] = fn
        return fn

    return register


def _as_number(value: Any) -> float:
    """An argument as a number."""
    # Whole number grows without limit. Too big to weigh lands here.
    try:
        number: float = float(value)

    except (TypeError, ValueError, OverflowError):
        raise VMError(f"Expected a number, got {value!r}") from None

    # Handlers round down. Infinity has no whole number.
    if not isfinite(number):
        raise VMError(f"Expected a number, got {value!r}")

    return number


def _as_xyz(value: Any) -> XYZ:
    """An argument as a position."""
    if not isinstance(value, XYZ):
        raise VMError(f"Expected a position, got {value!r}")

    return value


def _as_window_path(value: Any) -> list[str]:
    """An argument as a window path."""
    if not isinstance(value, list) or not all(isinstance(part, str) for part in value):
        raise VMError(f"Expected a window path, got {value!r}")

    return value


def _joined_zone_path(value: Any) -> str:
    """An argument as a zone path."""
    if isinstance(value, list):
        return "/".join(str(part) for part in value)

    return str(value)


async def _record_baseline(ctx: ExecContext, kind: str, read: Callable[[SprintyClient], Awaitable[str | None]]) -> None:
    """Record the baseline for `changed`."""
    for client in ctx.clients:
        current: str | None = await read(client)
        ctx.vm.logged_data[kind][client.title] = current
        logger.debug(f"Client {client.title}: Current {kind}: {current}")


async def _tp_to_position(ctx: ExecContext, tg: asyncio.TaskGroup) -> None:
    """Teleport every client onto a position."""
    for client in ctx.clients:
        pos: XYZ = _as_xyz(await ctx.eval_arg(ctx.args[1], client))
        tg.create_task(client.teleport(pos))


async def _tp_by_offset(ctx: ExecContext, tg: asyncio.TaskGroup, sign: int) -> None:
    """Teleport every client a distance off."""
    for client in ctx.clients:
        offset: XYZ = _as_xyz(await ctx.eval_arg(ctx.args[1], client))
        current: XYZ = await client.body.position()
        target: XYZ = XYZ(
            current.x + sign * offset.x,
            current.y + sign * offset.y,
            current.z + sign * offset.z,
        )
        tg.create_task(client.teleport(target))


async def _tp_to_entity(ctx: ExecContext, tg: asyncio.TaskGroup, vague: bool) -> None:
    """Teleport every client to the closest entity a name picks out."""
    # A nav marker sits just before the name, and means walk there rather than teleport.
    use_navmap: bool = len(ctx.args) > 2 and ctx.args[-2] == TeleportKind.nav

    # The parser always hands over a string, but a constant standing in for it may hold anything.
    name: str = str(await ctx.eval_arg(ctx.args[-1], ctx.clients[0]))

    async def tp_to_closest(client: SprintyClient) -> None:
        """Teleport one client to that entity."""
        entity: DynamicClientObject | None = (
            await client.find_closest_by_vague_name(name) if vague else await client.find_closest_by_name(name)
        )

        if entity:
            pos: XYZ = await entity.location()

            if use_navmap:
                await collision_tp(client, pos)
            else:
                await client.teleport(pos)

    for client in ctx.clients:
        tg.create_task(tp_to_closest(client))


async def _tp_to_mob(ctx: ExecContext, tg: asyncio.TaskGroup) -> None:
    """Teleport every client to the closest mob."""
    for client in ctx.clients:
        tg.create_task(client.tp_to_closest_mob())


async def _tp_to_quest(ctx: ExecContext, tg: asyncio.TaskGroup) -> None:
    """Send every client to its quest."""
    for client in ctx.clients:
        pos: XYZ = await client.quest_position.position()
        tg.create_task(collision_tp(client, pos))


async def _tp_to_friend_icon(ctx: ExecContext, tg: asyncio.TaskGroup) -> None:
    """Teleport to the friend at the icon."""

    async def tp_to_icon(client: SprintyClient) -> None:
        """Teleport one client to the icon."""
        async with client.mouse_handler:
            await teleport_to_friend_from_list(client, icon_list=2, icon_index=0)

    for client in ctx.clients:
        tg.create_task(tp_to_icon(client))


async def _tp_to_friend_name(ctx: ExecContext, tg: asyncio.TaskGroup) -> None:
    """Teleport to a friend by name."""
    # A constant standing in for the name may hold anything, so settle it on a string.
    friend_name: str = str(await ctx.eval_arg(ctx.args[-1], ctx.clients[0]))

    async def tp_to_named_friend(client: SprintyClient) -> None:
        """Teleport one client to that friend."""
        async with client.mouse_handler:
            await teleport_to_friend_from_list(client, name=friend_name)

    for client in ctx.clients:
        tg.create_task(tp_to_named_friend(client))


async def _tp_to_client(ctx: ExecContext, tg: asyncio.TaskGroup) -> None:
    """Teleport every client onto another."""
    num: int = await ctx.eval_arg(ctx.args[-1], ctx.clients[0])
    target_client: SprintyClient | None = ctx.vm.player_by_num(num)

    if target_client is None:
        raise VMError(f"Cannot teleport to p{num}: no such client is open")

    target_pos: XYZ = await target_client.body.position()

    for client in ctx.clients:
        tg.create_task(client.teleport(target_pos))


@handler(CommandKind.teleport)
async def teleport(ctx: ExecContext) -> None:
    """Teleport each client to its destination."""
    assert isinstance(ctx.args[0], TeleportKind)

    async with asyncio.TaskGroup() as tg:
        match ctx.args[0]:
            case TeleportKind.position:
                await _tp_to_position(ctx, tg)

            case TeleportKind.plusteleport:
                await _tp_by_offset(ctx, tg, sign=1)

            case TeleportKind.minusteleport:
                await _tp_by_offset(ctx, tg, sign=-1)

            case TeleportKind.entity_literal:
                await _tp_to_entity(ctx, tg, vague=False)

            case TeleportKind.entity_vague:
                await _tp_to_entity(ctx, tg, vague=True)

            case TeleportKind.mob:
                await _tp_to_mob(ctx, tg)

            case TeleportKind.quest:
                await _tp_to_quest(ctx, tg)

            case TeleportKind.friend_icon:
                await _tp_to_friend_icon(ctx, tg)

            case TeleportKind.friend_name:
                await _tp_to_friend_name(ctx, tg)

            case TeleportKind.client_num:
                await _tp_to_client(ctx, tg)

            case _:
                raise VMError(f"Unimplemented teleport kind: {ctx.instruction}")


@handler(CommandKind.goto)
async def goto(ctx: ExecContext) -> None:
    """Walk each client to a position."""
    async with asyncio.TaskGroup() as tg:
        for client in ctx.clients:
            pos: XYZ = _as_xyz(await ctx.eval_arg(ctx.args[0], client))
            tg.create_task(client.goto(pos.x, pos.y))


@handler(CommandKind.sendkey)
async def sendkey(ctx: ExecContext) -> None:
    """Press a key on each client."""
    args: list[Any] = ctx.args

    async with asyncio.TaskGroup() as tg:
        for client in ctx.clients:
            key: Keycode = await ctx.eval_arg(args[0], client)
            time: float = 0.1 if args[1] is None else _as_number(await ctx.eval_arg(args[1], client))
            tg.create_task(client.send_key(key, time))


# What each simple `waitfor` polls.
WAITFOR_CHECKS: dict[WaitforKind, Callable[[SprintyClient], Awaitable[bool]]] = {
    WaitforKind.dialog: Client.is_in_dialog,
    WaitforKind.battle: Client.in_battle,
    WaitforKind.free: is_free,
}


@handler(CommandKind.waitfor)
async def waitfor(ctx: ExecContext) -> None:
    """Block until a condition holds everywhere."""
    args: list[Any] = ctx.args
    clients: list = ctx.clients
    completion: bool = args[-1]
    assert isinstance(completion, bool)

    async def waitfor_poll(poll: Callable[[], Awaitable[bool]], invert: bool, interval: float = 0.25) -> None:
        """Poll until the condition settles."""
        # Inverting waits for the condition to stop holding.
        while not (invert ^ await poll()):
            await asyncio.sleep(interval)

    async def waitfor_until_settled(poll: Callable[[], Awaitable[bool]], interval: float = 0.25) -> None:
        """Poll, inverting when `completion` was given."""
        await waitfor_poll(poll, completion, interval)

    if args[0] in WAITFOR_CHECKS:
        check: Callable[[Any], Awaitable[bool]] = WAITFOR_CHECKS[args[0]]

        async with asyncio.TaskGroup() as tg:
            for client in clients:

                async def client_matches(client: SprintyClient = client) -> bool:
                    """Whether this client satisfies it."""
                    return await check(client)

                tg.create_task(waitfor_until_settled(client_matches))

    else:
        match args[0]:
            case WaitforKind.zonechange:
                # Completed means the loading screen is gone, not just a new name.
                if completion:
                    async with asyncio.TaskGroup() as tg:
                        for client in clients:
                            tg.create_task(waitfor_poll(client.is_loading, True))

                else:
                    async with asyncio.TaskGroup() as tg:
                        for client in clients:
                            starting_zone: str | None = await client.zone_name()

                            async def left_starting_zone(
                                client: SprintyClient = client, starting_zone: str | None = starting_zone
                            ) -> bool:
                                """Whether this client left its zone."""
                                return starting_zone != (await client.zone_name())

                            tg.create_task(waitfor_poll(left_starting_zone, False))

            case WaitforKind.window:
                window_path: list[str] = _as_window_path(await ctx.eval_arg(args[1], clients[0]))

                async with asyncio.TaskGroup() as tg:
                    for client in clients:

                        async def window_is_visible(client: SprintyClient = client) -> bool:
                            """Whether the window is visible here."""
                            return await is_visible_by_path(client, window_path)

                        tg.create_task(waitfor_until_settled(window_is_visible))

            case _:
                raise VMError(f"Unimplemented waitfor kind: {ctx.instruction}")


@handler(CommandKind.usepotion)
async def usepotion(ctx: ExecContext) -> None:
    """Drink a potion, optionally when low."""
    args: list[Any] = ctx.args

    async def use_potion_if_needed(client: SprintyClient, health_threshold: float, mana_threshold: float) -> None:
        """Drink a potion below the thresholds."""
        async with client.mouse_handler:
            await client.use_potion_if_needed(int(health_threshold), int(mana_threshold))

    async def use_potion(client: SprintyClient) -> None:
        """Drink a potion unconditionally."""
        async with client.mouse_handler:
            await client.use_potion()

    async with asyncio.TaskGroup() as tg:
        if len(args) > 0:
            for client in ctx.clients:
                health_num: float = _as_number(await ctx.eval_arg(args[0], client))
                mana_num: float = _as_number(await ctx.eval_arg(args[1], client))
                tg.create_task(use_potion_if_needed(client, health_num, mana_num))

        else:
            for client in ctx.clients:
                tg.create_task(use_potion(client))


@handler(CommandKind.buypotions)
async def buypotions(ctx: ExecContext) -> None:
    """Refill potions, optionally only when low."""
    ifneeded: bool = await ctx.eval_arg(ctx.args[0], ctx.clients[0])

    async with asyncio.TaskGroup() as tg:
        for client in ctx.clients:
            if ifneeded:
                tg.create_task(refill_potions_if_needed(client, mark=True, recall=True))
            else:
                tg.create_task(refill_potions(client, mark=True, recall=True))


@handler(CommandKind.relog)
async def relog(ctx: ExecContext) -> None:
    """Log each client out and back in."""
    async with asyncio.TaskGroup() as tg:
        for client in ctx.clients:
            tg.create_task(logout_and_in(client))


@handler(CommandKind.click)
async def click(ctx: ExecContext) -> None:
    """Click a screen position or a window."""
    args: list[Any] = ctx.args

    async def click_position(client: SprintyClient, screen_x: float, screen_y: float) -> None:
        """Click a position on one client."""
        async with client.mouse_handler:
            await client.mouse_handler.click(int(screen_x), int(screen_y))

    async def click_window(client: SprintyClient, path: list[str]) -> None:
        """Click a window on one client."""
        await click_window_by_path(client, path)

    async with asyncio.TaskGroup() as tg:
        match args[0]:
            case ClickKind.position:
                for client in ctx.clients:
                    screen_x: float = _as_number(await ctx.eval_arg(args[1], client))
                    screen_y: float = _as_number(await ctx.eval_arg(args[2], client))
                    tg.create_task(click_position(client, screen_x, screen_y))

            case ClickKind.window:
                for client in ctx.clients:
                    path: list[str] = _as_window_path(await ctx.eval_arg(args[1], client))
                    tg.create_task(click_window(client, path))

            case _:
                raise VMError(f"Unimplemented click kind: {ctx.instruction}")


@handler(CommandKind.tozone)
async def tozone(ctx: ExecContext) -> None:
    """Travel each client to a zone."""
    async with asyncio.TaskGroup() as tg:
        for client in ctx.clients:
            zone_path: str = _joined_zone_path(await ctx.eval_arg(ctx.args[0], client))
            tg.create_task(toZone([client], zone_path))


@handler(CommandKind.select_friend)
async def select_friend(ctx: ExecContext) -> None:
    """Pick a friend by name."""
    # The parser always hands over a string, but a constant standing in for the name may hold anything.
    friend_name: str = str(await ctx.eval_arg(ctx.args[0], ctx.clients[0]))

    async with asyncio.TaskGroup() as tg:
        for client in ctx.clients:
            tg.create_task(ctx.vm.select_friend_from_list(client, friend_name))


@handler(CommandKind.autopet)
async def autopet(ctx: ExecContext) -> None:
    """Play the pet dance minigame."""

    async def play_dance_game(client: Client) -> bool:
        """Play one client's pet dance minigame."""
        try:
            logger.debug(f"Client {client.title}: Starting pet dance game.")
            logger.debug(f"Client {client.title}: Activating dance game hook")
            await attempt_activate_dance_hook(client)
            await dancedance(client)
            logger.debug(f"Client {client.title}: Finished pet dance game.")
            return True

        except Exception as error:  # noqa: BLE001
            logger.error(f"Error in pet play dance game for {client.title}: {error}")
            return False

    async def timeout_dance_game(client: Client) -> bool:
        """Play the dance game with a timeout."""
        try:
            return await asyncio.wait_for(play_dance_game(client), timeout=60)
        except TimeoutError:
            logger.error(f"Client {client.title}: Pet dance game timed out.")

        return False

    async with asyncio.TaskGroup() as tg:
        for client in ctx.clients:
            tg.create_task(timeout_dance_game(client))

    logger.debug("All clients have finished pet dance game")


@handler(CommandKind.set_goal)
async def set_goal(ctx: ExecContext) -> None:
    """Record the baseline for `goalchanged`."""
    await _record_baseline(ctx, "goal", ctx.vm._fetch_tracked_goal_text)


@handler(CommandKind.set_quest)
async def set_quest(ctx: ExecContext) -> None:
    """Record the baseline for `questchanged`."""
    await _record_baseline(ctx, "quest", ctx.vm._fetch_tracked_quest_text)


@handler(CommandKind.set_zone)
async def set_zone(ctx: ExecContext) -> None:
    """Record the baseline for `zonechanged`."""
    await _record_baseline(ctx, "zone", lambda client: client.zone_name())


@handler(CommandKind.restart_client)
async def restart_client(ctx: ExecContext) -> None:
    """Relaunch each client and carry on."""
    if ctx.vm.on_restart_client is None:
        logger.warning("Ignoring restartclient: this VM was not given a way to relaunch clients")
        return

    # Mass selector hands back the VM's own list. Swap rewrites it in place.
    restarting: list[SprintyClient] = list(ctx.clients)
    logger.debug(f"Restarting {', '.join(client.title for client in restarting)}")

    ctx.vm.replace_clients(restarting, await ctx.vm.on_restart_client(restarting))


@handler(CommandKind.cursor)
async def cursor(ctx: ExecContext) -> None:
    """Move the cursor to a position or window."""
    args: list[Any] = ctx.args

    async def move_cursor(client: SprintyClient, screen_x: float, screen_y: float) -> None:
        """Move one cursor to a position."""
        async with client.mouse_handler:
            await client.mouse_handler.set_mouse_position(int(screen_x), int(screen_y))

    async def move_cursor_to_window(client: SprintyClient, path: list[str]) -> None:
        """Move one cursor over a window."""
        window: Window | Literal[False] = await get_window_from_path(client.root_window, path)

        if window:
            async with client.mouse_handler:
                await client.mouse_handler.set_mouse_position_to_window(window)

    async with asyncio.TaskGroup() as tg:
        match args[0]:
            case CursorKind.position:
                for client in ctx.clients:
                    screen_x: float = _as_number(await ctx.eval_arg(args[1], client))
                    screen_y: float = _as_number(await ctx.eval_arg(args[2], client))
                    tg.create_task(move_cursor(client, screen_x, screen_y))
                    await asyncio.sleep(0.2)

            case CursorKind.window:
                for client in ctx.clients:
                    path: list[str] = _as_window_path(await ctx.eval_arg(args[1], client))
                    tg.create_task(move_cursor_to_window(client, path))
                    await asyncio.sleep(0.2)


@instruction_handler("kill")
async def instr_kill(ctx: InstructionContext) -> None:
    """Stop the bot where it stands."""
    ctx.vm._any_player_client = []
    ctx.vm.kill()
    logger.debug("Bot Killed")


@instruction_handler("sleep")
async def instr_sleep(ctx: InstructionContext) -> None:
    """Pause the whole program."""
    assert ctx.instruction.data is not None
    time: float = _as_number(await ctx.vm._eval_operand(ctx.instruction.data, None))
    await asyncio.sleep(time)
    ctx.vm.current_task.ip += 1


@instruction_handler("log_single")
async def instr_log_single(ctx: InstructionContext) -> None:
    """Log one value, resolving a bare identifier as a constant."""
    assert isinstance(ctx.instruction.data, Expression)

    if isinstance(ctx.instruction.data, IdentExpression):
        ident: str = ctx.instruction.data.ident

        if ident in ctx.vm._constants:
            logger.debug(f"{ident} = {ctx.vm._constants[ident]}")
        else:
            logger.debug(ident)

    else:
        value: Any = await ctx.vm.eval(ctx.instruction.data)
        logger.debug(value)

    ctx.vm.current_task.ip += 1


@instruction_handler("log_multi")
async def instr_log_multi(ctx: InstructionContext) -> None:
    """Log a value per client."""
    assert isinstance(ctx.instruction.data, list)
    clients: list[SprintyClient] = ctx.vm._command_players(ctx.instruction.data[0])
    expr: Expression = ctx.instruction.data[1]

    for client in clients:
        string: str = await ctx.vm.eval(expr, client)
        logger.debug(f"{client.title} - {string}")

    ctx.vm.current_task.ip += 1


@instruction_handler("load_playstyle")
async def instr_load_playstyle(ctx: InstructionContext) -> None:
    """Spread a playstyle across clients."""
    logger.debug("Loading playstyle")

    # A playstyle may be written as a string, a constant or an expression, so resolve it before splitting it.
    playstyle: str = str(await ctx.vm._extract_data_info(ctx.instruction.data))
    delegated: dict[int, str] = delegate_combat_configs(playstyle, len(ctx.vm._clients))
    logger.debug(delegated)

    for index, client in enumerate(ctx.vm._clients):
        client.combat_config = delegated.get(index, default_config)

    ctx.vm.current_task.ip += 1


@instruction_handler("toggle_combat")
async def instr_toggle_combat(ctx: InstructionContext) -> None:
    """Turn auto combat on or off."""
    args: list[str] = ctx.instruction.data

    # Written bare, flips whatever combat is doing now.
    desired: bool | None = args[0].lower() == "on" if args else None

    if ctx.vm.on_toggle_combat is None:
        logger.warning("Ignoring togglecombat: this VM was not given a combat task to drive")
    else:
        await ctx.vm.on_toggle_combat(desired)

    ctx.vm.current_task.ip += 1


@instruction_handler("set_yaw")
async def instr_set_yaw(ctx: InstructionContext) -> None:
    """Point selected clients at a heading."""
    assert isinstance(ctx.instruction.data, list)
    selector: PlayerSelector = ctx.instruction.data[0]

    # A heading may be written as a number, a constant or an expression, so resolve it before writing memory.
    yaw: float = _as_number(await ctx.vm._extract_data_info(ctx.instruction.data[1]))

    clients: list[SprintyClient] = ctx.vm._command_players(selector)

    if clients:
        async with asyncio.TaskGroup() as tg:
            for client in clients:
                tg.create_task(client.body.write_yaw(yaw))

    ctx.vm.current_task.ip += 1


@instruction_handler("setdeck")
async def instr_setdeck(ctx: InstructionContext) -> None:
    """Apply a deck preset."""

    async def setdeck(client: SprintyClient) -> None:
        """Apply the preset to one client."""
        logger.debug(f"Setting {client.title}'s deck...")

        async with DeckBuilder(client) as deck_builder:
            await deck_builder.set_deck_preset(deck)

    assert isinstance(ctx.instruction.data, list)
    clients: list[SprintyClient] = ctx.vm._command_players(ctx.instruction.data[0])
    token: Any = await ctx.vm._extract_data_info(ctx.instruction.data[1])

    if not isinstance(token, str):
        raise VMError(f"Could not read the deck token: {token!r} is not text")

    if not token:
        raise VMError("Could not read the deck token: it is empty")

    coder: DeckEncoderDecoder = DeckEncoderDecoder(token=token)

    try:
        deck: dict = coder.decode()

    except ValueError as error:
        raise VMError(f"Could not read the deck token: {error}") from error

    async with asyncio.TaskGroup() as tg:
        for client in clients:
            tg.create_task(setdeck(client))

    logger.debug("Successfully set decks.")
    ctx.vm.current_task.ip += 1


@instruction_handler("getdeck")
async def instr_getdeck(ctx: InstructionContext) -> None:
    """Log each deck as a token."""

    async def getdeck(client: SprintyClient) -> None:
        """Log one deck as a token."""
        async with DeckBuilder(client) as deck_builder:
            deck: dict = await deck_builder.get_deck_preset()
            coder: DeckEncoderDecoder = DeckEncoderDecoder(deck=deck)
            token: str = coder.encode()
            logger.debug(f"{client.title}: --> {token} <--")

    assert isinstance(ctx.instruction.data, list)
    clients: list[SprintyClient] = ctx.vm._command_players(ctx.instruction.data[0])
    logger.debug("Reading deck...")

    async with asyncio.TaskGroup() as tg:
        for client in clients:
            tg.create_task(getdeck(client))

    ctx.vm.current_task.ip += 1
