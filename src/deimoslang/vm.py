"""Runs a compiled bot."""

import asyncio
import re
from collections.abc import Awaitable, Callable
from typing import Any

from loguru import logger
from wizwalker import XYZ, Client, Keycode
from wizwalker.extensions.scripting.utils import (
    _click_on_friend,
    _cycle_to_online_friends,
    _maybe_get_named_window,
)
from wizwalker.extensions.wizsprinter import SprintyClient  # ty: ignore[unresolved-import]
from wizwalker.extensions.wizsprinter.wiz_sprinter import upgrade_clients  # ty: ignore[unresolved-import]
from wizwalker.memory import DynamicWindow
from wizwalker.memory.memory_objects.quest_client_manager import QuestClientManager
from wizwalker.memory.memory_objects.quest_data import QuestData

from ..drop_logger import filter_drops, find_new_stuff, get_chat
from ..utils import (
    _cycle_friends_list,
    get_quest_name,
)
from .ast import (
    AddExpression,
    AndExpression,
    CommandExpression,
    CommandKind,
    ConstantCheckExpression,
    ConstantExpression,
    ConstantReferenceExpression,
    ContainsStringExpression,
    DivideExpression,
    EquivalentExpression,
    Eval,
    Expression,
    ExprKind,
    GreaterEqualExpression,
    GreaterExpression,
    IdentExpression,
    IndexAccessExpression,
    Instruction,
    InstructionKind,
    KeyExpression,
    ListExpression,
    ModuloExpression,
    MultiplyExpression,
    NumberExpression,
    OrExpression,
    PlayerSelector,
    RangeMaxExpression,
    RangeMinExpression,
    ReadVarExpr,
    SelectorGroup,
    StackLocExpression,
    StrFormatExpression,
    StringExpression,
    SubExpression,
    UnaryExpression,
    UnaryOp,
    VMError,
    XYZExpression,
    asks_any_player,
)
from .commands import HANDLERS as COMMAND_HANDLERS
from .commands import INSTRUCTION_HANDLERS, EvalContext, ExecContext, InstructionContext, StatContext
from .commands import PREDICATES as EXPR_PREDICATES
from .commands import STATS as EXPR_STATS
from .commands.handlers import Handler, InstructionHandler
from .commands.predicates import Predicate, StatEvaluator, StatValue
from .compiler import Compiler
from .lexer import LineInfo

MAX_STACK_DEPTH: int = 1024


class Task:
    """One running program."""

    def __init__(self) -> None:
        self.stack: list[Any] = []
        self.ip: int = 0
        self.running: bool = True


class Scheduler:
    """Round-robins between tasks."""

    def __init__(self) -> None:
        self.tasks: list[Task] = []
        self.current_task_index: int = 0

    def add_task(self, task: Task) -> None:
        """Add a task to the round-robin."""
        self.tasks.append(task)

    def remove_task(self, task: Task) -> None:
        """Remove a task from the round-robin."""
        self.tasks.remove(task)

    def get_current_task(self) -> Task:
        """The task whose turn it is."""
        return self.tasks[self.current_task_index]

    def switch_task(self) -> None:
        """Advance to the next task."""
        self.current_task_index = (self.current_task_index + 1) % len(self.tasks)


def _error_leaves(error: BaseException) -> list[BaseException]:
    """Every plain error inside a group."""
    if isinstance(error, BaseExceptionGroup):
        return [leaf for inner in error.exceptions for leaf in _error_leaves(inner)]

    return [error]


def _vm_error_from_group(error_group: BaseExceptionGroup) -> VMError | None:
    """One VMError for a group's many."""
    leaves: list[BaseException] = _error_leaves(error_group)
    vm_errors: list[VMError] = [error for error in leaves if isinstance(error, VMError)]

    # Anything else is a Deimos fault. Group says more.
    if not vm_errors or len(vm_errors) != len(leaves):
        return None

    # Mass fails the same way per client.
    messages: list[str] = []

    for error in vm_errors:
        if error.message not in messages:
            messages.append(error.message)

    if len(messages) == 1:
        return vm_errors[0]

    return VMError("; ".join(messages))


def _as_number(value: Any, expression: Expression) -> float:
    """A calculation side, as a number."""
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise VMError(f"Expected a number in {expression}, got {value!r}")

    return value


def _as_bool(value: Any) -> Any:
    """`true` or `false` as a boolean."""
    if isinstance(value, str) and value.lower() in ("true", "false"):
        return value.lower() == "true"

    return value


class UntilInfo:
    """A running `until` and its exit."""

    def __init__(
        self, expr: Expression, id: int, exit_point: int, stack_size: int, line_info: LineInfo | None = None
    ) -> None:
        self.expr: Expression = expr
        self.id: int = id
        self.exit_point: int = exit_point
        self.stack_size: int = stack_size
        self.line_info: LineInfo | None = line_info


class VM:
    """Runs a compiled bot on clients."""

    def __init__(self, clients: list[Client]) -> None:
        self._clients: list[SprintyClient] = upgrade_clients(clients)
        self.program: list[Instruction] = []
        self.running: bool = False
        self.killed: bool = False
        self._scheduler: Scheduler = Scheduler()
        self._scheduler.add_task(Task())
        self.current_task: Task = self._scheduler.get_current_task()
        self._any_player_client: list[SprintyClient] = []
        self.on_toggle_combat: Callable[[bool | None], Awaitable[None]] | None = None
        self._timers: dict[str, float] = {}
        self._counters: dict[str, int] = {}
        self.logged_data: dict[str, dict[str, str | None]] = {"goal": {}, "quest": {}, "zone": {}}

        # True and False are defined up front so `$True` and `$False` resolve without being declared.
        self._constants: dict[str, Any] = {
            "True": True,
            "False": False,
        }

        self._until_infos: list[UntilInfo] = []

        # Set while an until is only polling.
        self._polling_untils: bool = False

    def reset(self) -> None:
        """Clear every bit of running state."""
        self.program = []

        self._scheduler.tasks.clear()
        self._scheduler.current_task_index = 0
        self._scheduler.add_task(Task())
        self.current_task = self._scheduler.get_current_task()
        self._until_infos = []
        self._polling_untils = False
        self._timers = {}
        self._counters = {}
        self._any_player_client = []
        self._constants = {
            "True": True,
            "False": False,
        }
        self.logged_data = {"goal": {}, "quest": {}, "zone": {}}

    def stop(self) -> None:
        """Stop stepping."""
        self.running = False

    def kill(self) -> None:
        """Stop stepping and do not restart."""
        self.stop()
        self.killed = True

    async def define_constant(self, name: str, value: Any) -> None:
        """Set a constant to whatever was written, without reading anything into it."""
        self._constants[name] = value

    def load_from_text(self, code: str, filename: str | None = None) -> None:
        """Compile source text into a program."""
        compiler: Compiler = Compiler.from_text(code, filename=filename)
        self.program = compiler.compile()

    def player_by_num(self, num: int) -> SprintyClient | None:
        """The client `p<n>` names."""
        if not 1 <= num <= len(self._clients):
            return None

        return self._clients[num - 1]

    async def select_friend_from_list(self, client: SprintyClient, name: str) -> bool:
        """Pick a friend from the list."""
        async with client.mouse_handler:
            try:
                friends_window: DynamicWindow = await _maybe_get_named_window(
                    client.root_window, "NewFriendsListWindow"
                )

                if not await friends_window.is_visible():
                    raise ValueError("Friends list not visible")

            except ValueError:
                friend_button: DynamicWindow = await _maybe_get_named_window(client.root_window, "btnFriends")
                await client.mouse_handler.click_window(friend_button)
                await asyncio.sleep(0.4)
                friends_window = await _maybe_get_named_window(client.root_window, "NewFriendsListWindow")

            await _cycle_to_online_friends(client, friends_window)

            friends_list_window: DynamicWindow = await _maybe_get_named_window(friends_window, "listFriends")

            right_button: DynamicWindow = await _maybe_get_named_window(friends_window, "btnArrowDown")
            page_number: DynamicWindow = await _maybe_get_named_window(friends_window, "PageNumber")

            page_number_text: str = await page_number.maybe_text()
            current_page, _ = map(
                int,
                page_number_text.replace("<center>", "").replace("</center>", "").replace(" ", "").split("/"),
            )

            friend, friend_index = await _cycle_friends_list(
                client,
                right_button,
                friends_list_window,
                None,
                None,
                name,
                current_page,
            )

            if friend is None:
                logger.error(f"Could not find friend with name {name}")
                return False

            await _click_on_friend(client, friends_list_window, friend_index)

            return True

    def _select_players(self, selector: PlayerSelector) -> list[SprintyClient]:
        """The clients a selector resolves to."""
        if selector.mass:
            return self._clients

        # anyplayer decides its clients while the condition runs. None known yet.
        elif selector.any_player:
            return []

        elif selector.same_any:
            return self._any_player_client

        else:
            result: list[SprintyClient] = []

            if selector.inverted:
                for index in range(len(self._clients)):
                    if index + 1 in selector.player_nums:
                        continue

                    result.append(self._clients[index])

            else:
                for num in selector.player_nums:
                    client: SprintyClient | None = self.player_by_num(num)
                    if client is not None:
                        result.append(client)

            return result

    def _command_players(self, selector: PlayerSelector) -> list[SprintyClient]:
        """The clients a command runs on."""
        if selector.any_player:
            return self._any_player_client

        return self._select_players(selector)

    async def _fetch_tracked_quest(self, client: SprintyClient) -> QuestData:
        """The quest the client is tracking."""
        tracked_id: int = await client.quest_id()
        qm: QuestClientManager = await client.quest_manager()

        for quest_id, quest in (await qm.quest_data()).items():
            if quest_id == tracked_id:
                return quest

        raise VMError(f"Unable to fetch the currently tracked quest for client with title {client.title}")

    async def _fetch_tracked_quest_text(self, client: SprintyClient) -> str:
        """Name of the tracked quest, lowercased."""
        quest: QuestData = await self._fetch_tracked_quest(client)
        name_key: str = await quest.name_lang_key()

        if name_key == "Quest Finder":
            name: str = name_key
        else:
            name: str = await client.cache_handler.get_langcode_name(name_key)

        return name.lower().strip()

    async def _fetch_quests(self, client: SprintyClient) -> list[tuple[int, QuestData]]:
        """Every quest in the client's log."""
        result: list[tuple[int, QuestData]] = []
        qm: QuestClientManager = await client.quest_manager()

        for quest_id, quest in (await qm.quest_data()).items():
            result.append((quest_id, quest))

        return result

    async def _fetch_quest_text(self, client: SprintyClient, quest: QuestData) -> str:
        """Name of a quest, lowercased."""
        name_key: str = await quest.name_lang_key()

        if name_key == "Quest Finder":
            name: str = name_key
        else:
            name: str = await client.cache_handler.get_langcode_name(name_key)

        return name.lower().strip()

    async def _fetch_tracked_goal_text(self, client: SprintyClient) -> str:
        """Text of the tracked goal, lowercased."""
        goal_txt: str = await get_quest_name(client)
        goal_txt = re.sub(r"<[^>]*>", "", goal_txt)

        if "(" in goal_txt:
            goal_txt = goal_txt[: goal_txt.find("(")]

        return goal_txt.lower().strip()

    async def _check_drops(self, client: SprintyClient, item_names: str | list[str], consume: bool = True) -> bool:
        """Whether a named item dropped."""
        chat_text: str = await get_chat(client)
        if not chat_text:
            return False

        drops: list[str] = filter_drops(chat_text.split("\n"))

        if not hasattr(client, "_last_chat_state"):
            client._last_chat_state = ""

        # Only lines since the last look. One drop would keep answering true.
        new_chat_content: str = find_new_stuff(client._last_chat_state, "\n".join(drops))

        # A poll only looks. It leaves the drop for the next ask.
        if consume:
            client._last_chat_state = "\n".join(drops)

        if not new_chat_content:
            return False

        new_drops: list[str] = new_chat_content.split("\n")
        wanted: list[str] = [item_names] if isinstance(item_names, str) else item_names

        for drop in new_drops:
            if not drop:
                continue

            for item_name in wanted:
                if item_name.lower() in drop.lower():
                    logger.debug(f"Found new dropped item matching '{item_name}': {drop}")
                    return True

        return False

    def _named_constant(self, text: str) -> Any:
        """What a `$name` stands for, or the text itself when nothing was declared under that name."""
        const_name: str = text[1:]

        if const_name in self._constants:
            return self._constants[const_name]

        logger.warning(f"Constant '{const_name}' not found")
        return text

    async def _extract_data_info(self, data: Any) -> Any:
        """Resolve an argument to a value."""
        # A number is already the value. Evaluating would only fail.
        if isinstance(data, (int, float)):
            return data

        # A bare string is either a $constant reference or the value itself.
        if isinstance(data, str):
            return self._named_constant(data) if data.startswith("$") else data

        elif isinstance(data, StringExpression):
            return self._named_constant(data.string) if data.string.startswith("$") else data.string

        elif isinstance(data, IdentExpression):
            ident: str = data.ident

            if ident.startswith("$"):
                return self._named_constant(ident)

            elif ident in self._constants:
                return self._constants[ident]

            # Not a constant, so the identifier is treated as something to evaluate.
            else:
                try:
                    return await self.eval(data)
                except Exception as error:  # noqa: BLE001
                    logger.error(f"Failed to evaluate identifier {ident}: {error}")
                    return ident

        # A list of strings is a zone path. The game spells it with slashes.
        elif isinstance(data, list) and all(isinstance(item, str) for item in data):
            return "/".join(data)

        else:
            try:
                return await self.eval(data)
            except Exception as error:  # noqa: BLE001
                logger.error(f"Failed to extract zone name: {error}")
                return str(data)

    async def _eval_command_expression(self, expression: CommandExpression) -> bool:
        """Evaluate a condition on the clients."""
        assert expression.command.kind == CommandKind.expr
        assert type(expression.command.data) is list

        if not expression.command.data:
            return False

        assert type(expression.command.data[0]) is ExprKind

        selector: PlayerSelector | None = expression.command.player_selector
        assert selector is not None
        clients: list[SprintyClient] = self._select_players(selector)

        if not clients and not selector.any_player:
            return False

        kind: ExprKind = expression.command.data[0]
        check: Predicate | None = EXPR_PREDICATES.get(kind)

        if check is None:
            raise VMError(f"Unimplemented expression: {expression}")

        if selector.negated:
            return await self._eval_negated_check(check, selector, expression)

        return await check(
            EvalContext(
                vm=self,
                clients=clients,
                selector=selector,
                expression=expression,
                polling=self._polling_untils,
            )
        )

    async def _eval_negated_check(
        self, check: Predicate, selector: PlayerSelector, expression: CommandExpression
    ) -> bool:
        """Ask a check per client."""
        asked: list[SprintyClient] = self._clients if selector.any_player else self._select_players(selector)
        alone: PlayerSelector = PlayerSelector()
        failed: list[SprintyClient] = []

        for client in asked:
            asking: EvalContext = EvalContext(
                vm=self, clients=[client], selector=alone, expression=expression, polling=self._polling_untils
            )

            if not await check(asking):
                failed.append(client)

        if selector.any_player:
            self._any_player_client = failed
            return len(failed) > 0

        return len(asked) > 0 and len(failed) == len(asked)

    async def eval(self, expression: Expression, client: Client | None = None) -> Any:
        """Evaluate an expression, maybe per client."""
        match expression:
            case IdentExpression():
                if expression.ident in self._constants:
                    return self._constants[expression.ident]

                return expression.ident

            case ConstantReferenceExpression():
                if expression.name in self._constants:
                    return self._constants[expression.name]

                raise VMError(f"Unknown constant: ${expression.name}")

            case ConstantExpression():
                # `true` and `false` are spelled as text, but a constant declared from one holds a boolean.
                return _as_bool(await self.eval(expression.value, client))

            case ConstantCheckExpression():
                constant_name: str = expression.name
                expected_value: Any = await self.eval(expression.value)

                if constant_name in self._constants:
                    actual_value: Any = self._constants[constant_name]

                    if isinstance(expected_value, bool) != isinstance(actual_value, bool):
                        return _as_bool(actual_value) == _as_bool(expected_value)

                    return actual_value == expected_value

                return False

            case RangeMinExpression():
                return await self._eval_range_end(expression.range_expr, 0, client)

            case RangeMaxExpression():
                return await self._eval_range_end(expression.range_expr, 1, client)

            case IndexAccessExpression():
                container: Any = await self.eval(expression.expr, client)
                index: Any = await self.eval(expression.index, client)

                if isinstance(container, list) and isinstance(index, (int, float)):
                    index_int: int = int(index)

                    # A missing number reads as 0. The condition fails instead of raising.
                    if 0 <= index_int < len(container):
                        return container[index_int]
                    else:
                        return 0.0

                else:
                    return 0.0

            case AndExpression():
                matched: list[SprintyClient] | None = None

                for expr in expression.expressions:
                    if not await self.eval(expr, client):
                        return False

                    if not asks_any_player(expr):
                        continue

                    # Keep only the clients that answered every `any` so far.
                    if matched is not None:
                        self._any_player_client = [player for player in self._any_player_client if player in matched]

                        if not self._any_player_client:
                            return False

                    matched = self._any_player_client

                return True

            case OrExpression():
                # Stopping at the first side that holds loses the clients the others matched.
                if not any(asks_any_player(expr) for expr in expression.expressions):
                    for expr in expression.expressions:
                        if await self.eval(expr, client):
                            return True

                    return False

                gathered: list[SprintyClient] = []
                held: bool = False

                for expr in expression.expressions:
                    # Once a side holds, only another `any` adds clients. Nothing else to ask.
                    if held and not asks_any_player(expr):
                        continue

                    if not await self.eval(expr, client):
                        continue

                    held = True

                    # Every `any` that held adds its clients. A later command aims at all.
                    if asks_any_player(expr):
                        gathered.extend(player for player in self._any_player_client if player not in gathered)

                self._any_player_client = gathered
                return held

            case CommandExpression():
                return await self._eval_command_expression(expression)

            case NumberExpression():
                return expression.number

            case XYZExpression():
                return XYZ(
                    await self.eval(expression.x, client),
                    await self.eval(expression.y, client),
                    await self.eval(expression.z, client),
                )

            case UnaryExpression():
                match expression.operator:
                    case UnaryOp.negate:
                        result: Any = await self.eval(expression.expr, client)
                        return -result

                    case UnaryOp.not_:
                        expr_result: Any = await self.eval(expression.expr, client)

                        # Negating flips the match. Later commands aim at the rest.
                        if asks_any_player(expression.expr):
                            current_matches: list[SprintyClient] = self._any_player_client
                            self._any_player_client = [
                                player for player in self._clients if player not in current_matches
                            ]

                        return not expr_result

                    case _:
                        raise VMError(f"Unimplemented unary expression: {expression}")

            case StringExpression():
                return expression.string

            case StrFormatExpression():
                format_str: str = expression.format_str
                values: list[Any] = []

                for value_expr in expression.values:
                    result: Any = await self.eval(value_expr, client)
                    values.append(result)

                return format_str % tuple(values)

            case KeyExpression():
                key: str = expression.key

                # A name that is not a key itself may be a constant naming one, read only where a key is wanted.
                if key not in Keycode.__members__:
                    held: Any = self._constants.get(key.removeprefix("$"))

                    if isinstance(held, str) and held in Keycode.__members__:
                        return Keycode[held]

                    raise VMError(f"Unknown key: {key}")

                return Keycode[expression.key]

            case EquivalentExpression():
                left: Any = await self.eval(expression.lhs, client)
                right: Any = await self.eval(expression.rhs, client)

                # Reading a window hands back a list. A bare comparison means its first.
                if isinstance(left, list) and len(left) > 0:
                    left = left[0]

                if isinstance(right, list) and len(right) > 0:
                    right = right[0]

                return left == right

            case AddExpression() | SubExpression() | MultiplyExpression():
                left: float = _as_number(await self.eval(expression.lhs, client), expression)
                right: float = _as_number(await self.eval(expression.rhs, client), expression)

                if isinstance(expression, AddExpression):
                    return left + right

                if isinstance(expression, SubExpression):
                    return left - right

                return left * right

            case DivideExpression() | ModuloExpression():
                left: float = _as_number(await self.eval(expression.lhs, client), expression)
                right: float = _as_number(await self.eval(expression.rhs, client), expression)

                if right == 0:
                    act: str = "Modulo" if isinstance(expression, ModuloExpression) else "Division"
                    raise VMError(f"{act} by zero in {expression}")

                if isinstance(expression, ModuloExpression):
                    return left % right

                return left / right

            case GreaterExpression() | GreaterEqualExpression():
                left: Any = await self.eval(expression.lhs, client)
                right: Any = await self.eval(expression.rhs, client)

                if isinstance(left, list) and len(left) > 0:
                    left = left[0]

                if isinstance(right, list) and len(right) > 0:
                    right = right[0]

                if isinstance(expression, GreaterEqualExpression):
                    return left >= right

                return left > right

            case Eval():
                return await self._eval_expression(expression, client)

            case SelectorGroup():
                players: list[SprintyClient] = self._select_players(expression.players)
                expr: Expression = expression.expr

                # anyplayer holds when one client matches, and remembers which did.
                if expression.players.any_player:
                    self._any_player_client = []
                    found_any: bool = False

                    for anyplayer in self._clients:
                        result = await self.eval(expr, anyplayer)

                        # Negated counts the failures instead.
                        if bool(result) != expression.players.negated:
                            self._any_player_client.append(anyplayer)
                            found_any = True

                    return found_any

                else:
                    # Nobody running, so nobody to answer either way.
                    if not players:
                        return False

                    for player in players:
                        if bool(await self.eval(expr, player)) == expression.players.negated:
                            return False

                    return True

            case ReadVarExpr():
                loc: Any = await self.eval(expression.loc)
                assert isinstance(loc, int)
                return self.current_task.stack[loc]

            case StackLocExpression():
                return expression.offset

            case ListExpression():
                result: list[Any] = []

                for item in expression.items:
                    evaluated_item: Any = await self.eval(item, client)

                    if isinstance(evaluated_item, list):
                        result.extend(evaluated_item)
                    else:
                        result.append(evaluated_item)

                return result

            case ContainsStringExpression():
                lhs: Any = await self.eval(expression.lhs, client)
                rhs: Any = await self.eval(expression.rhs, client)

                if isinstance(rhs, list):
                    return any(item in lhs for item in rhs)

                return rhs in lhs

            case _:
                raise VMError(f"Unimplemented expression type: {expression}")

    async def _eval_range_end(self, range_expr: Expression, index: int, client: Client | None) -> float:
        """One end of a written range."""
        range_value: Any = await self.eval(range_expr, client)

        if isinstance(range_value, str):
            parts: list[str] = range_value.split("-")

            if len(parts) != 2:
                raise VMError(f"Invalid range format: {range_value}. Expected format like '1-100'")

            try:
                return float(parts[index])
            except ValueError:
                raise VMError(f"Invalid range format: {range_value}. Expected format like '1-100'") from None

        if isinstance(range_value, list) and len(range_value) == 2:
            try:
                return float(range_value[index])
            except (TypeError, ValueError):
                raise VMError(f"Invalid range end: {range_value[index]}. Expected a number") from None

        raise VMError(f"Range must be a string like '1-100' or a two item list, got {range_value}")

    async def _eval_expression(self, eval: Eval, client: Client | None) -> StatValue:
        """Read one value off a client."""
        evaluator: StatEvaluator | None = EXPR_STATS.get(eval.kind)

        if evaluator is None:
            raise VMError(f"Unimplemented eval kind: {eval.kind}")

        return await evaluator(StatContext(vm=self, client=client, eval=eval))

    async def exec_deimos_call(self, instruction: Instruction) -> None:
        """Run one command on its clients."""
        assert instruction.kind == InstructionKind.deimos_call
        assert isinstance(instruction.data, list)
        selector: PlayerSelector = instruction.data[0]

        clients: list[SprintyClient] = self._command_players(selector)

        if not clients:
            return

        async def eval_arg(arg: Any, client: Client | None) -> Any:
            """Resolve one argument to a value."""
            if isinstance(arg, Expression):
                if isinstance(arg, IdentExpression):
                    # A $name and a bare name both refer to the same constant.
                    constant_name: str = arg.ident.removeprefix("$")

                    if constant_name in self._constants:
                        return self._constants[constant_name]

                    return arg.ident

                return await self.eval(arg, client)

            elif isinstance(arg, str) and arg.startswith("$"):
                constant_name: str = arg[1:]
                if constant_name in self._constants:
                    return self._constants[constant_name]
                else:
                    logger.error(f"Undefined constant: {arg}")
                    return arg

            return arg

        command_handler: Handler | None = COMMAND_HANDLERS.get(instruction.data[1])
        if command_handler is None:
            raise VMError(f"Unimplemented deimos call: {instruction}")

        try:
            await command_handler(
                ExecContext(
                    vm=self,
                    clients=clients,
                    args=instruction.data[2],
                    eval_arg=eval_arg,
                    instruction=instruction,
                )
            )

        # Handlers drive clients through a task group, which buries the reason.
        except BaseExceptionGroup as error_group:
            script_error: VMError | None = _vm_error_from_group(error_group)

            if script_error is None:
                raise

            raise script_error from None

    async def _process_untils(self) -> None:
        """Jump out if an `until` holds."""
        # Polling must not rewrite what sameany and anyplayer read.
        watching: list[SprintyClient] = self._any_player_client
        self._polling_untils = True

        try:
            # Innermost regions first. The tightest until wins.
            for index in range(len(self._until_infos) - 1, -1, -1):
                info: UntilInfo = self._until_infos[index]

                try:
                    # An until that ends hands its own clients on.
                    if await self.eval(info.expr):
                        self.current_task.ip = info.exit_point
                        return

                except VMError as error:
                    error.locate(info.line_info)
                    logger.warning(f"Leaving an until region because its condition could not be read: {error.brief()}")
                    self.current_task.ip = info.exit_point
                    break

        finally:
            self._polling_untils = False

        self._any_player_client = watching

    async def step(self) -> None:
        """Execute one instruction."""
        if not self.running:
            return

        await asyncio.sleep(0)
        self.current_task = self._scheduler.get_current_task()
        await self._process_untils()

        if not self.current_task.running:
            self._scheduler.switch_task()
            return

        instruction: Instruction = self.program[self.current_task.ip]
        instr_handler: InstructionHandler | None = INSTRUCTION_HANDLERS.get(instruction.kind.name)

        # Every handler moves the pointer itself. Jumps decide where it lands.
        try:
            if instr_handler is not None:
                await instr_handler(InstructionContext(vm=self, instruction=instruction))
            else:
                await self._exec_instruction(instruction)

        # Nothing deeper knows the line.
        except VMError as error:
            error.locate(instruction.line_info)
            raise

        if self.current_task.ip >= len(self.program):
            self.current_task.running = False

        # Stops once nothing is left to run, or something asked.
        if not any(task.running for task in self._scheduler.tasks) or not self.running:
            self.stop()
        else:
            self._scheduler.switch_task()
            await asyncio.sleep(0)

    async def _exec_instruction(self, instruction: Instruction) -> None:
        """Run an instruction with no handler."""
        match instruction.kind:
            case InstructionKind.restart_bot:
                self.reset()
                self.current_task.ip = 0
                logger.debug("Bot Restarted")

            case InstructionKind.declare_constant:
                name, expr = instruction.data
                value: Any = await self.eval(expr)
                await self.define_constant(name, value)
                self.current_task.ip += 1

            case InstructionKind.start_timer:
                assert isinstance(instruction.data, str), "Timer name must be a string"
                timer_name: str = instruction.data
                self._timers[timer_name] = asyncio.get_event_loop().time()
                logger.debug(f"Timer '{timer_name}' started")
                self.current_task.ip += 1

            case InstructionKind.reset_timer:
                assert isinstance(instruction.data, str), "Timer name must be a string"
                timer_name = instruction.data

                if timer_name not in self._timers:
                    raise VMError(f"Timer '{timer_name}' was never started")

                self._timers[timer_name] = asyncio.get_event_loop().time()
                logger.debug(f"Timer '{timer_name}' reset")
                self.current_task.ip += 1

            case InstructionKind.end_timer:
                assert isinstance(instruction.data, str), "Timer name must be a string"
                timer_name = instruction.data

                if timer_name not in self._timers:
                    raise VMError(f"Timer '{timer_name}' was never started")

                elapsed_seconds: float = asyncio.get_event_loop().time() - self._timers[timer_name]
                hours, remainder = divmod(int(elapsed_seconds), 3600)
                minutes, seconds = divmod(remainder, 60)

                time_str: str = f"{hours:02}:{minutes:02}:{seconds:02}"

                logger.debug(f"Timer '{timer_name}' ended - Elapsed time: {time_str}")
                del self._timers[timer_name]
                self.current_task.ip += 1

            case InstructionKind.start_counter:
                assert isinstance(instruction.data, str), "Counter name must be a string"
                counter_name: str = instruction.data
                self._counters[counter_name] = 0
                logger.debug(f"Counter '{counter_name}' started")
                self.current_task.ip += 1

            case InstructionKind.reset_counter:
                assert isinstance(instruction.data, str), "Counter name must be a string"
                counter_name = instruction.data

                if counter_name not in self._counters:
                    raise VMError(f"Counter '{counter_name}' was never started")

                self._counters[counter_name] = 0
                logger.debug(f"Counter '{counter_name}' reset")
                self.current_task.ip += 1

            case InstructionKind.end_counter:
                assert isinstance(instruction.data, str), "Counter name must be a string"
                counter_name = instruction.data

                if counter_name not in self._counters:
                    raise VMError(f"Counter '{counter_name}' was never started")

                logger.debug(f"Counter '{counter_name}' ended - {self._counters[counter_name]}")
                del self._counters[counter_name]
                self.current_task.ip += 1

            case InstructionKind.change_counter:
                assert isinstance(instruction.data, list)
                counter_name, delta = instruction.data

                # Changing an unstarted counter is a typo, not a new counter.
                if counter_name not in self._counters:
                    raise VMError(f"Counter '{counter_name}' was never started")

                self._counters[counter_name] += delta
                self.current_task.ip += 1

            case InstructionKind.jump:
                assert isinstance(instruction.data, int)
                self.current_task.ip += instruction.data

            case InstructionKind.jump_if:
                assert isinstance(instruction.data, list)

                try:
                    condition_met: Any = await self.eval(instruction.data[0])
                    if condition_met:
                        self.current_task.ip += instruction.data[1]
                    else:
                        self.current_task.ip += 1

                # An unreadable condition takes the forward branch. The bot never stalls.
                except VMError as error:
                    error.locate(instruction.line_info)
                    logger.warning(f"Could not read a condition, taking the forward branch: {error.brief()}")

                    if instruction.data[1] > 1:
                        self.current_task.ip += instruction.data[1]
                    else:
                        self.current_task.ip += 1

            case InstructionKind.jump_ifn:
                assert isinstance(instruction.data, list)

                try:
                    condition_met = await self.eval(instruction.data[0])
                    if condition_met:
                        self.current_task.ip += 1
                    else:
                        self.current_task.ip += instruction.data[1]

                except VMError as error:
                    error.locate(instruction.line_info)
                    logger.warning(f"Could not read a condition, entering the loop body: {error.brief()}")
                    self.current_task.ip += 1

            # The stack doubles as the return stack. A call pushes where to come back.
            case InstructionKind.call:
                assert isinstance(instruction.data, int)

                if len(self.current_task.stack) >= MAX_STACK_DEPTH:
                    raise VMError(
                        f"Nested more than {MAX_STACK_DEPTH} deep without returning. "
                        "A block that calls itself never stops"
                    )

                self.current_task.stack.append(self.current_task.ip + 1)
                self.current_task.ip += instruction.data

            case InstructionKind.ret:
                self.current_task.ip = self.current_task.stack.pop()

            case InstructionKind.enter_until:
                assert isinstance(instruction.data, list)
                self._until_infos.append(
                    UntilInfo(
                        expr=instruction.data[0],
                        id=instruction.data[1],
                        exit_point=self.current_task.ip + instruction.data[2],
                        stack_size=len(self.current_task.stack),
                        line_info=instruction.line_info,
                    )
                )
                self.current_task.ip += 1

            case InstructionKind.exit_until:
                for index in range(len(self._until_infos) - 1, -1, -1):
                    info: UntilInfo = self._until_infos[index]

                    # Jumping out skips the pops inside. Cut the stack back by hand.
                    if info.id == instruction.data:
                        self._until_infos = self._until_infos[:index]
                        self.current_task.stack = self.current_task.stack[: info.stack_size]
                        break

                self.current_task.ip += 1

            case InstructionKind.label | InstructionKind.nop:
                self.current_task.ip += 1

            case InstructionKind.push_stack:
                self.current_task.stack.append(None)
                self.current_task.ip += 1

            case InstructionKind.write_stack:
                assert instruction.data is not None
                offset, expr = instruction.data
                self.current_task.stack[offset] = await self.eval(expr)
                self.current_task.ip += 1

            case InstructionKind.pop_stack:
                self.current_task.stack.pop()
                self.current_task.ip += 1

            case InstructionKind.deimos_call:
                await self.exec_deimos_call(instruction)
                self.current_task.ip += 1

            case _:
                raise VMError(f"Unimplemented instruction: {instruction}")

    async def run(self) -> None:
        """Step until the program stops."""
        self.running = True

        while self.running:
            await self.step()
