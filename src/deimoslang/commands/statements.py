"""How each statement command parses."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, replace
from typing import TYPE_CHECKING, Any

from wizwalker import Keycode

from ..ast import (
    ClickKind,
    CommandKind,
    ConstantReferenceExpression,
    CursorKind,
    Eval,
    EvalKind,
    Expression,
    IdentExpression,
    KeyExpression,
    ListExpression,
    LogKind,
    NumberExpression,
    StrFormatExpression,
    StringExpression,
    TeleportKind,
    WaitforKind,
    XYZExpression,
    describe_expression,
)
from ..tokens import TokenKind

if TYPE_CHECKING:
    from ..lexer import Token
    from ..parser import Parser

type ArgShape = Callable[["Parser"], list[Any]]

type Discriminator = TeleportKind | CursorKind | ClickKind | WaitforKind


@dataclass(frozen=True)
class CommandSpec:
    """How one statement command parses."""

    token: TokenKind
    kind: CommandKind
    parse_args: ArgShape
    every_client: bool = False


# Filled by the command calls below.
REGISTRY: dict[TokenKind, CommandSpec] = {}


def command(token: TokenKind, kind: CommandKind, parse_args: ArgShape, every_client: bool = False) -> CommandSpec:
    """Register one statement command."""
    if token in REGISTRY:
        raise ValueError(f"{token.name} is already registered to {REGISTRY[token].kind.name}")

    spec: CommandSpec = CommandSpec(token=token, kind=kind, parse_args=parse_args, every_client=every_client)
    REGISTRY[token] = spec
    return spec


def nullary(_parser: Parser) -> list[Any]:
    """Parse no arguments."""
    return []


# Written out in full. No command resolves them into something else.
_LITERALS: tuple[type[Expression], ...] = (StringExpression, NumberExpression, ListExpression, XYZExpression)


def _reject_literal(parser: Parser, expr: Expression, wanted: str, allowed: type[Expression] | None = None) -> None:
    """Refuse a literal the command can never use, since otherwise only the VM would say so."""
    if allowed is not None and isinstance(expr, allowed):
        return

    if isinstance(expr, _LITERALS):
        parser.err(parser.tokens[parser.pos - 1], f"Expected {wanted}, got {describe_expression(expr)}")


def _duration(parser: Parser) -> Expression:
    """Parse a wait time."""
    expr: Expression = parser.parse_expression()
    _reject_literal(parser, expr, "a number of seconds", NumberExpression)
    return expr


def duration(parser: Parser) -> list[Any]:
    """Parse how long to wait."""
    return [_duration(parser)]


def _looks_like_xyz(parser: Parser) -> bool:
    """Whether an `XYZ(...)` starts next."""
    return parser.tokens[parser.pos].kind == TokenKind.keyword_xyz


def _starts_window_path(parser: Parser, pos: int) -> bool:
    """Whether a [bracketed] path or a $constant naming one begins at `pos`."""
    if pos >= len(parser.tokens):
        return False

    token: Token = parser.tokens[pos]
    return token.kind == TokenKind.square_open or (token.kind == TokenKind.identifier and token.literal.startswith("$"))


def _position(parser: Parser) -> Expression:
    """Parse a position."""
    if _looks_like_xyz(parser):
        return parser.parse_xyz()

    expr: Expression = parser.parse_expression()
    _reject_literal(parser, expr, "a position")
    return expr


def xyz_or_expression(parser: Parser) -> list[Any]:
    """Parse a position argument."""
    return [_position(parser)]


def prefixed_xyz_or_expression(prefix: Discriminator) -> ArgShape:
    """Shape for a tagged position."""

    def shape(parser: Parser) -> list[Any]:
        """Parse the position after the tag."""
        return [prefix, _position(parser)]

    return shape


def prefixed_window_or_expression(prefix: Discriminator) -> ArgShape:
    """Shape for a tagged window path."""

    def shape(parser: Parser) -> list[Any]:
        """Parse the path after the tag."""
        if _starts_window_path(parser, parser.pos):
            return [prefix, parser.parse_window_path()]

        expr: Expression = parser.parse_expression()
        _reject_literal(parser, expr, "a window path")
        return [prefix, expr]

    return shape


def waitfor(prefix: WaitforKind) -> ArgShape:
    """Shape for a plain waitfor."""

    def shape(parser: Parser) -> list[Any]:
        """Parse the optional `completion` keyword."""
        return [prefix, parser.parse_completion_optional()]

    return shape


def waitfor_window(prefix: WaitforKind) -> ArgShape:
    """Shape for waitfor on a window."""

    def shape(parser: Parser) -> list[Any]:
        """Parse the path and `completion`."""
        return [prefix, parser.parse_window_path(), parser.parse_completion_optional()]

    return shape


def flag_presence(token: TokenKind) -> ArgShape:
    """Shape for an optional keyword."""

    def shape(parser: Parser) -> list[Any]:
        """Record whether the keyword was there."""
        return [parser.consume_optional(token) is not None]

    return shape


def optional_choice(*tokens: TokenKind) -> ArgShape:
    """Shape for an optional choice."""
    choices: list[TokenKind] = list(tokens)

    def shape(parser: Parser) -> list[Any]:
        """Keep the chosen keyword's literal."""
        if parser.tokens[parser.pos].kind in choices:
            return [parser.expect_consume_any(choices).literal]

        return []

    return shape


def string_literal(parser: Parser) -> list[Any]:
    """Parse a required quoted string."""
    return [parser.expect_consume(TokenKind.string).value]


def first_of(
    primary: TokenKind,
    extract: Callable[[Parser], Any],
    wanted: str,
    allowed: type[Expression] | None = None,
) -> ArgShape:
    """Shape accepting several forms."""

    def shape(parser: Parser) -> list[Any]:
        """Parse whichever form is next."""
        if parser.tokens[parser.pos].kind == primary:
            return [extract(parser)]

        if parser.tokens[parser.pos].kind == TokenKind.identifier:
            return [IdentExpression(parser.expect_consume(TokenKind.identifier).literal)]

        expr: Expression = parser.parse_expression()
        _reject_literal(parser, expr, wanted, allowed)
        return [expr]

    return shape


def _scalar(parser: Parser) -> Expression | None:
    """Parse a number or a constant name, or nothing if neither is next."""
    kind: TokenKind = parser.tokens[parser.pos].kind

    if kind == TokenKind.number:
        return NumberExpression(parser.expect_consume(TokenKind.number).value)

    if kind == TokenKind.identifier:
        return IdentExpression(parser.expect_consume(TokenKind.identifier).literal)

    return None


def optional_pair(parser: Parser) -> list[Any]:
    """Parse an optional `x, y` pair."""
    first: Expression | None = _scalar(parser)

    if first is None:
        return []

    parser.skip_comma()
    second: Expression | None = _scalar(parser)

    if second is None:
        parser.err(parser.tokens[parser.pos], "Expected a number or identifier as the second argument")

    return [first, second]


def xy_pair(prefix: Discriminator) -> ArgShape:
    """Shape for a required `x, y` pair."""

    def shape(parser: Parser) -> list[Any]:
        """Parse both coordinates, tagged."""
        first: Expression | None = _scalar(parser)

        if first is None:
            parser.err(parser.tokens[parser.pos], "Expected a number or identifier as the first argument")

        parser.skip_comma()
        second: Expression | None = _scalar(parser)

        if second is None:
            parser.err(parser.tokens[parser.pos], "Expected a number or identifier as the second argument")

        return [prefix, first, second]

    return shape


def key_with_optional_expression(parser: Parser) -> list[Any]:
    """Parse a key and an optional duration."""
    key: KeyExpression = parser.parse_key()

    # The game names its keys, so a wrong one is worth catching here rather than mid-run. A $name says
    # the key comes from a constant, which only the VM can look up.
    if not key.key.startswith("$") and key.key not in Keycode.__members__:
        parser.err(parser.tokens[parser.pos - 1], f"Unknown key: {key.key}")

    data: list[Any] = [key]
    parser.skip_comma()

    if parser.tokens[parser.pos].kind != TokenKind.END_LINE:
        data.append(_duration(parser))
    else:
        data.append(None)

    return data


def greedy_name(parser: Parser) -> list[Any]:
    """Parse the line as one name."""
    # A lone identifier is already the whole name and may be a $constant, so keep it intact.
    if (
        parser.tokens[parser.pos].kind == TokenKind.identifier
        and parser.pos + 1 < len(parser.tokens)
        and parser.tokens[parser.pos + 1].kind == TokenKind.END_LINE
    ):
        return [parser.expect_consume(TokenKind.identifier).literal]

    # Written unquoted. Glue back the tokens the lexer split.
    parts: list[str] = []

    while parser.pos < len(parser.tokens) and parser.tokens[parser.pos].kind != TokenKind.END_LINE:
        parts.append(parser.tokens[parser.pos].literal)
        parser.pos += 1

    if not parts:
        parser.err(parser.tokens[min(parser.pos, len(parser.tokens) - 1)], "Expected a name")

    return [" ".join(parts)]


@dataclass(frozen=True)
class _LogValue:
    """One value `log` can print."""

    kind: LogKind
    label: str
    template: str
    reads: tuple[EvalKind, ...]
    takes_window_path: bool = False
    takes_name: bool = False
    is_constant: bool = False


_LOG_VALUES: dict[TokenKind, _LogValue] = {
    TokenKind.command_expr_health: _LogValue(LogKind.multi, "health", "%d/%d", (EvalKind.health, EvalKind.max_health)),
    TokenKind.command_expr_mana: _LogValue(LogKind.multi, "mana", "%d/%d", (EvalKind.mana, EvalKind.max_mana)),
    TokenKind.command_expr_energy: _LogValue(LogKind.multi, "energy", "%d/%d", (EvalKind.energy, EvalKind.max_energy)),
    TokenKind.command_expr_bagcount: _LogValue(
        LogKind.multi, "bagcount", "%d/%d", (EvalKind.bagcount, EvalKind.max_bagcount)
    ),
    TokenKind.command_expr_gold: _LogValue(LogKind.multi, "gold", "%d/%d", (EvalKind.gold, EvalKind.max_gold)),
    TokenKind.command_expr_potion_count: _LogValue(
        LogKind.multi, "potioncount", "%d/%d", (EvalKind.potioncount, EvalKind.max_potioncount)
    ),
    TokenKind.command_expr_playercount: _LogValue(LogKind.single, "playercount", "%d", (EvalKind.playercount,)),
    TokenKind.command_expr_counter: _LogValue(LogKind.single, "counter", "%d", (EvalKind.counter,), takes_name=True),
    TokenKind.command_expr_account_level: _LogValue(LogKind.multi, "accountlevel", "%d", (EvalKind.account_level,)),
    TokenKind.command_expr_duel_round: _LogValue(LogKind.multi, "duelround", "%d", (EvalKind.duel_round,)),
    TokenKind.command_expr_any_player_list: _LogValue(
        LogKind.single, "clients using anyplayer", "%s", (EvalKind.any_player_list,)
    ),
    TokenKind.command_expr_window_text: _LogValue(
        LogKind.multi, "windowtext", "%s", (EvalKind.windowtext,), takes_window_path=True
    ),
    TokenKind.command_expr_window_num: _LogValue(
        LogKind.multi, "windownum", "%s", (EvalKind.windownum,), takes_window_path=True
    ),
}


def _read_log_value_kind(parser: Parser, pos: int) -> _LogValue | None:
    """What the token at `pos` prints."""
    token: Token = parser.tokens[pos]

    # Spelled as plain words, so not in the table.
    if token.kind == TokenKind.identifier:
        if token.literal == "window" and _starts_window_path(parser, pos + 1):
            return _LOG_VALUES[TokenKind.command_expr_window_text]

        if token.literal.startswith("$"):
            return _LogValue(LogKind.single, token.literal[1:], "%s", (), is_constant=True)

        return None

    return _LOG_VALUES.get(token.kind)


def _read_log_value(parser: Parser) -> tuple[_LogValue, list[Expression]] | None:
    """Consume the next token as a value."""
    value: _LogValue | None = _read_log_value_kind(parser, parser.pos)

    if value is None:
        return None

    parser.pos += 1

    if value.is_constant:
        return value, [ConstantReferenceExpression(value.label)]

    if value.takes_window_path:
        window_path: list[str] | Expression = parser.parse_window_path()
        return value, [Eval(value.reads[0], [window_path])]

    # The name is part of what a counter is called. Join the label.
    if value.takes_name:
        name: IdentExpression = parser.consume_any_ident()
        named: _LogValue = replace(value, label=f"{value.label} {name.ident}")
        return named, [Eval(value.reads[0], [StringExpression(name.ident)])]

    return value, [Eval(read) for read in value.reads]


def _literal_text(token: Token) -> str:
    """How a token reads as text."""
    return token.value if token.kind == TokenKind.string else token.literal


def _all_literal(parser: Parser) -> list[Any]:
    """Join the line into one string."""
    parts: list[str] = []

    while parser.pos < len(parser.tokens) and parser.tokens[parser.pos].kind != TokenKind.END_LINE:
        parts.append(_literal_text(parser.tokens[parser.pos]))
        parser.pos += 1

    return [LogKind.single, StringExpression(" ".join(parts))]


def parse_log(parser: Parser) -> list[Any]:
    """Parse `log`, words mixed with values."""
    # Opening on a bare word means a sentence.
    if parser.tokens[parser.pos].kind != TokenKind.string and _read_log_value_kind(parser, parser.pos) is None:
        return _all_literal(parser)

    pieces: list[str | tuple[_LogValue, list[Expression]]] = []

    while parser.pos < len(parser.tokens) and parser.tokens[parser.pos].kind != TokenKind.END_LINE:
        read: tuple[_LogValue, list[Expression]] | None = _read_log_value(parser)

        if read is not None:
            pieces.append(read)
            continue

        pieces.append(_literal_text(parser.tokens[parser.pos]))
        parser.pos += 1

    values: list[tuple[_LogValue, list[Expression]]] = [piece for piece in pieces if not isinstance(piece, str)]

    if not values:
        return [LogKind.single, StringExpression(" ".join(str(piece) for piece in pieces))]

    # Nothing written says what it is. Name it.
    if len(pieces) == 1:
        value, reads = values[0]

        # Keeps its old `name = value` shape.
        if value.is_constant:
            return [value.kind, IdentExpression(value.label)]

        return [value.kind, StrFormatExpression(f"{value.label}: {value.template}", *reads)]

    # Script text, not a placeholder. Double it.
    template: str = " ".join(
        piece.replace("%", "%%") if isinstance(piece, str) else piece[0].template for piece in pieces
    )
    reads = [read for _, value_reads in values for read in value_reads]

    # One per-client value makes the whole line per client.
    kind: LogKind = LogKind.multi if any(value.kind == LogKind.multi for value, _ in values) else LogKind.single
    return [kind, StrFormatExpression(template, *reads)]


def parse_teleport(parser: Parser) -> list[Any]:
    """Parse `teleport` and its forms."""
    if parser.consume_optional(TokenKind.keyword_mob) is not None:
        return [TeleportKind.mob]

    if parser.consume_optional(TokenKind.keyword_quest) is not None:
        return [TeleportKind.quest]

    if num_tok := parser.consume_optional(TokenKind.player_num):
        return [TeleportKind.client_num, num_tok.value]

    return [TeleportKind.position, _position(parser)]


def parse_friend_teleport(parser: Parser) -> list[Any]:
    """Parse `friendtp`, icon or name."""
    first: Token = parser.expect_consume_any([TokenKind.keyword_icon, TokenKind.identifier])

    if first.kind == TokenKind.keyword_icon:
        return [TeleportKind.friend_icon]

    # A single word may be a constant holding the real name, so leave it for the VM to resolve.
    if parser.tokens[parser.pos].kind == TokenKind.END_LINE:
        return [TeleportKind.friend_name, IdentExpression(first.literal)]

    # Friend names hold spaces. Rest of the line is the name.
    name_parts: list[str] = [first.literal]

    while parser.pos < len(parser.tokens) and parser.tokens[parser.pos].kind != TokenKind.END_LINE:
        name_parts.append(parser.tokens[parser.pos].literal)
        parser.pos += 1

    return [TeleportKind.friend_name, " ".join(name_parts)]


def parse_entity_teleport(parser: Parser) -> list[Any]:
    """Parse a teleport to a named entity, walking there if asked."""
    nav_mode: bool = False

    # Leading nav walks instead of teleporting straight on.
    if parser.tokens[parser.pos].kind == TokenKind.command_nav:
        parser.pos += 1
        nav_mode = True

    arg: Token | None = parser.consume_optional(TokenKind.string)

    # Quoted name matches exactly. Bare one matches loosely.
    if arg is not None:
        data: list[Any] = [TeleportKind.entity_literal, arg.value]

    elif parser.tokens[parser.pos].kind == TokenKind.identifier:
        ident: Token = parser.expect_consume(TokenKind.identifier)
        data: list[Any] = [TeleportKind.entity_vague, ident.literal]

    else:
        token: Token = parser.tokens[parser.pos]
        parser.pos += 1
        data: list[Any] = [TeleportKind.entity_vague, token.literal]

    # Kind stays first so the VM dispatches before looking for nav.
    if nav_mode:
        data.insert(1, TeleportKind.nav)

    return data


# Take no arguments.
command(TokenKind.command_kill, CommandKind.kill, nullary)
command(TokenKind.command_restart_bot, CommandKind.restart_bot, nullary)
command(TokenKind.command_relog, CommandKind.relog, nullary)
command(TokenKind.command_autopet, CommandKind.autopet, nullary)
command(TokenKind.command_getdeck, CommandKind.getdeck, nullary)
command(TokenKind.command_set_zone, CommandKind.set_zone, nullary)
command(TokenKind.command_set_goal, CommandKind.set_goal, nullary)
command(TokenKind.command_set_quest, CommandKind.set_quest, nullary)


# Take one value.
command(TokenKind.command_sleep, CommandKind.sleep, duration)
command(TokenKind.command_goto, CommandKind.goto, xyz_or_expression)
command(TokenKind.command_setdeck, CommandKind.setdeck, string_literal)


# Offset the current position instead of replacing it.
command(TokenKind.command_plus_teleport, CommandKind.teleport, prefixed_xyz_or_expression(TeleportKind.plusteleport))
command(TokenKind.command_minus_teleport, CommandKind.teleport, prefixed_xyz_or_expression(TeleportKind.minusteleport))


# Aimed at a window, not a coordinate.
command(TokenKind.command_clickwindow, CommandKind.click, prefixed_window_or_expression(ClickKind.window))
command(TokenKind.command_move_cursor_window, CommandKind.cursor, prefixed_window_or_expression(CursorKind.window))


# Block until the game reaches a state.
command(TokenKind.command_waitfor_zonechange, CommandKind.waitfor, waitfor(WaitforKind.zonechange))
command(TokenKind.command_waitfor_battle, CommandKind.waitfor, waitfor(WaitforKind.battle))
command(TokenKind.command_waitfor_free, CommandKind.waitfor, waitfor(WaitforKind.free))
command(TokenKind.command_waitfor_dialog, CommandKind.waitfor, waitfor(WaitforKind.dialog))
command(TokenKind.command_waitfor_window, CommandKind.waitfor, waitfor_window(WaitforKind.window))


# Steered by an optional keyword.
command(TokenKind.command_buypotions, CommandKind.buypotions, flag_presence(TokenKind.keyword_ifneeded))
command(
    TokenKind.command_toggle_combat,
    CommandKind.toggle_combat,
    optional_choice(TokenKind.logical_on, TokenKind.logical_off),
    every_client=True,
)


# Commands taking one literal that a constant or expression may stand in for.
command(
    TokenKind.command_tozone,
    CommandKind.tozone,
    first_of(TokenKind.path, lambda parser: parser.parse_zone_path(), "a zone", StringExpression),
)
command(
    TokenKind.command_load_playstyle,
    CommandKind.load_playstyle,
    first_of(TokenKind.string, lambda parser: parser.expect_consume(TokenKind.string).value, "a playstyle name"),
)
command(
    TokenKind.command_set_yaw,
    CommandKind.set_yaw,
    first_of(TokenKind.number, lambda parser: parser.expect_consume(TokenKind.number).value, "an angle"),
)


# Take a coordinate pair, which usepotion may leave out.
command(TokenKind.command_move_cursor, CommandKind.cursor, xy_pair(CursorKind.position))
command(TokenKind.command_click, CommandKind.click, xy_pair(ClickKind.position))
command(TokenKind.command_usepotion, CommandKind.usepotion, optional_pair)


# Read the rest of the line loosely.
command(TokenKind.command_sendkey, CommandKind.sendkey, key_with_optional_expression)
command(TokenKind.command_select_friend, CommandKind.select_friend, greedy_name)


# Irregular enough to need their own parser.
command(TokenKind.command_log, CommandKind.log, parse_log)
command(TokenKind.command_teleport, CommandKind.teleport, parse_teleport)
command(TokenKind.command_friendtp, CommandKind.teleport, parse_friend_teleport)
command(TokenKind.command_entitytp, CommandKind.teleport, parse_entity_teleport)
