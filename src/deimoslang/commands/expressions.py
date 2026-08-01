"""Every expression command's surface syntax."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from ..ast import (
    Command,
    CommandExpression,
    CommandKind,
    ContainsStringExpression,
    EquivalentExpression,
    Eval,
    EvalKind,
    Expression,
    ExprKind,
    IdentExpression,
    ListExpression,
    OrExpression,
    PlayerSelector,
    SelectorGroup,
    StringExpression,
    describe_expression,
)
from ..tokens import TokenKind

if TYPE_CHECKING:
    from ..lexer import Token
    from ..parser import Parser

type ExprShape = Callable[[Parser, PlayerSelector, TokenKind], Expression]


@dataclass(frozen=True)
class ExprSpec:
    """How one expression command parses."""

    token: TokenKind
    parse: ExprShape


# Filled by the expr_command calls below.
EXPR_REGISTRY: dict[TokenKind, ExprSpec] = {}


def expr_command(token: TokenKind, parse: ExprShape) -> ExprSpec:
    """Register how one expression command parses."""
    if token in EXPR_REGISTRY:
        raise ValueError(f"{token.name} is already registered as an expression command")

    spec: ExprSpec = ExprSpec(token=token, parse=parse)
    EXPR_REGISTRY[token] = spec
    return spec


def _command_expression(parser: Parser, selector: PlayerSelector, data: list[Any]) -> CommandExpression:
    """Package a condition for the VM."""
    command: Command = Command()
    command.player_selector = selector
    command.kind = CommandKind.expr
    command.data = data
    return CommandExpression(command)


def numeric_stat(parser: Parser, selector: PlayerSelector, token: TokenKind) -> Expression:
    """Parse a stat comparison."""
    return parser.parse_numeric_stat_expression(token, selector)


def flag(kind: ExprKind) -> ExprShape:
    """Shape for a bare condition."""

    def shape(parser: Parser, selector: PlayerSelector, _token: TokenKind) -> Expression:
        """Parse the bare predicate."""
        parser.pos += 1
        return _command_expression(parser, selector, [kind])

    return shape


def value(kind: ExprKind, accepted: list[TokenKind | str]) -> ExprShape:
    """Shape for one plain value."""

    def shape(parser: Parser, selector: PlayerSelector, _token: TokenKind) -> Expression:
        """Parse the one value it takes."""
        parser.pos += 1
        return _command_expression(parser, selector, [kind, parser.parse_value(accepted)])

    return shape


def _lowered(text: Expression) -> str | Expression:
    """Lowercase a string, pass names through."""
    return text.string.lower() if isinstance(text, StringExpression) else text


def lowered_value(kind: ExprKind, accepted: list[TokenKind | str]) -> ExprShape:
    """Shape for one lowercased string."""

    def shape(parser: Parser, selector: PlayerSelector, _token: TokenKind) -> Expression:
        """Parse the value and lowercase it."""
        parser.pos += 1
        return _command_expression(parser, selector, [kind, _lowered(parser.parse_value(accepted))])

    return shape


def changed_to(kind: ExprKind, accepted: list[TokenKind | str]) -> ExprShape:
    """Shape for a change predicate."""

    def shape(parser: Parser, selector: PlayerSelector, _token: TokenKind) -> Expression:
        """Parse the change and its target."""
        parser.pos += 1

        if parser.pos < len(parser.tokens) and parser.tokens[parser.pos].kind == TokenKind.logical_to:
            parser.pos += 1
            return _command_expression(parser, selector, [kind, _lowered(parser.parse_value(accepted))])

        return _command_expression(parser, selector, [kind])

    return shape


def parse_items_dropped(parser: Parser, selector: PlayerSelector, _token: TokenKind) -> Expression:
    """Parse `itemdropped`, one item or list."""
    parser.pos += 1

    # List means any named item counts as a drop.
    if parser.pos < len(parser.tokens) and parser.tokens[parser.pos].kind == TokenKind.square_open:
        item_list: ListExpression = parser.parse_list()
        items: list[str] = []

        for item_expr in item_list.items:
            if isinstance(item_expr, StringExpression):
                items.append(item_expr.string.lower())
            else:
                parser.err(
                    parser.tokens[parser.pos - 1], f"Expected an item name, got {describe_expression(item_expr)}"
                )

        return _command_expression(parser, selector, [ExprKind.items_dropped, items])

    item: Expression = parser.parse_value([TokenKind.string, TokenKind.identifier])
    return _command_expression(parser, selector, [ExprKind.items_dropped, _lowered(item)])


def parse_counter(parser: Parser, selector: PlayerSelector, _token: TokenKind) -> Expression:
    """Parse `counter <name>` and its numeric comparison."""
    parser.pos += 1
    name: IdentExpression = parser.consume_any_ident()
    evaluated: Eval = Eval(EvalKind.counter, [StringExpression(name.ident)])
    return parser.parse_numeric_comparison(evaluated, selector)


def parse_timer(parser: Parser, selector: PlayerSelector, _token: TokenKind) -> Expression:
    """Parse `timer <name>` and its numeric comparison, in seconds."""
    parser.pos += 1
    name: IdentExpression = parser.consume_any_ident()
    evaluated: Eval = Eval(EvalKind.timer, [StringExpression(name.ident)])
    return parser.parse_numeric_comparison(evaluated, selector)


def parse_window_num(parser: Parser, selector: PlayerSelector, _token: TokenKind) -> Expression:
    """Parse `windownum` and its numeric comparison."""
    parser.pos += 1
    window_path: list[str] | Expression = parser.parse_window_path()
    evaluated: Eval = Eval(EvalKind.windownum, [window_path])
    return parser.parse_indexed_numeric_comparison(evaluated, selector)


def parse_window_text(parser: Parser, selector: PlayerSelector, _token: TokenKind) -> Expression:
    """Parse a window-text check."""
    parser.pos += 1
    window_path: list[str] | Expression = parser.parse_window_path()
    contains: Token | None = parser.consume_optional(TokenKind.contains)

    # List means any one string may match.
    if parser.pos < len(parser.tokens) and parser.tokens[parser.pos].kind == TokenKind.square_open:
        string_list: ListExpression = parser.parse_list()

        # ContainsStringExpression searches a whole list already. No unrolling.
        if contains:
            return SelectorGroup(
                selector,
                ContainsStringExpression(
                    Eval(EvalKind.windowtext, [window_path]),
                    string_list,
                ),
            )

        # Exact match unrolls to one test per entry, joined by or.
        or_expressions: list[Expression] = []
        window_text_eval: Eval = Eval(EvalKind.windowtext, [window_path])

        for string_expr in string_list.items:
            if isinstance(string_expr, StringExpression):
                or_expressions.append(
                    EquivalentExpression(window_text_eval, StringExpression(string_expr.string.lower()))
                )

            elif isinstance(string_expr, IdentExpression):
                or_expressions.append(EquivalentExpression(window_text_eval, string_expr))

        if len(or_expressions) == 1:
            return SelectorGroup(selector, or_expressions[0])

        return SelectorGroup(selector, OrExpression(or_expressions))

    target_expr: Expression = parser.parse_value([TokenKind.string, TokenKind.identifier])

    if isinstance(target_expr, StringExpression):
        string_value: str = target_expr.string.lower()

    # Name resolves at runtime. Cannot fold to lowercase here.
    elif isinstance(target_expr, IdentExpression):
        if contains:
            return SelectorGroup(
                selector, ContainsStringExpression(Eval(EvalKind.windowtext, [window_path]), target_expr)
            )

        return SelectorGroup(selector, EquivalentExpression(Eval(EvalKind.windowtext, [window_path]), target_expr))

    else:
        parser.err(
            parser.tokens[parser.pos - 1],
            f"Expected a quoted string or a name, got {describe_expression(target_expr)}",
        )

    if contains:
        return SelectorGroup(
            selector,
            ContainsStringExpression(Eval(EvalKind.windowtext, [window_path]), StringExpression(string_value)),
        )

    return SelectorGroup(
        selector,
        EquivalentExpression(Eval(EvalKind.windowtext, [window_path]), StringExpression(string_value)),
    )


_STAT_COMMANDS: list[TokenKind] = [
    TokenKind.command_expr_health,
    TokenKind.command_expr_health_above,
    TokenKind.command_expr_health_below,
    TokenKind.command_expr_mana,
    TokenKind.command_expr_mana_above,
    TokenKind.command_expr_mana_below,
    TokenKind.command_expr_energy,
    TokenKind.command_expr_energy_above,
    TokenKind.command_expr_energy_below,
    TokenKind.command_expr_gold,
    TokenKind.command_expr_gold_above,
    TokenKind.command_expr_gold_below,
    TokenKind.command_expr_bagcount,
    TokenKind.command_expr_bagcount_above,
    TokenKind.command_expr_bagcount_below,
    TokenKind.command_expr_potion_count,
    TokenKind.command_expr_potion_countabove,
    TokenKind.command_expr_potion_countbelow,
    TokenKind.command_expr_playercount,
    TokenKind.command_expr_playercountabove,
    TokenKind.command_expr_playercountbelow,
    TokenKind.command_expr_account_level,
    TokenKind.command_expr_duel_round,
]

# All parse identically. Only the token differs.
for _token in _STAT_COMMANDS:
    expr_command(_token, numeric_stat)


_FLAGS: list[tuple[TokenKind, ExprKind]] = [
    (TokenKind.command_expr_loading, ExprKind.loading),
    (TokenKind.command_expr_in_combat, ExprKind.in_combat),
    (TokenKind.command_expr_has_dialogue, ExprKind.has_dialogue),
    (TokenKind.command_expr_same_zone, ExprKind.same_zone),
    (TokenKind.command_expr_same_quest, ExprKind.same_quest),
    (TokenKind.command_expr_same_xyz, ExprKind.same_xyz),
    (TokenKind.command_expr_same_yaw, ExprKind.same_yaw),
    (TokenKind.command_expr_same_place, ExprKind.same_place),
]

for _token, _kind in _FLAGS:
    expr_command(_token, flag(_kind))


expr_command(TokenKind.command_expr_window_visible, value(ExprKind.window_visible, ["window_path"]))

expr_command(TokenKind.command_expr_window_disabled, value(ExprKind.window_disabled, ["window_path"]))

expr_command(TokenKind.command_expr_in_zone, value(ExprKind.in_zone, [TokenKind.path, TokenKind.identifier]))

expr_command(TokenKind.command_expr_has_xyz, value(ExprKind.has_xyz, [TokenKind.keyword_xyz, TokenKind.identifier]))

expr_command(TokenKind.command_expr_has_yaw, value(ExprKind.has_yaw, [TokenKind.number, TokenKind.identifier]))

expr_command(
    TokenKind.command_expr_has_quest,
    lowered_value(ExprKind.has_quest, [TokenKind.string, TokenKind.identifier]),
)

expr_command(
    TokenKind.command_expr_in_range,
    lowered_value(ExprKind.in_range, [TokenKind.string, TokenKind.identifier]),
)

expr_command(TokenKind.command_expr_tracking_quest, lowered_value(ExprKind.tracking_quest, [TokenKind.string]))

expr_command(TokenKind.command_expr_tracking_goal, lowered_value(ExprKind.tracking_goal, [TokenKind.string]))

expr_command(
    TokenKind.command_expr_zone_changed,
    changed_to(ExprKind.zone_changed, [TokenKind.path, TokenKind.identifier]),
)

expr_command(
    TokenKind.command_expr_goal_changed,
    changed_to(ExprKind.goal_changed, [TokenKind.string, TokenKind.identifier]),
)

expr_command(
    TokenKind.command_expr_quest_changed,
    changed_to(ExprKind.quest_changed, [TokenKind.string, TokenKind.identifier]),
)

expr_command(TokenKind.command_expr_item_dropped, parse_items_dropped)

expr_command(TokenKind.command_expr_window_num, parse_window_num)
expr_command(TokenKind.command_expr_counter, parse_counter)
expr_command(TokenKind.command_expr_timer, parse_timer)

expr_command(TokenKind.command_expr_window_text, parse_window_text)
