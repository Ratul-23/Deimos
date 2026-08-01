"""Tokens into a syntax tree."""

from typing import NoReturn

from .ast import (
    CROSS_CLIENT_CHECKS,
    AndExpression,
    BlockDefStmt,
    BreakStmt,
    CallStmt,
    Command,
    CommandExpression,
    CommandStmt,
    ConstantCheckExpression,
    ConstantDeclStmt,
    ConstantExpression,
    ConstantReferenceExpression,
    CounterAction,
    CounterStmt,
    DivideExpression,
    EquivalentExpression,
    Eval,
    EvalKind,
    Expression,
    GreaterEqualExpression,
    GreaterExpression,
    IdentExpression,
    IfStmt,
    IndexAccessExpression,
    KeyExpression,
    ListExpression,
    LoopStmt,
    MixinStmt,
    NumberExpression,
    OrExpression,
    ParallelCommandStmt,
    PlayerSelector,
    RangeMaxExpression,
    RangeMinExpression,
    ReturnStmt,
    SelectorError,
    SelectorGroup,
    Stmt,
    StmtList,
    StringExpression,
    TimerAction,
    TimerStmt,
    TimesStmt,
    UnaryExpression,
    UnaryOp,
    UntilStmt,
    WhileStmt,
    XYZExpression,
    asks_any_player,
    describe_expression,
    operand_selector,
)
from .commands import EXPR_REGISTRY as EXPR_COMMAND_REGISTRY
from .commands import REGISTRY as COMMAND_REGISTRY
from .commands import CommandSpec, ExprSpec
from .lexer import DeimosLangError, LineInfo, Token, TokenKind
from .tokens import describe, describe_any


class ParserError(DeimosLangError):
    """Tokens do not form valid syntax."""


def _joined_parts(expr: Expression, joiner: type[AndExpression] | type[OrExpression]) -> list[Expression]:
    """An `and` or `or`'s operands, flattened."""
    if not isinstance(expr, joiner):
        return [expr]

    return [part for operand in expr.expressions for part in _joined_parts(operand, joiner)]


def _narrow_to_any(expr: Expression) -> Expression:
    """Aim a bare operand at `any`."""
    # isbetween and joined conditions hold several operands, so narrow each.
    if isinstance(expr, AndExpression | OrExpression):
        expr.expressions = [_narrow_to_any(part) for part in expr.expressions]
        return expr

    # Same client, so the `not` moves onto the check.
    if isinstance(expr, UnaryExpression) and expr.operator == UnaryOp.not_:
        negated, inner = True, expr.expr
    else:
        negated, inner = False, expr

    # Cross-client checks never mean one client.
    if isinstance(inner, CommandExpression) and inner.command.data and inner.command.data[0] in CROSS_CLIENT_CHECKS:
        return expr

    selector: PlayerSelector | None = operand_selector(inner)

    if selector is None or not selector.implicit:
        return expr

    selector.mass = False
    selector.any_player = True
    selector.negated = negated
    return inner


# All parse the same, only the action differs.
_TIMER_ACTIONS: dict[TokenKind, TimerAction] = {
    TokenKind.keyword_starttimer: TimerAction.start,
    TokenKind.keyword_resettimer: TimerAction.reset,
    TokenKind.keyword_endtimer: TimerAction.end,
}

_COUNTER_ACTIONS: dict[TokenKind, CounterAction] = {
    TokenKind.keyword_startcounter: CounterAction.start,
    TokenKind.keyword_resetcounter: CounterAction.reset,
    TokenKind.keyword_endcounter: CounterAction.end,
    TokenKind.keyword_addone: CounterAction.add,
    TokenKind.keyword_minusone: CounterAction.subtract,
}


class Parser:
    """Builds a syntax tree from tokens."""

    def __init__(self, tokens: list[Token]) -> None:
        self.tokens: list[Token] = tokens
        self.pos: int = 0

    def err_manual(self, line_info: LineInfo, msg: str) -> NoReturn:
        """Raise a ParserError at a location."""
        raise ParserError(f"{msg}\n{line_info.render()}")

    def err(self, token: Token, msg: str) -> NoReturn:
        """Raise a ParserError at a token."""
        self.err_manual(token.line_info, msg)

    def skip_any(self, kinds: list[TokenKind]) -> None:
        """Consume the next token if accepted."""
        if self.pos < len(self.tokens) and self.tokens[self.pos].kind in kinds:
            self.pos += 1

    def skip_comma(self) -> None:
        """Consume a comma if present."""
        self.skip_any([TokenKind.comma])

    def expect_consume_any(self, kinds: list[TokenKind]) -> Token:
        """Require one of the accepted kinds."""
        if self.pos >= len(self.tokens):
            self.err(self.tokens[-1], f"Premature end of file, expected {describe_any(kinds)} before the end")

        result: Token = self.tokens[self.pos]

        if result.kind not in kinds:
            self.err(result, f"Expected {describe_any(kinds)} but got {describe(result.kind)}")

        self.pos += 1
        return result

    def expect_consume(self, kind: TokenKind) -> Token:
        """Require one exact kind."""
        return self.expect_consume_any([kind])

    def consume_any_optional(self, kinds: list[TokenKind]) -> Token | None:
        """An accepted kind, or None."""
        if self.pos >= len(self.tokens):
            return None

        result: Token = self.tokens[self.pos]

        if result.kind not in kinds:
            return None

        self.pos += 1
        return result

    def consume_optional(self, kind: TokenKind) -> Token | None:
        """One exact kind, or None."""
        return self.consume_any_optional([kind])

    def parse_numeric_comparison(self, evaluated: Expression, player_selector: PlayerSelector) -> Expression:
        """Parse a comparison against a value."""
        if self.pos < len(self.tokens) and self.tokens[self.pos].kind in [
            TokenKind.greater,
            TokenKind.less,
            TokenKind.equals,
        ]:
            operator: Token = self.tokens[self.pos]
            self.pos += 1
            target: Expression = self.parse_expression()

            if operator.kind == TokenKind.greater:
                return self.gen_greater_expression(evaluated, target, player_selector)

            elif operator.kind == TokenKind.less:
                return self.gen_greater_expression(target, evaluated, player_selector)

            elif operator.kind == TokenKind.equals:
                return self.gen_equivalent_expression(evaluated, target, player_selector)

        elif self.pos < len(self.tokens) and self.tokens[self.pos].kind == TokenKind.keyword_isbetween:
            self.pos += 1

            # The bounds come either from a named range or from a "min-max" string.
            if self.pos < len(self.tokens) and self.tokens[self.pos].kind == TokenKind.identifier:
                return self.gen_named_range_check(evaluated, player_selector)

            else:
                range_str: str = self.expect_consume(TokenKind.string).value

                try:
                    min_val, max_val = map(float, range_str.split("-"))
                    min_expr: Expression = self.gen_greater_equal_expression(
                        evaluated, NumberExpression(min_val), player_selector
                    )
                    max_expr: Expression = self.gen_greater_equal_expression(
                        NumberExpression(max_val), evaluated, player_selector
                    )
                    return AndExpression([min_expr, max_expr])

                except ValueError:
                    self.err(
                        self.tokens[self.pos - 1], f"Invalid range format: {range_str}. Expected format like '1-100'"
                    )

        # Nothing to compare against. The value is the condition.
        return SelectorGroup(player_selector, evaluated)

    def parse_indexed_numeric_comparison(self, evaluated: Expression, player_selector: PlayerSelector) -> Expression:
        """Compare each number in a window."""
        # One comparison per number found, matched by position.
        if self.pos < len(self.tokens) and self.tokens[self.pos].kind == TokenKind.square_open:
            self.pos += 1
            expressions: list[Expression] = []
            index: int = 0

            while self.pos < len(self.tokens) and self.tokens[self.pos].kind != TokenKind.square_close:
                if self.tokens[self.pos].kind == TokenKind.comma:
                    self.pos += 1
                    continue

                indexed_eval: IndexAccessExpression = IndexAccessExpression(evaluated, NumberExpression(index))

                if self.tokens[self.pos].kind in [TokenKind.greater, TokenKind.less, TokenKind.equals]:
                    operator: Token = self.tokens[self.pos]
                    self.pos += 1
                    target: Expression = self.parse_expression()

                    if operator.kind == TokenKind.greater:
                        expressions.append(self.gen_greater_expression(indexed_eval, target, player_selector))

                    elif operator.kind == TokenKind.less:
                        expressions.append(self.gen_greater_expression(target, indexed_eval, player_selector))

                    else:
                        expressions.append(self.gen_equivalent_expression(indexed_eval, target, player_selector))

                elif self.tokens[self.pos].kind == TokenKind.keyword_isbetween:
                    self.pos += 1

                    if self.pos < len(self.tokens) and self.tokens[self.pos].kind == TokenKind.identifier:
                        expressions.append(self.gen_named_range_check(indexed_eval, player_selector))

                    else:
                        range_str: str = self.expect_consume(TokenKind.string).value

                        try:
                            min_val, max_val = map(float, range_str.split("-"))
                            min_expr: Expression = self.gen_greater_equal_expression(
                                indexed_eval, NumberExpression(min_val), player_selector
                            )
                            max_expr: Expression = self.gen_greater_equal_expression(
                                NumberExpression(max_val), indexed_eval, player_selector
                            )
                            expressions.append(AndExpression([min_expr, max_expr]))

                        except ValueError:
                            self.err(
                                self.tokens[self.pos - 1],
                                f"Invalid range format: {range_str}. Expected format like '1-100'",
                            )

                else:
                    expressions.append(
                        self.gen_equivalent_expression(indexed_eval, self.parse_expression(), player_selector)
                    )

                index += 1

                if self.pos < len(self.tokens) and self.tokens[self.pos].kind == TokenKind.comma:
                    self.pos += 1

            closing: Token = self.expect_consume(TokenKind.square_close)

            if len(expressions) == 0:
                self.err(closing, "Expected at least one comparison between the brackets")

            if len(expressions) == 1:
                return expressions[0]

            return AndExpression(expressions)

        # Without brackets the check applies to the first number in the text.
        return self.parse_numeric_comparison(IndexAccessExpression(evaluated, NumberExpression(0)), player_selector)

    def gen_named_range_check(self, evaluated: Expression, player_selector: PlayerSelector) -> Expression:
        """Consume a range name and build an inclusive check of a value against it."""
        literal: str = self.tokens[self.pos].literal
        self.pos += 1

        range_expr: Expression = (
            ConstantReferenceExpression(literal[1:]) if literal.startswith("$") else IdentExpression(literal)
        )

        min_expr: Expression = self.gen_greater_equal_expression(
            evaluated, RangeMinExpression(range_expr), player_selector
        )
        max_expr: Expression = self.gen_greater_equal_expression(
            RangeMaxExpression(range_expr), evaluated, player_selector
        )

        return AndExpression([min_expr, max_expr])

    def parse_atom(self) -> Expression:
        """Parse a literal, list, path, coordinate or identifier."""
        # A $name reads the constant's value, while a bare name is used as written.
        if (
            self.pos < len(self.tokens)
            and self.tokens[self.pos].kind == TokenKind.identifier
            and self.tokens[self.pos].literal.startswith("$")
        ):
            constant_name: str = self.tokens[self.pos].literal[1:]
            self.pos += 1
            return ConstantReferenceExpression(constant_name)

        if self.pos < len(self.tokens) and self.tokens[self.pos].kind == TokenKind.boolean_true:
            token: Token = self.tokens[self.pos]
            self.pos += 1
            return ConstantExpression(token.literal, StringExpression("true"))

        if self.pos < len(self.tokens) and self.tokens[self.pos].kind == TokenKind.boolean_false:
            token: Token = self.tokens[self.pos]
            self.pos += 1
            return ConstantExpression(token.literal, StringExpression("false"))

        if self.pos < len(self.tokens) and self.tokens[self.pos].kind == TokenKind.square_open:
            return self.parse_list()

        if self.pos < len(self.tokens) and self.tokens[self.pos].kind == TokenKind.path:
            return self.parse_zone_path_expression()

        if self.pos < len(self.tokens) and self.tokens[self.pos].kind == TokenKind.keyword_xyz:
            return self.parse_xyz()

        if self.pos < len(self.tokens) and self.tokens[self.pos].kind == TokenKind.identifier:
            tok: Token = self.tokens[self.pos]
            self.pos += 1
            return IdentExpression(tok.literal)

        tok: Token = self.expect_consume_any([TokenKind.number, TokenKind.string, TokenKind.percent])

        match tok.kind:
            case TokenKind.number:
                return NumberExpression(tok.value)

            case TokenKind.percent:
                return NumberExpression(tok.value)

            case TokenKind.string:
                return StringExpression(tok.value)

            case _:
                self.err(tok, f"Expected a value, got {describe(tok.kind)}")

    def parse_unary_expression(self) -> UnaryExpression | Expression:
        """Parse a minus, then an atom."""
        kinds: list[TokenKind] = [TokenKind.minus]

        if self.tokens[self.pos].kind in kinds:
            self.expect_consume_any(kinds)
            return UnaryExpression(UnaryOp.negate, self.parse_unary_expression())

        else:
            return self.parse_atom()

    def gen_greater_expression(
        self, left: Expression, right: Expression, player_selector: PlayerSelector
    ) -> Expression:
        """A greater-than test, scoped."""
        return SelectorGroup(player_selector, GreaterExpression(left, right))

    def gen_greater_equal_expression(
        self, left: Expression, right: Expression, player_selector: PlayerSelector
    ) -> Expression:
        """A greater-or-equal test, scoped."""
        return SelectorGroup(player_selector, GreaterEqualExpression(left, right))

    def gen_equivalent_expression(
        self, left: Expression, right: Expression, player_selector: PlayerSelector
    ) -> Expression:
        """An equality test, scoped."""
        return SelectorGroup(player_selector, EquivalentExpression(left, right))

    def parse_value(self, expected_types: list[TokenKind | str] | None = None) -> Expression:
        """Parse one accepted value."""
        if expected_types is None:
            expected_types = [TokenKind.number, TokenKind.string, TokenKind.percent, TokenKind.identifier]

        # A window path may arrive as an identifier, since a constant can hold the whole path.
        if (
            (TokenKind.identifier in expected_types or "window_path" in expected_types)
            and self.pos < len(self.tokens)
            and self.tokens[self.pos].kind == TokenKind.identifier
        ):
            ident: str = self.tokens[self.pos].literal
            self.pos += 1

            if ident.startswith("$"):
                return ConstantReferenceExpression(ident[1:])

            return IdentExpression(ident)

        if (
            "window_path" in expected_types
            and self.pos < len(self.tokens)
            and self.tokens[self.pos].kind == TokenKind.square_open
        ):
            return self.parse_list()

        if TokenKind.path in expected_types and self.pos < len(self.tokens):
            if self.tokens[self.pos].kind == TokenKind.path:
                return self.parse_zone_path_expression()

            # A zone written without slashes still lexes as an identifier.
            elif self.tokens[self.pos].kind == TokenKind.identifier:
                ident = self.tokens[self.pos].literal
                self.pos += 1
                return StringExpression(ident)

        if (
            TokenKind.keyword_xyz in expected_types
            and self.pos < len(self.tokens)
            and self.tokens[self.pos].kind == TokenKind.keyword_xyz
        ):
            return self.parse_xyz()

        valid_types: list[TokenKind] = [
            kind
            for kind in expected_types
            if isinstance(kind, TokenKind) and kind in [TokenKind.number, TokenKind.string, TokenKind.percent]
        ]

        if not valid_types:
            self.err(self.tokens[self.pos], f"Expected {describe_any(expected_types)}")

        tok: Token = self.expect_consume_any(valid_types)

        match tok.kind:
            case TokenKind.number:
                return NumberExpression(tok.value)

            case TokenKind.percent:
                return NumberExpression(tok.value)

            case TokenKind.string:
                return StringExpression(tok.value)

            case _:
                self.err(tok, f"Expected a value, got {describe(tok.kind)}")

    def parse_numeric_stat_expression(self, token_kind: TokenKind, player_selector: PlayerSelector) -> Expression:
        """Parse a stat and its comparison."""
        self.pos += 1

        if self.pos < len(self.tokens) and self.tokens[self.pos].kind == TokenKind.keyword_isbetween:
            return self._handle_between_comparison(token_kind, player_selector)

        if self.pos < len(self.tokens) and self.tokens[self.pos].kind in [
            TokenKind.greater,
            TokenKind.less,
            TokenKind.equals,
        ]:
            return self._handle_explicit_comparison(token_kind, player_selector)

        return self._handle_implicit_comparison(token_kind, player_selector)

    def get_stat_eval_expression(self, token_kind: TokenKind, is_percent: bool) -> Expression:
        """The value a stat token reads."""
        if token_kind in [
            TokenKind.command_expr_health,
            TokenKind.command_expr_health_above,
            TokenKind.command_expr_health_below,
        ]:
            if is_percent:
                return DivideExpression(Eval(EvalKind.health), Eval(EvalKind.max_health))
            else:
                return Eval(EvalKind.health)

        elif token_kind in [
            TokenKind.command_expr_mana,
            TokenKind.command_expr_mana_above,
            TokenKind.command_expr_mana_below,
        ]:
            if is_percent:
                return DivideExpression(Eval(EvalKind.mana), Eval(EvalKind.max_mana))
            else:
                return Eval(EvalKind.mana)

        elif token_kind in [
            TokenKind.command_expr_energy,
            TokenKind.command_expr_energy_above,
            TokenKind.command_expr_energy_below,
        ]:
            if is_percent:
                return DivideExpression(Eval(EvalKind.energy), Eval(EvalKind.max_energy))
            else:
                return Eval(EvalKind.energy)

        elif token_kind in [
            TokenKind.command_expr_bagcount,
            TokenKind.command_expr_bagcount_above,
            TokenKind.command_expr_bagcount_below,
        ]:
            if is_percent:
                return DivideExpression(Eval(EvalKind.bagcount), Eval(EvalKind.max_bagcount))
            else:
                return Eval(EvalKind.bagcount)

        elif token_kind in [
            TokenKind.command_expr_gold,
            TokenKind.command_expr_gold_above,
            TokenKind.command_expr_gold_below,
        ]:
            if is_percent:
                return DivideExpression(Eval(EvalKind.gold), Eval(EvalKind.max_gold))
            else:
                return Eval(EvalKind.gold)

        elif token_kind == TokenKind.command_expr_account_level:
            return Eval(EvalKind.account_level)

        elif token_kind in [
            TokenKind.command_expr_potion_count,
            TokenKind.command_expr_potion_countbelow,
            TokenKind.command_expr_potion_countabove,
        ]:
            if is_percent:
                return DivideExpression(Eval(EvalKind.potioncount), Eval(EvalKind.max_potioncount))
            else:
                return Eval(EvalKind.potioncount)

        elif token_kind in [
            TokenKind.command_expr_playercount,
            TokenKind.command_expr_playercountabove,
            TokenKind.command_expr_playercountbelow,
        ]:
            return Eval(EvalKind.playercount)

        elif token_kind == TokenKind.command_expr_window_text:
            return Eval(EvalKind.windowtext, [self.parse_value(["window_path"])])

        elif token_kind == TokenKind.command_expr_window_num:
            return Eval(EvalKind.windownum, [self.parse_value(["window_path"])])

        elif token_kind == TokenKind.command_expr_duel_round:
            return Eval(EvalKind.duel_round)

        else:
            self.err(self.tokens[self.pos - 1], f"Unexpected {describe(token_kind)}")

    def _handle_between_comparison(self, token_kind: TokenKind, player_selector: PlayerSelector) -> Expression:
        """Parse `isbetween`, taking either two bounds or one range value."""
        self.pos += 1

        bound_kinds: list[TokenKind | str] = [TokenKind.number, TokenKind.percent, TokenKind.identifier]
        first: Expression = self.parse_value(bound_kinds + [TokenKind.string])

        # A second bound means both were written. Otherwise the first names a range.
        if self.pos < len(self.tokens) and self.tokens[self.pos].kind in bound_kinds:
            max_value: Expression = self.parse_value(bound_kinds)
            min_value: Expression = first

            # Parsing turns a percent into a number. The tokens decide about fractions.
            is_percent: bool = (
                isinstance(min_value, NumberExpression) and self.tokens[self.pos - 2].kind == TokenKind.percent
            ) or (isinstance(max_value, NumberExpression) and self.tokens[self.pos - 1].kind == TokenKind.percent)

        else:
            min_value: Expression = RangeMinExpression(first)
            max_value: Expression = RangeMaxExpression(first)
            is_percent: bool = False

        evaluated: Expression = self.get_stat_eval_expression(token_kind, is_percent)

        min_expr: Expression = self.gen_greater_equal_expression(evaluated, min_value, player_selector)
        max_expr: Expression = self.gen_greater_equal_expression(max_value, evaluated, player_selector)

        return AndExpression([min_expr, max_expr])

    def _handle_explicit_comparison(self, token_kind: TokenKind, player_selector: PlayerSelector) -> Expression:
        """Parse a stat compared with an explicit operator."""
        operator: Token = self.tokens[self.pos]
        self.pos += 1

        target: Expression = self.parse_value([TokenKind.number, TokenKind.percent, TokenKind.identifier])
        evaluated: Expression = self.get_stat_eval_expression(token_kind, False)

        if operator.kind == TokenKind.greater:
            return self.gen_greater_expression(evaluated, target, player_selector)

        elif operator.kind == TokenKind.less:
            return self.gen_greater_expression(target, evaluated, player_selector)

        else:
            return self.gen_equivalent_expression(evaluated, target, player_selector)

    def _handle_implicit_comparison(self, token_kind: TokenKind, player_selector: PlayerSelector) -> Expression:
        """Parse a bare value to compare."""
        value_expr: Expression = self.parse_value([TokenKind.number, TokenKind.percent])

        if not isinstance(value_expr, NumberExpression):
            self.err(self.tokens[self.pos - 1], f"Expected number or percent, got {value_expr}")

        is_percent: bool = self.tokens[self.pos - 1].kind == TokenKind.percent
        evaluated: Expression = self.get_stat_eval_expression(token_kind, is_percent)

        # With no operator written, the command spelling decides the direction.
        above_tokens: list[TokenKind] = [
            TokenKind.command_expr_health_above,
            TokenKind.command_expr_mana_above,
            TokenKind.command_expr_energy_above,
            TokenKind.command_expr_bagcount_above,
            TokenKind.command_expr_gold_above,
            TokenKind.command_expr_potion_countabove,
            TokenKind.command_expr_playercountabove,
        ]

        below_tokens: list[TokenKind] = [
            TokenKind.command_expr_health_below,
            TokenKind.command_expr_mana_below,
            TokenKind.command_expr_energy_below,
            TokenKind.command_expr_bagcount_below,
            TokenKind.command_expr_gold_below,
            TokenKind.command_expr_potion_countbelow,
            TokenKind.command_expr_playercountbelow,
        ]

        if token_kind in above_tokens:
            return self.gen_greater_expression(evaluated, value_expr, player_selector)

        elif token_kind in below_tokens:
            return self.gen_greater_expression(value_expr, evaluated, player_selector)

        else:
            return self.gen_equivalent_expression(evaluated, value_expr, player_selector)

    def parse_command_expression(self) -> Expression:
        """Parse a command as a condition."""
        player_selector: PlayerSelector = self.parse_player_selector()

        # Asks each client the opposite, unlike a leading `not`.
        negation: Token | None = self.consume_optional(TokenKind.keyword_not)
        player_selector.negated = negation is not None

        # An identifier followed by = tests a constant's value rather than naming a command.
        if self.pos < len(self.tokens) and self.tokens[self.pos].kind == TokenKind.identifier:
            ident: str = self.tokens[self.pos].literal
            self.pos += 1

            if self.pos < len(self.tokens) and self.tokens[self.pos].kind == TokenKind.equals:
                if negation is not None:
                    self.err(negation, f"Write `not {ident} = ...` instead, since a constant has no client to ask")

                self.pos += 1

                if self.pos < len(self.tokens):
                    if self.tokens[self.pos].kind == TokenKind.boolean_true:
                        token: Token = self.tokens[self.pos]
                        self.pos += 1
                        return ConstantCheckExpression(
                            ident, ConstantExpression(token.literal, StringExpression("true"))
                        )

                    elif self.tokens[self.pos].kind == TokenKind.boolean_false:
                        token: Token = self.tokens[self.pos]
                        self.pos += 1
                        return ConstantCheckExpression(
                            ident, ConstantExpression(token.literal, StringExpression("false"))
                        )

                value: Expression = self.parse_expression()
                return ConstantCheckExpression(ident, value)

            # No = followed. Give the identifier back to the rest of this method.
            else:
                self.pos -= 1

        spec: ExprSpec | None = EXPR_COMMAND_REGISTRY.get(self.tokens[self.pos].kind)

        if spec is not None:
            return spec.parse(self, player_selector, spec.token)

        # Only a check reads the selector.
        if negation is not None:
            self.err(negation, "`not` here needs a check after it, or write it before the selector")

        return self.parse_unary_expression()

    def _token_for(self, expr: Expression, start: int, end: int) -> Token:
        """The token that wrote a value."""
        for pos in range(start, min(end, len(self.tokens))):
            token: Token = self.tokens[pos]

            if isinstance(expr, IdentExpression) and token.kind == TokenKind.identifier and token.literal == expr.ident:
                return token

            if isinstance(expr, NumberExpression) and token.value == expr.number:
                return token

            if isinstance(expr, StringExpression) and token.kind == TokenKind.string and token.value == expr.string:
                return token

        return self.tokens[start]

    def _reject_bare_word(self, expr: Expression, start: int, end: int) -> None:
        """Refuse a non-condition here."""
        if isinstance(expr, IdentExpression):
            self.err(self._token_for(expr, start, end), f"Unknown condition: {expr.ident}")

        # A literal decides the branch before the game is ever asked.
        if isinstance(expr, (NumberExpression, StringExpression)):
            literal: Token = self._token_for(expr, start, end)
            self.err(literal, f"Expected a condition, got the value {literal.literal}")

        if isinstance(expr, ListExpression):
            self.err(self._token_for(expr, start, end), "Expected a condition, got a list")

        if isinstance(expr, XYZExpression):
            self.err(self._token_for(expr, start, end), "Expected a condition, got a position")

        # Only operators that keep their operands in condition position.
        if isinstance(expr, (AndExpression, OrExpression)):
            for part in expr.expressions:
                self._reject_bare_word(part, start, end)

        elif isinstance(expr, UnaryExpression):
            self._reject_bare_word(expr.expr, start, end)

    def parse_condition(self) -> Expression:
        """Parse a branch condition."""
        start: int = self.pos
        expr: Expression = self.parse_expression()

        # A bare word names no condition. It reads as itself, which always holds.
        self._reject_bare_word(expr, start, self.pos)
        return expr

    def parse_negation_expression(self) -> Expression:
        """Parse a `not`, then a command."""
        kinds: list[TokenKind] = [TokenKind.keyword_not]

        if self.tokens[self.pos].kind in kinds:
            self.expect_consume_any(kinds)
            return UnaryExpression(UnaryOp.not_, self.parse_command_expression())

        else:
            return self.parse_command_expression()

    def parse_and_expression(self) -> Expression:
        """Parse conditions joined by `and`."""
        expr: Expression = self.parse_negation_expression()
        after_any: bool = asks_any_player(expr)

        # Flat list, so no operand hides from narrowing.
        parts: list[Expression] = _joined_parts(expr, AndExpression)

        while self.pos < len(self.tokens) and self.tokens[self.pos].kind == TokenKind.keyword_and:
            self.pos += 1
            right: Expression = self.parse_negation_expression()

            # No selector after an `any` means the same clients.
            if after_any:
                right = _narrow_to_any(right)

            after_any = after_any or asks_any_player(right)
            parts.extend(_joined_parts(right, AndExpression))

        return parts[0] if len(parts) == 1 else AndExpression(parts)

    def parse_logical_expression(self) -> Expression:
        """Parse conditions joined by `or`."""
        expr: Expression = self.parse_and_expression()
        after_any: bool = asks_any_player(expr)
        parts: list[Expression] = _joined_parts(expr, OrExpression)

        while self.pos < len(self.tokens) and self.tokens[self.pos].kind == TokenKind.keyword_or:
            self.pos += 1
            right: Expression = self.parse_and_expression()

            # No selector after an `any` means the same clients.
            if after_any:
                right = _narrow_to_any(right)

            after_any = after_any or asks_any_player(right)
            parts.extend(_joined_parts(right, OrExpression))

        return parts[0] if len(parts) == 1 else OrExpression(parts)

    def parse_expression(self) -> Expression:
        """Parse a full expression."""
        if self.pos < len(self.tokens) and self.tokens[self.pos].kind == TokenKind.logical_and:
            self.err(self.tokens[self.pos], "Expected an expression before &&")

        return self.parse_logical_expression()

    def parse_player_selector(self) -> PlayerSelector:
        """Parse a command's client selector."""
        result: PlayerSelector = PlayerSelector()
        valid_toks: list[TokenKind] = [
            TokenKind.keyword_same_any,
            TokenKind.keyword_any_player,
            TokenKind.keyword_mass,
            TokenKind.keyword_except,
            TokenKind.player_num,
            TokenKind.colon,
        ]
        expected_toks: list[TokenKind] = [
            TokenKind.keyword_same_any,
            TokenKind.keyword_any_player,
            TokenKind.keyword_mass,
            TokenKind.keyword_except,
            TokenKind.player_num,
        ]

        # expected_toks narrows as the selector is read. That rejects `p1 mass`.
        while self.pos < len(self.tokens) and self.tokens[self.pos].kind in valid_toks:
            if self.tokens[self.pos].kind not in expected_toks:
                self.err(
                    self.tokens[self.pos],
                    f"Invalid player selector: {describe(self.tokens[self.pos].kind)} does not belong here",
                )

            match self.tokens[self.pos].kind:
                case TokenKind.keyword_same_any:
                    result.same_any = True
                    expected_toks = []
                    self.pos += 1

                case TokenKind.keyword_any_player:
                    result.any_player = True
                    expected_toks = []
                    self.pos += 1

                case TokenKind.keyword_mass:
                    result.mass = True
                    expected_toks = []
                    self.pos += 1

                case TokenKind.keyword_except:
                    result.inverted = True
                    expected_toks = [TokenKind.player_num]
                    self.pos += 1

                case TokenKind.player_num:
                    result.player_nums.append(int(self.tokens[self.pos].value))
                    expected_toks = [TokenKind.colon]
                    self.pos += 1

                case TokenKind.colon:
                    expected_toks = [TokenKind.player_num]
                    self.pos += 1

                case _:
                    self.err(
                        self.tokens[self.pos],
                        f"Invalid player selector: {describe(self.tokens[self.pos].kind)} does not belong here",
                    )

        try:
            result.validate()
        except SelectorError as error:
            self.err(self.tokens[self.pos - 1], f"Invalid player selector: {error}")

        # A command written without a selector applies to every client.
        if len(result.player_nums) == 0 and not result.mass and not result.any_player and not result.same_any:
            result.mass = True
            result.implicit = True

        return result

    def parse_key(self) -> KeyExpression:
        """Parse a keyboard key name."""
        tok: Token = self.expect_consume_any([TokenKind.identifier, TokenKind.command_kill])
        return KeyExpression(tok.literal)

    def parse_xyz(self) -> XYZExpression:
        """Parse an `XYZ(x, y, z)` coordinate."""
        start_tok: Token = self.expect_consume(TokenKind.keyword_xyz)
        vals: list[Expression] = []
        valid_toks: list[TokenKind] = [
            TokenKind.paren_open,
            TokenKind.paren_close,
            TokenKind.comma,
            TokenKind.number,
            TokenKind.minus,
        ]
        expected_toks: list[TokenKind] = [TokenKind.paren_open]
        found_closing: bool = False

        while self.pos < len(self.tokens) and self.tokens[self.pos].kind in valid_toks:
            if self.tokens[self.pos].kind not in expected_toks:
                self.err(self.tokens[self.pos], "Invalid xyz encountered")

            match self.tokens[self.pos].kind:
                case TokenKind.paren_open:
                    self.pos += 1
                    expected_toks = [
                        TokenKind.comma,
                        TokenKind.number,
                        TokenKind.paren_close,
                        TokenKind.minus,
                    ]

                case TokenKind.paren_close:
                    self.pos += 1
                    expected_toks = []
                    found_closing = True

                case TokenKind.comma | TokenKind.number | TokenKind.minus:
                    # A skipped coordinate reads as 0, so XYZ(, , 10) only sets z.
                    if self.tokens[self.pos].kind == TokenKind.comma:
                        vals.append(NumberExpression(0.0))
                        self.pos += 1
                    else:
                        vals.append(self.parse_expression())

                        if self.tokens[self.pos].kind == TokenKind.comma:
                            self.pos += 1

                    expected_toks = [
                        TokenKind.comma,
                        TokenKind.paren_close,
                        TokenKind.number,
                        TokenKind.minus,
                    ]

        if not found_closing:
            self.err(start_tok, "Encountered unclosed XYZ")

        if len(vals) != 3:
            self.err(start_tok, "Encountered invalid XYZ")

        return XYZExpression(*vals)

    def parse_completion_optional(self) -> bool:
        """Consume a trailing `completion`."""
        if self.pos < len(self.tokens) and self.tokens[self.pos].kind == TokenKind.keyword_completion:
            self.pos += 1
            return True

        return False

    def parse_zone_path_optional(self) -> list[str] | None:
        """Parse a zone path, if any."""
        if self.pos < len(self.tokens) and self.tokens[self.pos].kind == TokenKind.path:
            result: Token = self.tokens[self.pos]
            self.pos += 1
            return result.value

        return None

    def parse_zone_path(self) -> list[str]:
        """Parse a required zone path."""
        res: list[str] | None = self.parse_zone_path_optional()

        if res is None:
            self.err(
                self.tokens[self.pos] if self.pos < len(self.tokens) else self.tokens[-1], "Failed to parse zone path"
            )

        return res

    def parse_zone_path_expression(self) -> Expression:
        """Parse a zone path as an expression."""
        self.pos += 1
        path_str: str = self.tokens[self.pos - 1].literal
        return StringExpression(path_str)

    def parse_list(self) -> ListExpression:
        """Parse a bracketed, comma-separated list."""
        self.expect_consume(TokenKind.square_open)
        items: list[Expression] = []

        while self.pos < len(self.tokens) and self.tokens[self.pos].kind != TokenKind.square_close:
            if self.tokens[self.pos].kind == TokenKind.comma:
                self.pos += 1
                continue

            items.append(self.parse_expression())

            if self.pos < len(self.tokens) and self.tokens[self.pos].kind != TokenKind.square_close:
                self.expect_consume(TokenKind.comma)

        self.expect_consume(TokenKind.square_close)
        return ListExpression(items)

    def parse_window_path(self) -> list[str] | Expression:
        """Parse a window path, either a list or a constant reference."""
        # A $constant may hold the whole path, so the VM resolves it instead of the parser.
        if (
            self.pos < len(self.tokens)
            and self.tokens[self.pos].kind == TokenKind.identifier
            and self.tokens[self.pos].literal.startswith("$")
        ):
            ident: str = self.tokens[self.pos].literal
            self.pos += 1
            return ConstantReferenceExpression(ident[1:])

        list_expr: ListExpression = self.parse_list()
        result: list[str] = []

        if len(list_expr.items) == 0:
            self.err(self.tokens[self.pos - 1], "Expected at least one window name in the path")

        for part in list_expr.items:
            if not isinstance(part, StringExpression):
                self.err(self.tokens[self.pos - 1], f"Expected a window name, got {describe_expression(part)}")

            result.append(part.string)

        return result

    def end_line(self) -> None:
        """Require the end of the line."""
        # && carries on to another command, so the statement has not ended yet.
        if self.pos < len(self.tokens) and self.tokens[self.pos].kind == TokenKind.logical_and:
            return

        self.expect_consume(TokenKind.END_LINE)

    def end_line_optional(self) -> None:
        """Consume a line end, if there."""
        if self.pos < len(self.tokens) and self.tokens[self.pos].kind == TokenKind.END_LINE:
            self.pos += 1

    def parse_command(self) -> Command | ParallelCommandStmt:
        """Parse one command, or several joined with `&&`."""
        commands: list[Command] = []
        commands.append(self._parse_simple_command())

        while self.pos < len(self.tokens) and self.tokens[self.pos].kind == TokenKind.logical_and:
            self.pos += 1
            commands.append(self._parse_simple_command())

        if len(commands) == 1:
            return commands[0]

        return ParallelCommandStmt(commands)

    def _parse_simple_command(self) -> Command:
        """Parse one command via its registry spec."""
        result: Command = Command()
        result.line_info = self.tokens[self.pos].line_info
        result.player_selector = self.parse_player_selector()

        spec: CommandSpec | None = COMMAND_REGISTRY.get(self.tokens[self.pos].kind)
        if spec is None:
            self.err(self.tokens[self.pos], "Unhandled command token")

        # Some commands hit the whole game. A selector on them would be a lie.
        if spec.every_client and not result.player_selector.mass:
            self.err(
                self.tokens[self.pos],
                f"{self.tokens[self.pos].literal} applies to every client, so it cannot take a player selector",
            )

        result.kind = spec.kind
        self.pos += 1
        result.data = spec.parse_args(self)
        self.end_line()
        return result

    def parse_block(self) -> StmtList:
        """Parse a braced block body."""
        inner: list[Stmt] = []
        self.expect_consume(TokenKind.curly_open)
        self.end_line_optional()

        while self.pos < len(self.tokens) and self.tokens[self.pos].kind != TokenKind.curly_close:
            inner.append(self.parse_stmt())

        self.expect_consume(TokenKind.curly_close)
        self.end_line_optional()
        return StmtList(inner)

    def consume_any_ident(self) -> IdentExpression:
        """Consume an identifier, any spelling."""
        result: Token = self.tokens[self.pos]

        if (
            result.kind != TokenKind.identifier
            and "keyword" not in result.kind.name
            and "command" not in result.kind.name
        ):
            self.err(result, "Unable to consume an identifier")

        self.pos += 1
        return IdentExpression(result.literal)

    def parse_stmt(self) -> Stmt:
        """Parse one statement, line stamped."""
        line_info: LineInfo = self.tokens[self.pos].line_info
        result: Stmt = self._parse_stmt_inner()

        # An arm that stamped its own knows better.
        if result.line_info is None:
            result.line_info = line_info

        return result

    def _parse_stmt_inner(self) -> Stmt:
        """Parse one statement."""
        match self.tokens[self.pos].kind:
            case TokenKind.keyword_con:
                self.pos += 1
                var_name: str = self.expect_consume(TokenKind.identifier).literal
                self.expect_consume(TokenKind.equals)
                expr: Expression = self.parse_expression()
                self.end_line()
                return ConstantDeclStmt(var_name, expr)

            case TokenKind.keyword_starttimer | TokenKind.keyword_resettimer | TokenKind.keyword_endtimer:
                timer_action: TimerAction = _TIMER_ACTIONS[self.tokens[self.pos].kind]
                self.pos += 1
                timer_name: IdentExpression = self.consume_any_ident()
                self.end_line()
                return TimerStmt(timer_action, timer_name.ident)

            case (
                TokenKind.keyword_startcounter
                | TokenKind.keyword_resetcounter
                | TokenKind.keyword_endcounter
                | TokenKind.keyword_addone
                | TokenKind.keyword_minusone
            ):
                action: CounterAction = _COUNTER_ACTIONS[self.tokens[self.pos].kind]
                self.pos += 1
                counter_name: IdentExpression = self.consume_any_ident()
                self.end_line()
                return CounterStmt(action, counter_name.ident)

            case TokenKind.keyword_block:
                self.pos += 1
                ident: IdentExpression = self.consume_any_ident()
                body: StmtList = self.parse_block()
                return BlockDefStmt(ident, body)

            case TokenKind.keyword_call:
                self.pos += 1
                ident: IdentExpression = self.consume_any_ident()
                self.end_line()
                return CallStmt(ident)

            case TokenKind.keyword_loop:
                self.pos += 1
                body: StmtList = self.parse_block()
                return LoopStmt(body)

            case TokenKind.keyword_while:
                self.pos += 1
                expr: Expression = self.parse_condition()
                body: StmtList = self.parse_block()
                return WhileStmt(expr, body)

            case TokenKind.keyword_until:
                self.pos += 1
                expr: Expression = self.parse_condition()
                body: StmtList = self.parse_block()
                return UntilStmt(expr, body)

            case TokenKind.keyword_times:
                self.pos += 1
                count_tok: Token = self.expect_consume(TokenKind.number)

                if count_tok.value != int(count_tok.value):
                    self.err(count_tok, f"Expected a whole number of repetitions, got {count_tok.value}")

                body: StmtList = self.parse_block()
                return TimesStmt(int(count_tok.value), body)

            case TokenKind.keyword_if:
                self.pos += 1
                expr: Expression = self.parse_condition()
                true_body: StmtList = self.parse_block()
                elif_body_stack: list[IfStmt] = []
                else_body: StmtList = StmtList([])

                while self.pos < len(self.tokens) and self.tokens[self.pos].kind in [
                    TokenKind.keyword_else,
                    TokenKind.keyword_elif,
                ]:
                    if self.tokens[self.pos].kind == TokenKind.keyword_else:
                        self.pos += 1
                        else_body = self.parse_block()
                        break

                    # Each elif becomes the else branch of the one before it, nesting the chain.
                    elif self.tokens[self.pos].kind == TokenKind.keyword_elif:
                        elif_line_info: LineInfo = self.tokens[self.pos].line_info
                        self.pos += 1
                        elif_expr: Expression = self.parse_condition()
                        elif_body: StmtList = self.parse_block()
                        elif_stmt: IfStmt = IfStmt(elif_expr, elif_body, StmtList([]))

                        # Never goes through `parse_stmt`, so stamp it here.
                        elif_stmt.line_info = elif_line_info

                        if len(elif_body_stack) > 0:
                            elif_body_stack[-1].branch_false = StmtList([elif_stmt])

                        elif_body_stack.append(elif_stmt)

                # The real else lands on the innermost elif. The outermost becomes ours.
                if len(elif_body_stack) > 0:
                    elif_body_stack[-1].branch_false = else_body
                    else_body = StmtList([elif_body_stack[0]])

                return IfStmt(expr, true_body, else_body)

            case TokenKind.keyword_break:
                self.pos += 1
                self.end_line()
                return BreakStmt()

            case TokenKind.keyword_return:
                self.pos += 1
                self.end_line()
                return ReturnStmt()

            case TokenKind.keyword_mixin:
                self.pos += 1
                ident = self.consume_any_ident()
                self.end_line()
                return MixinStmt(ident.ident)

            # No keyword opened the line, so it must be a command.
            case _:
                return CommandStmt(self.parse_command())

    def parse(self) -> list[Stmt]:
        """Parse the whole program."""
        result: list[Stmt] = []

        while self.pos < len(self.tokens):
            result.append(self.parse_stmt())

        return result
