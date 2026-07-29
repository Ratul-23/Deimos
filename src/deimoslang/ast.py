"""Every node a program parses into."""

from enum import Enum, auto
from typing import Any


class CommandKind(Enum):
    """What a statement command does."""

    invalid = auto()

    expr = auto()

    kill = auto()
    sleep = auto()
    log = auto()
    teleport = auto()
    goto = auto()
    sendkey = auto()
    waitfor = auto()
    usepotion = auto()
    buypotions = auto()
    relog = auto()
    click = auto()
    tozone = auto()
    load_playstyle = auto()
    set_yaw = auto()
    setdeck = auto()
    getdeck = auto()
    select_friend = auto()
    autopet = auto()
    set_goal = auto()
    set_quest = auto()
    set_zone = auto()
    toggle_combat = auto()
    restart_bot = auto()
    cursor = auto()


class TeleportKind(Enum):
    """Which destination form a teleport uses."""

    position = auto()
    friend_icon = auto()
    friend_name = auto()
    entity_vague = auto()
    entity_literal = auto()
    mob = auto()
    quest = auto()
    client_num = auto()
    nav = auto()
    plusteleport = auto()
    minusteleport = auto()


class EvalKind(Enum):
    """One value read off a client."""

    health = auto()
    max_health = auto()
    mana = auto()
    max_mana = auto()
    energy = auto()
    max_energy = auto()
    bagcount = auto()
    max_bagcount = auto()
    gold = auto()
    max_gold = auto()
    potioncount = auto()
    max_potioncount = auto()
    playercount = auto()

    windowtext = auto()
    windownum = auto()

    any_player_list = auto()
    account_level = auto()
    duel_round = auto()


class WaitforKind(Enum):
    """What a `waitfor` waits on."""

    dialog = auto()
    battle = auto()
    zonechange = auto()
    free = auto()
    window = auto()


class CursorKind(Enum):
    """What a cursor move targets."""

    position = auto()
    window = auto()


class ClickKind(Enum):
    """What a click targets."""

    position = auto()
    window = auto()


class LogKind(Enum):
    """One value, or one per client."""

    multi = auto()
    single = auto()


class UnaryOp(Enum):
    """A unary operator."""

    negate = auto()
    not_ = auto()


class ExprKind(Enum):
    """Which condition an expression command checks."""

    window_visible = auto()
    window_disabled = auto()

    in_zone = auto()
    in_combat = auto()
    in_range = auto()

    same_zone = auto()
    same_quest = auto()
    same_xyz = auto()
    same_yaw = auto()
    same_place = auto()

    has_dialogue = auto()
    has_xyz = auto()
    has_quest = auto()
    has_yaw = auto()

    tracking_quest = auto()
    tracking_goal = auto()
    quest_changed = auto()
    goal_changed = auto()
    zone_changed = auto()

    loading = auto()
    items_dropped = auto()


class TimerAction(Enum):
    """Whether a timer statement starts or ends a timer."""

    start = auto()
    end = auto()


class SymbolKind(Enum):
    """What a symbol names."""

    variable = auto()
    block = auto()
    label = auto()


class SelectorError(Exception):
    """A selector combines options that clash."""


class PlayerSelector:
    """Which clients a command applies to."""

    def __init__(self) -> None:
        self.player_nums: list[int] = []
        self.mass: bool = False
        self.inverted: bool = False
        self.any_player: bool = False
        self.same_any: bool = False

    def validate(self) -> None:
        """Check the selector, then sort it."""
        if self.mass and self.inverted:
            raise SelectorError("mass cannot be combined with except")

        if self.mass and len(self.player_nums) > 0:
            raise SelectorError("mass cannot name individual clients")

        if self.inverted and len(self.player_nums) == 0:
            raise SelectorError("except needs at least one client to leave out")

        if self.any_player and (self.mass or len(self.player_nums) > 0):
            raise SelectorError("anyplayer cannot be combined with mass or named clients")

        if self.same_any and (self.mass or len(self.player_nums) > 0):
            raise SelectorError("sameany cannot be combined with mass or named clients")

        # Sorting makes p2,p1 and p1,p2 the same selector.
        self.player_nums.sort()

    def __repr__(self) -> str:
        return (
            f"PlayerSelector(nums: {self.player_nums}, mass: {self.mass}, inverted: {self.inverted}, "
            f"any_player: {self.any_player}, same_any: {self.same_any})"
        )


class Command:
    """A parsed command."""

    def __init__(self) -> None:
        self.kind: CommandKind = CommandKind.invalid
        self.data: list[Any] = []
        self.player_selector: PlayerSelector | None = None

    def __repr__(self) -> str:
        params_str: str = ", ".join([str(item) for item in self.data])

        if self.player_selector is None:
            return f"{self.kind.name}({params_str})"
        else:
            return f"{self.kind.name}({params_str}) @ {self.player_selector}"


class Expression:
    """Base for anything with a value."""


class ConstantExpression(Expression):
    """A named constant paired with its value."""

    def __init__(self, name: str, value: Expression) -> None:
        self.name: str = name
        self.value: Expression = value

    def __repr__(self) -> str:
        return f"ConstE({self.name}, {self.value})"


class ListExpression(Expression):
    """A bracketed list of expressions."""

    def __init__(self, items: list[Expression]) -> None:
        self.items: list[Expression] = items

    def __repr__(self) -> str:
        return f"ListE({self.items})"


class NumberExpression(Expression):
    """A numeric literal."""

    def __init__(self, number: float) -> None:
        self.number: float = number

    def __repr__(self) -> str:
        return f"Number({self.number})"


class StringExpression(Expression):
    """A string literal."""

    def __init__(self, string: str) -> None:
        self.string: str = string

    def __repr__(self) -> str:
        return f"String({self.string})"


class StrFormatExpression(Expression):
    """A printf template and its values."""

    def __init__(self, format_str: str, *args: Expression) -> None:
        self.format_str: str = format_str
        self.values: tuple[Expression, ...] = args

    def __repr__(self) -> str:
        return f"StrFormat({self.format_str}, {self.values})"


class UnaryExpression(Expression):
    """An operator applied to one operand."""

    def __init__(self, operator: UnaryOp, expr: Expression) -> None:
        self.operator: UnaryOp = operator
        self.expr: Expression = expr

    def __repr__(self) -> str:
        return f"Unary({self.operator}, {self.expr})"


class KeyExpression(Expression):
    """A keyboard key name."""

    def __init__(self, key: str) -> None:
        self.key: str = key

    def __repr__(self) -> str:
        return f"Key({self.key})"


class CommandExpression(Expression):
    """A command used as a condition."""

    def __init__(self, command: Command) -> None:
        self.command: Command = command

    def __repr__(self) -> str:
        return f"ComE({self.command})"


class XYZExpression(Expression):
    """A coordinate triple."""

    def __init__(self, x: Expression, y: Expression, z: Expression) -> None:
        self.x: Expression = x
        self.y: Expression = y
        self.z: Expression = z

    def __repr__(self) -> str:
        return f"XYZE({self.x}, {self.y}, {self.z})"


class BinaryExpression(Expression):
    """Base for two-sided operators."""

    def __init__(self, lhs: Expression, rhs: Expression) -> None:
        self.lhs: Expression = lhs
        self.rhs: Expression = rhs


class SubExpression(BinaryExpression):
    """Subtraction."""

    def __repr__(self) -> str:
        return f"SubE({self.lhs}, {self.rhs})"


class DivideExpression(BinaryExpression):
    """Division."""

    def __repr__(self) -> str:
        return f"DivideE({self.lhs}, {self.rhs})"


class EquivalentExpression(BinaryExpression):
    """Equality test."""

    def __repr__(self) -> str:
        return f"EquivalentE({self.lhs}, {self.rhs})"


class ContainsStringExpression(BinaryExpression):
    """Substring test, one string or list."""

    def __repr__(self) -> str:
        return f"ContainsStrE({self.lhs}, {self.rhs})"


class GreaterExpression(BinaryExpression):
    """Greater-than test."""

    def __repr__(self) -> str:
        return f"GreaterE({self.lhs}, {self.rhs})"


class AndExpression(Expression):
    """True when every operand is true."""

    def __init__(self, expressions: list[Expression]) -> None:
        self.expressions: list[Expression] = expressions

    def __repr__(self) -> str:
        return f"AndE({', '.join(str(expr) for expr in self.expressions)})"


class OrExpression(Expression):
    """True when any operand is true."""

    def __init__(self, expressions: list[Expression]) -> None:
        self.expressions: list[Expression] = expressions

    def __repr__(self) -> str:
        return f"OrE({', '.join(str(expr) for expr in self.expressions)})"


class ConstantReferenceExpression(Expression):
    """A `$name` reference to a constant."""

    def __init__(self, name: str) -> None:
        self.name: str = name

    def __repr__(self) -> str:
        return f"ConstRef(${self.name})"


class ConstantCheckExpression(Expression):
    """A `name = value` test against a constant."""

    def __init__(self, name: str, value: Expression) -> None:
        self.name: str = name
        self.value: Expression = value

    def __repr__(self) -> str:
        return f"ConstCheck({self.name}, {self.value})"


class RangeMinExpression(Expression):
    """A string range's low end."""

    def __init__(self, range_expr: Expression) -> None:
        self.range_expr: Expression = range_expr

    def __repr__(self) -> str:
        return f"RangeMin({self.range_expr})"


class RangeMaxExpression(Expression):
    """A string range's high end."""

    def __init__(self, range_expr: Expression) -> None:
        self.range_expr: Expression = range_expr

    def __repr__(self) -> str:
        return f"RangeMax({self.range_expr})"


class IndexAccessExpression(Expression):
    """One element of a list expression."""

    def __init__(self, expr: Expression, index: Expression) -> None:
        self.expr: Expression = expr
        self.index: Expression = index

    def __repr__(self) -> str:
        return f"IndexAccess({self.expr}[{self.index}])"


class SelectorGroup(Expression):
    """An expression scoped to clients."""

    def __init__(self, players: PlayerSelector, expr: Expression) -> None:
        self.players: PlayerSelector = players
        self.expr: Expression = expr

    def __repr__(self) -> str:
        return f"SelectorG({self.players}, {self.expr})"


class IdentExpression(Expression):
    """A bare identifier, resolved later."""

    def __init__(self, ident: str) -> None:
        self.ident: str = ident

    def __repr__(self) -> str:
        return f"IdentE({self.ident})"


class SymExpression(Expression):
    """A name resolved to a symbol."""

    def __init__(self, sym: "Symbol") -> None:
        self.sym: Symbol = sym

    def __repr__(self) -> str:
        return f"SymE({self.sym})"


class StackLocExpression(Expression):
    """A slot in the value stack."""

    def __init__(self, offset: int) -> None:
        self.offset: int = offset

    def __repr__(self) -> str:
        return f"StackLocE({self.offset})"


class ReadVarExpr(Expression):
    """Reads a compiler-generated variable."""

    def __init__(self, loc: Expression) -> None:
        self.loc: Expression = loc

        # Lowering overwrites `loc` with a slot. Keep the symbol for a repeat pass.
        self.sym: Symbol | None = loc.sym if isinstance(loc, SymExpression) else None

    def __repr__(self) -> str:
        return f"ReadVarE {self.loc}"


class Eval(Expression):
    """Reads one value off a client."""

    def __init__(self, eval_kind: EvalKind, args: list[Expression | list[str]] | None = None) -> None:
        self.kind: EvalKind = eval_kind
        self.args: list[Expression | list[str]] = [] if args is None else args

    def __repr__(self) -> str:
        return f"Eval({self.kind})"


# What to call a parsed value in an error, so a message names what was written rather than the class.
_EXPRESSION_NAMES: dict[type[Expression], str] = {
    StringExpression: "a quoted string",
    NumberExpression: "a number",
    ListExpression: "a list",
    XYZExpression: "a position",
    ConstantExpression: "a boolean",
    ConstantReferenceExpression: "a constant",
    IdentExpression: "a name",
    KeyExpression: "a key",
    Eval: "a stat",
    SubExpression: "a calculation",
    DivideExpression: "a calculation",
    IndexAccessExpression: "a numbered part of a window",
    RangeMinExpression: "the low end of a range",
    RangeMaxExpression: "the high end of a range",
    StrFormatExpression: "a formatted string",
}

# Everything that reads as true or false, which is worth naming as one category rather than nine.
_CONDITIONS: tuple[type[Expression], ...] = (
    CommandExpression,
    SelectorGroup,
    AndExpression,
    OrExpression,
    ConstantCheckExpression,
    EquivalentExpression,
    GreaterExpression,
    ContainsStringExpression,
)


def describe_expression(expr: Expression) -> str:
    """A value's name for an error."""
    # A minus sign says nothing about what follows it, and a `not` only ever fronts a condition.
    if isinstance(expr, UnaryExpression):
        return describe_expression(expr.expr) if expr.operator == UnaryOp.negate else "a condition"

    for kind, named in _EXPRESSION_NAMES.items():
        if isinstance(expr, kind):
            return named

    if isinstance(expr, _CONDITIONS):
        return "a condition"

    return "something else"


class Stmt:
    """Base for anything the compiler emits."""


class ConstantDeclStmt(Stmt):
    """A `con name = value` declaration."""

    def __init__(self, name: str, value: Expression) -> None:
        self.name: str = name
        self.value: Expression = value

    def __repr__(self) -> str:
        return f"ConstDeclS({self.name}, {self.value})"


# The compiler lowers these one after another, so the name states an intent, not a guarantee.
class ParallelCommandStmt(Stmt):
    """Commands joined with `&&`."""

    def __init__(self, commands: list[Command]) -> None:
        self.commands: list[Command] = commands

    def __repr__(self) -> str:
        return f"ParallelCommandStmt({self.commands})"


class StmtList(Stmt):
    """Statements run in order."""

    def __init__(self, stmts: list[Stmt]) -> None:
        self.stmts: list[Stmt] = stmts

    def __repr__(self) -> str:
        return "StmtList{" + "; ".join([str(item) for item in self.stmts]) + "}"


class TimerStmt(Stmt):
    """Starts or ends a named timer."""

    def __init__(self, action: TimerAction, timer_name: str) -> None:
        self.action: TimerAction = action
        self.timer_name: str = timer_name

    def __repr__(self) -> str:
        action_str: str = "settimer" if self.action == TimerAction.start else "endtimer"
        return f"TimerS {action_str} {self.timer_name}"


class CommandStmt(Stmt):
    """A command used as a statement."""

    def __init__(self, command: Command | ParallelCommandStmt) -> None:
        self.command: Command | ParallelCommandStmt = command

    def __repr__(self) -> str:
        return f"ComS({self.command})"


class IfStmt(Stmt):
    """A conditional with both branches."""

    def __init__(self, expr: Expression, branch_true: StmtList, branch_false: StmtList) -> None:
        self.expr: Expression = expr
        self.branch_true: StmtList = branch_true
        self.branch_false: StmtList = branch_false

    def __repr__(self) -> str:
        return f"IfS {self.expr} {{ {self.branch_true} }} else {{ {self.branch_false} }}"


class BreakStmt(Stmt):
    """Leaves the innermost loop."""

    def __repr__(self) -> str:
        return "BreakS"


class ReturnStmt(Stmt):
    """Leaves the enclosing block."""

    def __repr__(self) -> str:
        return "ReturnS"


class MixinStmt(Stmt):
    """Declares a block."""

    def __init__(self, name: str) -> None:
        self.name: str = name

    def __repr__(self) -> str:
        return f"MixinS {self.name}"


class LoopStmt(Stmt):
    """Repeats its body forever."""

    def __init__(self, body: StmtList) -> None:
        self.body: StmtList = body

    def __repr__(self) -> str:
        return f"LoopS {{ {self.body} }}"


class WhileStmt(Stmt):
    """Repeats while a condition holds."""

    def __init__(self, expr: Expression, body: StmtList) -> None:
        self.expr: Expression = expr
        self.body: StmtList = body

    def __repr__(self) -> str:
        return f"WhileS {self.expr} {{ {self.body} }}"


class UntilStmt(Stmt):
    """Repeats until a condition holds."""

    def __init__(self, expr: Expression, body: StmtList) -> None:
        self.expr: Expression = expr
        self.body: StmtList = body

    def __repr__(self) -> str:
        return f"UntilS {self.expr} {{ {self.body} }}"


class TimesStmt(Stmt):
    """Repeats its body a fixed number of times."""

    def __init__(self, num: int, body: StmtList) -> None:
        self.num: int = num
        self.body: StmtList = body

    def __repr__(self) -> str:
        return f"TimesS {self.num} {{ {self.body} }}"


class BlockDefStmt(Stmt):
    """Defines a named block."""

    def __init__(self, name: Expression, body: StmtList) -> None:
        self.name: Expression = name
        self.body: StmtList = body
        self.mixins: set[str] = set()

    def __repr__(self) -> str:
        return f"BlockDefS {self.name} {{ {self.body} }}"


class CallStmt(Stmt):
    """Calls a named block."""

    def __init__(self, name: Expression) -> None:
        self.name: Expression = name

    def __repr__(self) -> str:
        return f"CallS {self.name}"


class DefVarStmt(Stmt):
    """Reserves a stack slot."""

    def __init__(self, sym: "Symbol") -> None:
        self.sym: Symbol = sym

    def __repr__(self) -> str:
        return f"DefVarS {self.sym}"


class WriteVarStmt(Stmt):
    """Stores a value in a slot."""

    def __init__(self, sym: "Symbol", expr: Expression) -> None:
        self.sym: Symbol = sym
        self.expr: Expression = expr

    def __repr__(self) -> str:
        return f"WriteVarS {self.sym} = {self.expr}"


class KillVarStmt(Stmt):
    """Releases a stack slot."""

    def __init__(self, sym: "Symbol") -> None:
        self.sym: Symbol = sym

    def __repr__(self) -> str:
        return f"KillVarS {self.sym}"


class UntilRegion(Stmt):
    """The span an `until` can abort."""

    def __init__(self, expr: Expression, body: Stmt) -> None:
        self.expr: Expression = expr
        self.body: Stmt = body

    def __repr__(self) -> str:
        return f"UntilRegionS ({self.expr}) {self.body}"


class Symbol:
    """A named block, variable or label."""

    def __init__(self, literal: str, sym_id: int, kind: SymbolKind) -> None:
        self.literal: str = literal
        self.id: int = sym_id
        self.kind: SymbolKind = kind
        self.defnode: Stmt | None = None

    def __repr__(self) -> str:
        return f"{self.literal}:{self.id}_{self.kind.name}"


# This lives here rather than in vm.py because the command modules raise it, and vm.py imports them.
class VMError(Exception):
    """An instruction cannot be carried out."""


class InstructionKind(Enum):
    """Every instruction the compiler can emit."""

    kill = auto()
    sleep = auto()
    restart_bot = auto()

    log_single = auto()
    log_multi = auto()

    jump = auto()
    jump_if = auto()
    jump_ifn = auto()

    enter_until = auto()
    exit_until = auto()

    label = auto()
    ret = auto()
    call = auto()

    # Carries a whole Command for the VM to perform.
    deimos_call = auto()

    load_playstyle = auto()
    toggle_combat = auto()
    set_yaw = auto()
    setdeck = auto()
    getdeck = auto()

    push_stack = auto()
    pop_stack = auto()
    write_stack = auto()

    set_timer = auto()
    end_timer = auto()

    declare_constant = auto()

    nop = auto()


class Instruction:
    """One instruction and its kind-dependent payload."""

    def __init__(self, kind: InstructionKind, data: Any = None) -> None:
        self.kind: InstructionKind = kind
        self.data: Any = data

    def __repr__(self) -> str:
        if self.data is not None:
            return f"{self.kind.name} {self.data}"

        return f"{self.kind.name}"
