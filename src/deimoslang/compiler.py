"""Analysis, then lowering to instructions."""

import copy
from typing import Any

from .ast import (
    CROSS_CLIENT_CHECKS,
    AndExpression,
    BinaryExpression,
    BlockDefStmt,
    BreakStmt,
    CallStmt,
    Command,
    CommandExpression,
    CommandKind,
    CommandStmt,
    ConstantCheckExpression,
    ConstantDeclStmt,
    ConstantExpression,
    ConstantReferenceExpression,
    CounterAction,
    CounterStmt,
    DefVarStmt,
    Eval,
    Expression,
    GreaterExpression,
    IdentExpression,
    IfStmt,
    IndexAccessExpression,
    Instruction,
    InstructionKind,
    KeyExpression,
    KillVarStmt,
    ListExpression,
    LogKind,
    LoopStmt,
    MixinStmt,
    NumberExpression,
    OrExpression,
    PlayerSelector,
    RangeMaxExpression,
    RangeMinExpression,
    ReadVarExpr,
    ReturnStmt,
    SelectorGroup,
    StackLocExpression,
    Stmt,
    StmtList,
    StringExpression,
    SubExpression,
    Symbol,
    SymbolKind,
    SymExpression,
    TimerAction,
    TimerStmt,
    TimesStmt,
    UnaryExpression,
    UnaryOp,
    UntilRegion,
    UntilStmt,
    WhileStmt,
    WholeNumberExpression,
    WriteVarStmt,
    XYZExpression,
)
from .lexer import LineInfo, LocatedError, Tokenizer
from .parser import Parser


class SemError(LocatedError):
    """A parsed program makes no sense."""


class Scope:
    """One nesting level of names."""

    def __init__(self, parent: "Scope | None", is_block: bool) -> None:
        self.parent: Scope | None = parent
        self.syms: list[Symbol] = []
        self.mixins: set[str] = set()
        self.active_vars: list[Symbol] = []
        self.is_block: bool = is_block

    def new_block(self) -> "Scope":
        """Open a scope variables cannot escape."""
        return Scope(parent=self, is_block=True)

    def new_branch(self) -> "Scope":
        """Open a scope sharing live variables."""
        res: Scope = Scope(parent=self, is_block=False)
        res.active_vars = self.active_vars[:]
        return res

    def lookup_block_by_name(self, literal: str) -> Symbol | None:
        """Find a block symbol by name."""
        for sym in reversed(self.syms):
            if sym.kind != SymbolKind.block:
                continue

            if sym.literal == literal:
                return sym

        if self.parent is not None:
            return self.parent.lookup_block_by_name(literal)

        return None

    def is_mixin(self, literal: str) -> bool:
        """Whether this block declared the mixin."""
        cur: Scope | None = self

        # Branches reach their block's mixins. A nested block declares its own.
        while cur is not None:
            if literal in cur.mixins:
                return True
            elif cur.is_block:
                break

            cur = cur.parent

        return False

    def is_block_local_var(self, sym: Symbol) -> bool:
        """Whether a symbol is this block's."""
        cur: Scope | None = self

        while cur is not None:
            if sym in cur.syms:
                return True
            elif cur.is_block:
                break

            cur = cur.parent

        return False

    def put_sym(self, sym: Symbol) -> Symbol:
        """Add a symbol to this scope."""
        self.syms.append(sym)
        return sym

    def activate_var(self, sym: Symbol) -> None:
        """Mark a variable as live."""
        if sym in self.active_vars:
            raise SemError(f"Attempted to activate an already active variable: {sym}")

        self.active_vars.append(sym)

    def kill_var(self, sym: Symbol) -> None:
        """Mark a variable dead."""
        if not self.is_block_local_var(sym):
            raise SemError("Attempted to kill a variable that isn't local to the current block")

        if sym not in self.active_vars:
            raise SemError(f"Attempted to kill an inactive variable: {sym}")

        self.active_vars.remove(sym)


class Analyzer:
    """Resolves names and desugars loops."""

    def __init__(self, stmts: list[Stmt]) -> None:
        self.scope: Scope = Scope(parent=None, is_block=False)
        self._next_sym_id: int = 0
        self.block_defs: list[BlockDefStmt] = []
        self.stmts: list[Stmt] = stmts
        self._mixin_cache: dict[tuple[Symbol, frozenset[Symbol]], Symbol] = {}
        self._block_nesting_level: int = 0
        self._loop_nesting_level: int = 0
        self._loop_nesting_stack: list[int] = []

    def _pop_scope(self) -> Scope:
        """Return to the enclosing scope."""
        parent: Scope | None = self.scope.parent
        if parent is None:
            raise SemError("Attempted to leave the outermost scope")

        return parent

    def open_block(self) -> None:
        """Enter a block, resetting loop nesting."""
        self.scope = self.scope.new_block()
        self._loop_nesting_stack.append(self._loop_nesting_level)
        self._loop_nesting_level = 0
        self._block_nesting_level += 1

    def close_block(self) -> None:
        """Leave a block, restoring loop nesting."""
        self.scope = self._pop_scope()
        self._loop_nesting_level = self._loop_nesting_stack.pop()
        self._block_nesting_level -= 1

    def open_loop(self) -> None:
        """Enter a loop body."""
        self.scope = self.scope.new_branch()
        self._loop_nesting_level += 1

    def close_loop(self) -> None:
        """Leave a loop body."""
        self.scope = self._pop_scope()
        self._loop_nesting_level -= 1

    def gen_sym_id(self) -> int:
        """The next unique symbol id."""
        result: int = self._next_sym_id
        self._next_sym_id += 1
        return result

    def gen_block_sym(self, name: str) -> Symbol:
        """Create a symbol naming a block."""
        return self.scope.put_sym(Symbol(name, self.gen_sym_id(), SymbolKind.block))

    def gen_var_sym(self, name: str = "anonymous") -> Symbol:
        """Create a symbol naming a variable."""
        return self.scope.put_sym(Symbol(f":{name}:", self.gen_sym_id(), SymbolKind.variable))

    def def_var(self) -> Symbol:
        """Create a live variable."""
        var_sym: Symbol = self.gen_var_sym()
        self.scope.activate_var(var_sym)
        return var_sym

    def mark_var_dead(self, sym: Symbol) -> None:
        """Mark a variable dead here."""
        self.scope.kill_var(sym)

    def gen_cleanup_all_vars(self) -> StmtList:
        """Kill every variable still live here."""
        res: list[Stmt] = []

        # Killing drops from the list being walked. Walk a copy.
        for var in self.scope.active_vars[::-1]:
            self.mark_var_dead(var)
            res.append(KillVarStmt(var))

        return StmtList(res)

    def sem_body(self, body: Stmt) -> StmtList:
        """Analyse the statements inside a block."""
        result: Stmt | None = self.sem_stmt(body)
        assert isinstance(result, StmtList), f"expected a StmtList body, got {result!r}"
        return result

    def sem_required(self, stmt: Stmt) -> Stmt:
        """Analyse a statement nothing consumes."""
        result: Stmt | None = self.sem_stmt(stmt)
        assert result is not None, f"{type(stmt).__name__} was unexpectedly hoisted"
        return result

    def copy_for_mixing(self, defnode: BlockDefStmt) -> BlockDefStmt:
        """Copy a block for mixing."""
        # A queued block keeps its identity, or calls name an uncompiled copy.
        memo: dict[int, BlockDefStmt | Symbol] = {}

        for block_def in self.block_defs:
            memo[id(block_def)] = block_def

            if isinstance(block_def.name, SymExpression):
                memo[id(block_def.name.sym)] = block_def.name.sym

        return copy.deepcopy(defnode, memo)

    def mix_block(self, stmt: BlockDefStmt, cache_key: tuple[Symbol, frozenset[Symbol]]) -> None:
        """Point mixin calls at real blocks."""
        assert isinstance(stmt.name, SymExpression)
        mixed_block: BlockDefStmt = stmt
        mixed_sym: Symbol = stmt.name.sym

        def _mix_stmt(stmt: Stmt, mixins: set[str]) -> None:
            """Point one statement's mixin calls."""
            # Only statements that can hold a call.
            match stmt:
                case StmtList():
                    for inner in stmt.stmts:
                        _mix_stmt(inner, mixins)

                case CallStmt():
                    if isinstance(stmt.name, IdentExpression):
                        if stmt.name.ident in mixins:
                            sym: Symbol | None = self.scope.lookup_block_by_name(stmt.name.ident)
                            if sym is None:
                                raise SemError(f"Unable to find symbol in scope: {stmt.name.ident}")

                            if sym.defnode is not None:
                                assert isinstance(sym.defnode, BlockDefStmt)

                                if len(sym.defnode.mixins) > 0:
                                    raise SemError("Recursive mixins aren't allowed")

                            stmt.name = SymExpression(sym)

                        else:
                            raise SemError(f"Undeclared identifer during mixin stage: {stmt.name.ident}")

                    elif isinstance(stmt.name, SymExpression):
                        target_defnode: Stmt | None = stmt.name.sym.defnode

                        # The copy's own calls still name the original. They must follow the copy.
                        if target_defnode is mixed_block:
                            stmt.name = SymExpression(mixed_sym)

                        else:
                            assert isinstance(target_defnode, BlockDefStmt)

                            if len(target_defnode.mixins) > 0:
                                raise SemError("Recursive mixins aren't allowed")

                    else:
                        raise SemError(f"Invalid call target: {stmt.name}")

                case WhileStmt():
                    _mix_stmt(stmt.body, mixins)

                case IfStmt():
                    _mix_stmt(stmt.branch_true, mixins)
                    _mix_stmt(stmt.branch_false, mixins)

                case LoopStmt():
                    _mix_stmt(stmt.body, mixins)

                case UntilRegion():
                    _mix_stmt(stmt.body, mixins)

        self._mixin_cache[cache_key] = mixed_sym
        _mix_stmt(stmt.body, stmt.mixins)
        stmt.mixins = set()

    def sem_stmt(self, stmt: Stmt) -> Stmt | None:
        """Analyse one statement, keeping its line."""
        try:
            result: Stmt | None = self._sem_stmt_inner(stmt)

        except SemError as error:
            error.locate(stmt.line_info)
            raise

        # Desugared statements keep the line they replaced.
        if result is not None and result.line_info is None:
            result.line_info = stmt.line_info

        return result

    def _sem_stmt_inner(self, stmt: Stmt) -> Stmt | None:
        """Analyse one statement, return its replacement."""
        match stmt:
            case TimerStmt() | CounterStmt():
                return stmt

            case ConstantDeclStmt():
                return stmt

            case BlockDefStmt():
                if not isinstance(stmt.name, IdentExpression):
                    raise SemError("Only IdentExpression is allowed during block declaration")

                # A second definition wins silently, leaving the first body unreachable.
                if self.scope.lookup_block_by_name(stmt.name.ident) is not None:
                    raise SemError(f"Block is already defined: {stmt.name.ident}")

                sym: Symbol = self.gen_block_sym(stmt.name.ident)

                # Every block ends in a return. No spilling into the next.
                stmt.body.stmts.append(ReturnStmt())
                self.open_block()
                stmt.body = self.sem_body(stmt.body)
                stmt.mixins = self.scope.mixins
                self.close_block()
                stmt.name = SymExpression(sym)
                sym.defnode = stmt

                # A block holding mixins waits for a call to resolve them.
                if len(stmt.mixins) == 0:
                    self.block_defs.append(stmt)

                # Returning nothing drops the definition. It is emitted apart.
                return None

            case StmtList():
                res: list[Stmt] = []

                for inner in stmt.stmts:
                    if semmed := self.sem_stmt(inner):
                        res.append(semmed)

                return StmtList(res)

            case CallStmt():
                if isinstance(stmt.name, IdentExpression):
                    call_name: str = stmt.name.ident
                    target_sym: Symbol | None = self.scope.lookup_block_by_name(call_name)

                elif isinstance(stmt.name, SymExpression):
                    target_sym: Symbol | None = stmt.name.sym
                    call_name: str = target_sym.literal

                else:
                    raise SemError(f"Malformed call: {stmt}")

                if self.scope.is_mixin(call_name):
                    return stmt

                else:
                    if target_sym is None:
                        raise SemError(f"Unable to find symbol in scope: {call_name}")

                    if target_sym.defnode is not None:
                        assert isinstance(target_sym.defnode, BlockDefStmt)

                        if len(target_sym.defnode.mixins) > 0:
                            mixed_syms: dict[str, Symbol] = {}

                            for mixin_name in target_sym.defnode.mixins:
                                ms: Symbol | None = self.scope.lookup_block_by_name(mixin_name)
                                if ms is None:
                                    raise SemError(f"Unable to resolve mixin: {mixin_name}")

                                mixed_syms[mixin_name] = ms

                            # Keyed on block plus mixins, so the same pairing reuses this copy.
                            key: tuple[Symbol, frozenset[Symbol]] = (target_sym, frozenset(mixed_syms.values()))

                            if key in self._mixin_cache:
                                target_sym = self._mixin_cache[key]

                            # A fresh pairing needs its own copy. Mixing rewrites calls in place.
                            else:
                                mixed_sym: Symbol = self.gen_block_sym(f":mixed_{call_name}")
                                mixed_defnode: BlockDefStmt = self.copy_for_mixing(target_sym.defnode)
                                mixed_sym.defnode = mixed_defnode
                                mixed_defnode.name = SymExpression(mixed_sym)
                                self.mix_block(mixed_defnode, key)
                                target_sym = mixed_sym
                                self.block_defs.append(mixed_defnode)

                    stmt.name = SymExpression(target_sym)
                    return stmt

            case CommandStmt():
                return stmt

            case IfStmt():
                self.scope = self.scope.new_branch()
                stmt.branch_true = self.sem_body(stmt.branch_true)
                self.scope = self._pop_scope()
                self.scope = self.scope.new_branch()
                stmt.branch_false = self.sem_body(stmt.branch_false)
                self.scope = self._pop_scope()
                return stmt

            case LoopStmt():
                self.open_loop()
                stmt.body = self.sem_body(stmt.body)
                self.close_loop()
                return stmt

            case WhileStmt():
                self.open_loop()
                stmt.body = self.sem_body(stmt.body)
                self.close_loop()
                return stmt

            case UntilStmt():
                self.open_loop()
                body: StmtList = self.sem_body(stmt.body)
                self.close_loop()

                # `until c { body }`: skip if c holds, else loop while it does not.
                return IfStmt(
                    stmt.expr,
                    branch_true=StmtList([]),
                    branch_false=StmtList(
                        [
                            UntilRegion(
                                expr=stmt.expr,
                                body=WhileStmt(UnaryExpression(UnaryOp.not_, stmt.expr), body),
                            ),
                        ]
                    ),
                )

            # A counter on the stack counts down to zero.
            case TimesStmt():
                var_sym: Symbol = self.def_var()
                prologue: list[Stmt] = [
                    DefVarStmt(var_sym),
                    WriteVarStmt(var_sym, WholeNumberExpression(stmt.count)),
                ]
                epilogue: list[Stmt] = [
                    KillVarStmt(var_sym),
                ]
                cond: GreaterExpression = GreaterExpression(ReadVarExpr(SymExpression(var_sym)), NumberExpression(0))
                stmt.body.stmts.append(
                    WriteVarStmt(var_sym, SubExpression(ReadVarExpr(SymExpression(var_sym)), NumberExpression(1)))
                )

                expanded: StmtList = StmtList(prologue + [self.sem_required(WhileStmt(cond, stmt.body))] + epilogue)
                self.mark_var_dead(var_sym)
                return expanded

            case ReturnStmt():
                if self._block_nesting_level <= 0:
                    raise SemError("Return used outside of block scope")

                # Leaving early still releases the slots it took.
                return StmtList([self.gen_cleanup_all_vars(), stmt])

            case BreakStmt():
                if self._loop_nesting_level <= 0:
                    raise SemError("Break used outside of loop scope")

                return stmt

            case DefVarStmt() | WriteVarStmt() | KillVarStmt():
                return stmt

            case MixinStmt():
                if not self.scope.is_block:
                    raise SemError("Mixin is only allowed inside blocks")

                self.scope.mixins.add(stmt.name)

                return None

            case _:
                raise SemError(f"Unhandled statement type: {stmt}")

    def analyze_program(self) -> None:
        """Analyse the program, hoisting blocks."""
        res: list[Stmt] = []

        for stmt in self.stmts:
            if semmed := self.sem_stmt(stmt):
                res.append(semmed)

        self.stmts = res


class CompilerError(LocatedError):
    """A valid program cannot be lowered."""


class StackInfo:
    """Which stack slot each symbol holds."""

    def __init__(self) -> None:
        self.offset: int = 0
        self.slots: dict[Symbol, int] = {}

    def push(self, sym: Symbol) -> None:
        """Give a symbol the next slot."""
        self.slots[sym] = self.offset
        self.offset += 1

    def pop(self, sym: Symbol) -> None:
        """Release `sym`'s slot from the top."""
        self.offset -= 1
        if self.slots[sym] != self.offset:
            raise CompilerError(f"Attempted to pop a stack value that is not placed at the top: {sym}\n{self.slots}")

        del self.slots[sym]

    def loc(self, sym: Symbol) -> int:
        """A symbol's slot, from the top."""
        return self.slots[sym] - self.offset


class Compiler:
    """Lowers an analysed program, once."""

    def __init__(self, analyzer: Analyzer) -> None:
        self.analyzer: Analyzer = analyzer
        self._program: list[Instruction] = []
        self._stacks: list[StackInfo] = [StackInfo()]
        self._loop_label_stack: list[Symbol] = []
        self._outermost_until: int | None = None
        self._current_line: LineInfo | None = None

    @staticmethod
    def from_text(code: str, filename: str | None = None) -> "Compiler":
        """Lex, parse and analyse source text."""
        tokenizer: Tokenizer = Tokenizer()
        parser: Parser = Parser(tokenizer.tokenize(code, filename=filename))
        analyzer: Analyzer = Analyzer(parser.parse())
        analyzer.analyze_program()
        return Compiler(analyzer=analyzer)

    def enter_branch(self) -> None:
        """Start a branch, cleanup kept apart."""
        top: StackInfo = self._stacks[-1]
        new_top: StackInfo = StackInfo()
        new_top.offset = top.offset
        new_top.slots = top.slots.copy()
        self._stacks.append(new_top)

    def exit_branch(self) -> None:
        """End a branch."""
        self._stacks.pop()

    def stack_loc(self, sym: Symbol) -> int:
        """Find a symbol's stack offset."""
        # Inner frames copy their parent's slots. Only the innermost holds the depth.
        for info in reversed(self._stacks):
            if sym in info.slots:
                return info.loc(sym)

        raise CompilerError(f"Failed to determine the stack location for symbol {sym}")

    def emit(self, kind: InstructionKind, data: Any = None) -> None:
        """Append one instruction with its line."""
        self._program.append(Instruction(kind, data, self._current_line))

    def gen_label(self, name: str = "anonymous") -> Symbol:
        """Create a fresh jump label."""
        return Symbol(f":{name}", self.analyzer.gen_sym_id(), SymbolKind.label)

    def emit_deimos_call(self, com: Command) -> None:
        """Emit a command for the VM."""
        self.emit(InstructionKind.deimos_call, [com.player_selector, com.kind.name, com.data])

    def compile_command(self, com: Command) -> None:
        """Emit code for one command."""
        match com.kind:
            case CommandKind.restart_bot:
                self.emit(InstructionKind.restart_bot)

            case CommandKind.toggle_combat:
                self.emit(InstructionKind.toggle_combat, com.data)

            case CommandKind.kill:
                self.emit(InstructionKind.kill)

            case CommandKind.sleep:
                self.emit(InstructionKind.sleep, com.data[0])

            case CommandKind.log:
                kind: LogKind = com.data[0]

                match kind:
                    case LogKind.multi:
                        self.emit(InstructionKind.log_multi, [com.player_selector, com.data[1]])

                    # Nothing is read off a client. A selector would print it twice.
                    case LogKind.single:
                        if com.player_selector is not None and not com.player_selector.mass:
                            raise CompilerError(
                                "This log line reads nothing off a client, so it cannot take a player selector"
                            )

                        self.emit(InstructionKind.log_single, com.data[1])

                    case _:
                        raise CompilerError(f"Unimplemented log kind: {com}")

            case (
                CommandKind.sendkey
                | CommandKind.click
                | CommandKind.teleport
                | CommandKind.goto
                | CommandKind.usepotion
                | CommandKind.buypotions
                | CommandKind.relog
                | CommandKind.restart_client
                | CommandKind.tozone
                | CommandKind.cursor
                | CommandKind.select_friend
                | CommandKind.set_zone
                | CommandKind.set_goal
                | CommandKind.set_quest
                | CommandKind.autopet
            ):
                self.emit_deimos_call(com)

            # With completion, wait for it to appear, then to finish.
            case CommandKind.waitfor:
                non_inverted_com: Command = copy.copy(com)
                non_inverted_data: list[Any] = com.data[:]
                non_inverted_data[-1] = False
                non_inverted_com.data = non_inverted_data
                self.emit_deimos_call(non_inverted_com)

                if com.data[-1] is True:
                    self.emit_deimos_call(com)

            case CommandKind.set_yaw:
                self.emit(InstructionKind.set_yaw, [com.player_selector, com.data[0]])

            case CommandKind.load_playstyle:
                self.emit(InstructionKind.load_playstyle, com.data[0])

            case CommandKind.setdeck:
                self.emit(InstructionKind.setdeck, [com.player_selector, com.data[0]])

            case CommandKind.getdeck:
                self.emit(InstructionKind.getdeck, [com.player_selector])

            case _:
                raise CompilerError(f"Unimplemented command: {com}")

    def process_labels(self, program: list[Instruction]) -> list[Instruction]:
        """Replace labels with relative jump offsets."""
        new_program: list[Instruction] = []
        offsets: dict[Symbol, int] = {}

        # First pass drops labels, recording where each landed.
        for idx, instr in enumerate(program):
            match instr.kind:
                case InstructionKind.label:
                    sym: Symbol = instr.data
                    offsets[sym] = len(new_program)

                    # A label at the end has nothing to point at. Give it something.
                    if idx + 1 == len(program):
                        new_program.append(Instruction(InstructionKind.nop))

                case _:
                    new_program.append(instr)

        program = new_program

        def label_offset(sym: Symbol) -> int:
            """Where a label landed."""
            if sym not in offsets:
                raise CompilerError(f"Jump to a label that was never emitted: {sym}")

            return offsets[sym]

        # Second pass swaps each label for a jump distance.
        for idx, instr in enumerate(program):
            match instr.kind:
                case InstructionKind.call:
                    assert isinstance(instr.data, list)
                    sym = instr.data[0]
                    offset: int = label_offset(sym)
                    instr.data[0] = offset - idx

                case InstructionKind.jump:
                    sym = instr.data
                    offset = label_offset(sym)
                    instr.data = offset - idx

                case InstructionKind.jump_if | InstructionKind.jump_ifn:
                    assert isinstance(instr.data, list)
                    sym = instr.data[1]
                    offset = label_offset(sym)
                    instr.data[1] = offset - idx

                case InstructionKind.enter_until:
                    assert isinstance(instr.data, list)
                    sym = instr.data[2]
                    offset = label_offset(sym)
                    instr.data[2] = offset - idx

                case _:
                    pass

        return program

    def compile_block_def(self, block_def: BlockDefStmt) -> None:
        """Emit a block's body and label."""
        if isinstance(block_def.name, SymExpression):
            enter_block_label: Symbol = block_def.name.sym
            self.emit(InstructionKind.label, enter_block_label)
            prev_until: int | None = self._outermost_until
            self._outermost_until = None
            self._compile(block_def.body)
            self._outermost_until = prev_until

        elif isinstance(block_def.name, IdentExpression):
            raise CompilerError(f"Encountered an unresolved block sym during compilation: {block_def}")

        else:
            raise CompilerError(f"Encountered a malformed block sym during compilation: {block_def}")

    def compile_call(self, call: CallStmt) -> None:
        """Emit a call to a named block."""
        if isinstance(call.name, SymExpression):
            self.emit(InstructionKind.call, [call.name.sym, call.player_selector])

        elif isinstance(call.name, IdentExpression):
            raise CompilerError(f"Encountered an unresolved call during compilation: {call}")

        else:
            raise CompilerError(f"Encountered a malformed call during compilation: {call}")

    def compile_if_stmt(self, stmt: IfStmt) -> None:
        """Emit a conditional and both branches."""
        after_if_label: Symbol = self.gen_label("after_if")
        branch_true_label: Symbol = self.gen_label("branch_true")
        self.prep_expression(stmt.expr)
        self.emit(InstructionKind.jump_if, [stmt.expr, branch_true_label])
        self.enter_branch()
        self._compile(stmt.branch_false)
        self.exit_branch()
        self.emit(InstructionKind.jump, after_if_label)
        self.emit(InstructionKind.label, branch_true_label)
        self.enter_branch()
        self._compile(stmt.branch_true)
        self.exit_branch()
        self.emit(InstructionKind.label, after_if_label)

    def compile_loop_stmt(self, stmt: LoopStmt) -> None:
        """Emit an endless loop."""
        start_loop_label: Symbol = self.gen_label("start_loop")
        end_loop_label: Symbol = self.gen_label("end_loop")
        self._loop_label_stack.append(end_loop_label)
        self.emit(InstructionKind.label, start_loop_label)
        self.enter_branch()
        self._compile(stmt.body)
        self.exit_branch()
        self.emit(InstructionKind.jump, start_loop_label)
        self.emit(InstructionKind.label, end_loop_label)
        self._loop_label_stack.pop()

    def compile_while_stmt(self, stmt: WhileStmt) -> None:
        """Emit a loop that tests first."""
        start_while_label: Symbol = self.gen_label("start_while")
        end_while_label: Symbol = self.gen_label("end_while")
        self._loop_label_stack.append(end_while_label)
        self.prep_expression(stmt.expr)

        # Testing on the way in and again at the bottom saves a jump per pass.
        self.emit(InstructionKind.jump_ifn, [stmt.expr, end_while_label])
        self.emit(InstructionKind.label, start_while_label)
        self.enter_branch()
        self._compile(stmt.body)
        self.exit_branch()
        self.emit(InstructionKind.jump_if, [stmt.expr, start_while_label])
        self.emit(InstructionKind.label, end_while_label)
        self._loop_label_stack.pop()

    def compile_until_region(self, stmt: UntilRegion) -> None:
        """Emit the span an `until` guards."""
        until_id: int = self.analyzer.gen_sym_id()
        exit_until_label: Symbol = self.gen_label("exit_until")
        self._loop_label_stack.append(exit_until_label)
        self.prep_expression(stmt.expr)
        self.emit(InstructionKind.enter_until, [stmt.expr, until_id, exit_until_label])

        # Only the outermost until is tracked. Leaving it leaves the nested ones.
        is_outermost: bool = self._outermost_until is None

        if is_outermost:
            self._outermost_until = until_id

        self.enter_branch()
        self._compile(stmt.body)
        self.exit_branch()

        if is_outermost:
            self._outermost_until = None

        self.emit(InstructionKind.label, exit_until_label)
        self.emit(InstructionKind.exit_until, until_id)
        self._loop_label_stack.pop()

    def compile_return_stmt(self) -> None:
        """Emit a return from the enclosing block."""
        # Returning out of an until closes it, or the VM watches a dead condition.
        if self._outermost_until is not None:
            self.emit(InstructionKind.exit_until, self._outermost_until)

        self.emit(InstructionKind.ret)

    def check_cross_client_selector(self, expr: CommandExpression) -> None:
        """Refuse a group check on one client."""
        selector: PlayerSelector | None = expr.command.player_selector

        if selector is None or not expr.command.data:
            return

        kind: Any = expr.command.data[0]

        if kind not in CROSS_CLIENT_CHECKS:
            return

        spelling: str = kind.name.replace("_", "")
        reason: str = f"{spelling} compares clients against each other, so "

        if selector.negated:
            raise CompilerError(
                f"{reason}a selector cannot negate it one client at a time. Write `not mass {spelling}` instead"
            )

        # One client always matches itself.
        if selector.any_player:
            raise CompilerError(f"{reason}anyplayer cannot answer it. Write `mass {spelling}` instead")

        # Whoever the last `any` matched may be one client, or none.
        if selector.same_any:
            raise CompilerError(f"{reason}sameany cannot answer it. Write `mass {spelling}` instead")

        # `except p1` leaves the others, so still a group.
        if len(selector.player_nums) == 1 and not selector.inverted:
            raise CompilerError(
                f"{reason}one client cannot answer it. Name a second client like `p1:p2`, "
                f"or write `mass {spelling}` instead"
            )

    def prep_expression(self, expr: Expression) -> None:
        """Rewrite an expression for the VM."""
        match expr:
            case ConstantCheckExpression():
                self.prep_expression(expr.value)

            case AndExpression() | OrExpression():
                for sub_expr in expr.expressions:
                    self.prep_expression(sub_expr)

            case BinaryExpression():
                self.prep_expression(expr.lhs)
                self.prep_expression(expr.rhs)

            # Stack offsets are knowable only here, once the code is laid out.
            case ReadVarExpr():
                if isinstance(expr.loc, SymExpression):
                    expr.sym = expr.loc.sym
                    expr.loc = StackLocExpression(self.stack_loc(expr.sym))

                # A desugared `until` condition arrives more than once. A repeat must agree, or one reads a dead slot.
                elif isinstance(expr.loc, StackLocExpression):
                    if expr.sym is None:
                        raise CompilerError(f"ReadVarExpr was lowered without keeping its symbol: {expr}")

                    current_loc: int = self.stack_loc(expr.sym)

                    if current_loc != expr.loc.offset:
                        raise CompilerError(
                            f"ReadVarExpr lowered twice to different slots: {expr.sym} "
                            f"was {expr.loc.offset}, now {current_loc}"
                        )

                else:
                    raise CompilerError(f"Malformed ReadVarExpr: {expr}")

            case SelectorGroup() | UnaryExpression() | WholeNumberExpression():
                self.prep_expression(expr.expr)

            case ListExpression():
                for item in expr.items:
                    self.prep_expression(item)

            case RangeMinExpression() | RangeMaxExpression():
                self.prep_expression(expr.range_expr)

            case IndexAccessExpression():
                self.prep_expression(expr.expr)
                self.prep_expression(expr.index)

            case CommandExpression():
                self.check_cross_client_selector(expr)

            case (
                ConstantExpression()
                | ConstantReferenceExpression()
                | NumberExpression()
                | StringExpression()
                | KeyExpression()
                | XYZExpression()
                | IdentExpression()
                | StackLocExpression()
                | Eval()
            ):
                pass

            case _:
                raise CompilerError(f"Unhandled expression type: {expr}")

    def _compile(self, stmt: Stmt) -> None:
        """Tag the line, then emit."""
        previous_line: LineInfo | None = self._current_line

        # Synthesized statements keep the enclosing line.
        if stmt.line_info is not None:
            self._current_line = stmt.line_info

        try:
            self._compile_inner(stmt)

        except CompilerError as error:
            error.locate(self._current_line)
            raise

        finally:
            self._current_line = previous_line

    def _compile_inner(self, stmt: Stmt) -> None:
        """Emit code for one statement."""
        match stmt:
            case ConstantDeclStmt():
                self.prep_expression(stmt.value)
                self.emit(InstructionKind.declare_constant, [stmt.name, stmt.value])

            case TimerStmt():
                match stmt.action:
                    case TimerAction.start:
                        self.emit(InstructionKind.start_timer, stmt.timer_name)

                    case TimerAction.reset:
                        self.emit(InstructionKind.reset_timer, stmt.timer_name)

                    case TimerAction.end:
                        self.emit(InstructionKind.end_timer, stmt.timer_name)

            case CounterStmt():
                match stmt.action:
                    case CounterAction.start:
                        self.emit(InstructionKind.start_counter, stmt.counter_name)

                    case CounterAction.reset:
                        self.emit(InstructionKind.reset_counter, stmt.counter_name)

                    case CounterAction.end:
                        self.emit(InstructionKind.end_counter, stmt.counter_name)

                    case CounterAction.add:
                        self.emit(InstructionKind.change_counter, [stmt.counter_name, 1])

                    case CounterAction.subtract:
                        self.emit(InstructionKind.change_counter, [stmt.counter_name, -1])

            case StmtList():
                for inner in stmt.stmts:
                    self._compile(inner)

            case CommandStmt():
                self.compile_command(stmt.command)

            case CallStmt():
                self.compile_call(stmt)

            case BlockDefStmt():
                self.compile_block_def(stmt)

            case IfStmt():
                self.compile_if_stmt(stmt)

            case LoopStmt():
                self.compile_loop_stmt(stmt)

            case WhileStmt():
                self.compile_while_stmt(stmt)

            case DefVarStmt():
                self._stacks[-1].push(stmt.sym)
                self.emit(InstructionKind.push_stack)

            case KillVarStmt():
                self.emit(InstructionKind.pop_stack)
                self._stacks[-1].pop(stmt.sym)

            case WriteVarStmt():
                self.prep_expression(stmt.expr)
                self.emit(InstructionKind.write_stack, [self.stack_loc(stmt.sym), stmt.expr])

            case BreakStmt():
                label: Symbol = self._loop_label_stack[-1]
                self.emit(InstructionKind.jump, label)

            case ReturnStmt():
                self.compile_return_stmt()

            case UntilRegion():
                self.compile_until_region(stmt)

            case _:
                raise CompilerError(f"Unknown statement: {stmt}\n{type(stmt)}")

    def compile(self) -> list[Instruction]:
        """Compile the analysed program, once."""
        # Block bodies sit at the front. The first instruction jumps over them.
        toplevel_start_label: Symbol = self.gen_label("program_start")
        self.emit(InstructionKind.jump, toplevel_start_label)

        for stmt in self.analyzer.block_defs:
            self._compile(stmt)

        self.emit(InstructionKind.label, toplevel_start_label)

        for stmt in self.analyzer.stmts:
            self._compile(stmt)

        return self.process_labels(self._program)
