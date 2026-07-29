"""Each command's syntax and behaviour."""

from .expressions import EXPR_REGISTRY, ExprSpec, expr_command
from .handlers import (
    HANDLERS,
    INSTRUCTION_HANDLERS,
    ExecContext,
    InstructionContext,
    handler,
    instruction_handler,
)
from .predicates import PREDICATES, STATS, EvalContext, StatContext, predicate, stat
from .statements import REGISTRY, CommandSpec, command

__all__: list[str] = [
    "EXPR_REGISTRY",
    "HANDLERS",
    "INSTRUCTION_HANDLERS",
    "PREDICATES",
    "REGISTRY",
    "STATS",
    "CommandSpec",
    "EvalContext",
    "ExecContext",
    "ExprSpec",
    "InstructionContext",
    "StatContext",
    "command",
    "expr_command",
    "handler",
    "instruction_handler",
    "predicate",
    "stat",
]
