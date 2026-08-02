"""Tokens, and the errors scripts cause."""

from math import isfinite
from typing import Any, Never

from .tokens import KEYWORDS, TokenKind

# float is a number, list[str] a path, int a client number.
type TokenValue = float | str | list[str] | int | None


class Percent(float):
    """A percentage literal, stored as a fraction."""


class LineInfo:
    """Where a token came from."""

    def __init__(
        self,
        line: int,
        column: int,
        last_column: int,
        last_line: int | None = None,
        filename: str | None = None,
        source_line: str | None = None,
    ) -> None:
        self.line: int = line
        self.column: int = column
        self.last_column: int = last_column
        self.filename: str | None = filename
        self.last_line: int = last_line if last_line is not None else line

        # One string per line, shared. Free to carry.
        self.source_line: str | None = source_line

    def location(self) -> str:
        """The file and line, as text."""
        if self.filename is None:
            return f"Line: {self.line}"

        return f"{self.filename}, Line: {self.line}"

    def render(self) -> str:
        """The source line and a caret."""
        if self.source_line is None:
            return self.location()

        prefix: str = self.source_line[: self.column - 1]

        # Tabs stay tabs, or the caret lands short.
        caret_indent: str = "".join("\t" if char == "\t" else " " for char in prefix)
        return f"{self.source_line}\n{caret_indent}^\n{self.location()}"

    def __repr__(self) -> str:
        if self.filename is not None:
            return f"{self.filename}:{self.line}:{self.column}-{self.last_column}"

        return f"{self.line}:{self.column}-{self.last_column}"


class DeimosLangError(Exception):
    """Base for errors a script causes."""


class TokenizerError(DeimosLangError):
    """Source text cannot be tokenized."""


class LocatedError(DeimosLangError):
    """An error that shows its line."""

    def __init__(self, message: str) -> None:
        super().__init__(message)
        self.message: str = message
        self.line_info: LineInfo | None = None

    def locate(self, line_info: LineInfo | None) -> None:
        """Attach a location, keeping the first."""
        if self.line_info is None:
            self.line_info = line_info

    def brief(self) -> str:
        """Message and line, on one line."""
        if self.line_info is None:
            return self.message

        return f"{self.message} ({self.line_info.location()})"

    def __str__(self) -> str:
        if self.line_info is None:
            return self.message

        return f"{self.message}\n{self.line_info.render()}"


class Token:
    """One lexed token."""

    def __init__(self, kind: TokenKind, literal: str, line_info: LineInfo, value: Any = None) -> None:
        self.kind: TokenKind = kind
        self.literal: str = literal
        self.value: Any = value
        self.line_info: LineInfo = line_info

    def __repr__(self) -> str:
        return f"{self.line_info} {self.kind.name}`{self.literal}`({self.value})"


def normalize_ident(dirty: str) -> str:
    """Lowercase a name, strip underscores."""
    return dirty.lower().replace("_", "")


def _written_number(text: str) -> float:
    """A written number, kept whole."""
    value: float = float(text)

    # Infinity is not whole. Asking raises.
    if "." not in text and isfinite(value) and value.is_integer():
        return int(value)

    return value


class Tokenizer:
    """Turns source text into tokens."""

    def __init__(self) -> None:
        self._in_multiline_string: bool = False
        self._multiline_buffer: str = ""
        self._multiline_start_line_info: LineInfo = LineInfo(0, 0, 0, 0)

    def tokenize_line(self, line: str, line_num: int, filename: str | None = None) -> list[Token]:
        """Tokenize one line of source."""
        result: list[Token] = []
        pos: int = 0

        def put_simple(kind: TokenKind, literal: str, value: Any = None) -> None:
            """Append a token at this spot."""
            # END_LINE carries no text. Widen the span or last column precedes first.
            line_info: LineInfo = LineInfo(
                line=line_num,
                column=pos + 1,
                last_column=pos + max(len(literal), 1),
                filename=filename,
                source_line=line,
            )
            result.append(Token(kind, literal, line_info, value))

        def err(message: str, column_start: int) -> Never:
            """Raise a TokenizerError at a column."""
            line_info: LineInfo = LineInfo(
                line=line_num,
                column=column_start + 1,
                last_column=column_start + 1,
                filename=filename,
                source_line=line,
            )
            raise TokenizerError(f"{message}\n{line_info.render()}")

        while pos < len(line):
            char: str = line[pos]

            # Backtick string collects until the closing backtick, lines included.
            if self._in_multiline_string:
                self._multiline_buffer += char

                if char == "`":
                    self._multiline_start_line_info.last_column = pos + 1
                    self._multiline_start_line_info.last_line = line_num
                    result.append(
                        Token(
                            TokenKind.string,
                            self._multiline_buffer,
                            self._multiline_start_line_info,
                            self._multiline_buffer[1:-1],
                        )
                    )
                    self._in_multiline_string = False
                    self._multiline_buffer = ""

                pos += 1

            else:
                match char:
                    case "&":
                        err("Unexpected &", pos)

                    # A star never starts a token. p* would split in two.
                    case "p" | "P" if pos + 1 < len(line) and line[pos + 1] == "*":
                        put_simple(TokenKind.player_all, line[pos : pos + 2])
                        pos += 2

                    case ":":
                        put_simple(TokenKind.colon, char)
                        pos += 1

                    case ",":
                        put_simple(TokenKind.comma, char)
                        pos += 1

                    case "+":
                        put_simple(TokenKind.plus, char)
                        pos += 1

                    case "-":
                        put_simple(TokenKind.minus, char)
                        pos += 1

                    case ">":
                        put_simple(TokenKind.greater, char)
                        pos += 1

                    case "<":
                        put_simple(TokenKind.less, char)
                        pos += 1

                    case "=":
                        if pos + 1 < len(line) and line[pos + 1] == "=":
                            put_simple(TokenKind.equals, "==")
                            pos += 2
                        else:
                            put_simple(TokenKind.equals, char)
                            pos += 1

                    case "*":
                        if pos + 1 < len(line) and line[pos + 1] == "*":
                            put_simple(TokenKind.star_star, "**")
                            pos += 2
                        else:
                            put_simple(TokenKind.star, char)
                            pos += 1

                    # A percentage keeps its %. A lone one is the leftover.
                    case "%":
                        put_simple(TokenKind.modulo, char)
                        pos += 1

                    case "/":
                        if pos + 1 < len(line) and line[pos + 1] == "/":
                            put_simple(TokenKind.slash_slash, "//")
                            pos += 2
                        else:
                            put_simple(TokenKind.slash, char)
                            pos += 1

                    case "(":
                        put_simple(TokenKind.paren_open, char)
                        pos += 1

                    case ")":
                        put_simple(TokenKind.paren_close, char)
                        pos += 1

                    case "[":
                        put_simple(TokenKind.square_open, char)
                        pos += 1

                    case "]":
                        put_simple(TokenKind.square_close, char)
                        pos += 1

                    case "{":
                        put_simple(TokenKind.curly_open, char)
                        pos += 1

                    case "}":
                        put_simple(TokenKind.curly_close, char)
                        pos += 1

                    case '"' | "'":
                        quote_kind: str = char
                        str_lit: str = char
                        scan: int = pos + 1

                        while scan < len(line) and line[scan] != quote_kind:
                            str_lit += line[scan]
                            scan += 1

                        if scan >= len(line):
                            err("Unclosed string encountered", pos)

                        str_lit += line[scan]
                        scan += 1
                        put_simple(TokenKind.string, str_lit, str_lit[1:-1])
                        pos = scan

                    case "`":
                        self._multiline_buffer = char
                        self._in_multiline_string = True
                        self._multiline_start_line_info = LineInfo(
                            line=line_num, column=pos + 1, last_column=pos + 1, filename=filename, source_line=line
                        )
                        pos += 1

                    # Everything after a # is a comment.
                    case "#":
                        break

                    # A run of characters, classified once fully read.
                    case _:
                        if char.isspace():
                            pos += 1

                        else:
                            full: str = ""
                            scan: int = pos

                            # Dot and slash build numbers and paths. Minus ends a number, unless exponent or hyphen.
                            while scan < len(line) and not (
                                line[scan].isspace()
                                or line[scan] in "(){}[]:,`+*<>=&#"
                                or (line[scan] == "-" and all(ch.isdigit() or ch in ".%" for ch in full))
                            ):
                                full += line[scan]
                                scan += 1

                            # A digit somewhere keeps names like `e` out.
                            if any(ch.isdigit() for ch in full) and all(ch.isnumeric() or ch in ".eE-%" for ch in full):
                                if "%" in full:
                                    # More number after the % means spaces were left out.
                                    if not full.endswith("%"):
                                        err(f"Put spaces around the % in {full} to take the leftover", pos)

                                    try:
                                        put_simple(TokenKind.percent, full, Percent(float(full[:-1]) / 100))
                                    except ValueError:
                                        err("Unable to convert to percent", pos)

                                else:
                                    try:
                                        put_simple(TokenKind.number, full, _written_number(full))
                                    except ValueError:
                                        err("Unable to convert to number", pos)

                            # Only numbers and slashes spell a division, not a path.
                            elif "/" in full and all(ch.isdigit() or ch == "." or ch == "/" for ch in full):
                                number_text: str = full[: full.index("/")]

                                try:
                                    put_simple(TokenKind.number, number_text, _written_number(number_text))
                                except ValueError:
                                    err("Unable to convert to number", pos)

                                scan = pos + len(number_text)

                            elif "/" in full:
                                if full.endswith("/"):
                                    err("Invalid path", pos)

                                put_simple(TokenKind.path, full, full.split("/"))

                            # p followed by digits is a client, such as p1.
                            elif full[0].lower() == "p" and full[1:].isnumeric():
                                try:
                                    put_simple(TokenKind.player_num, full, int(full[1:]))
                                except ValueError:
                                    err("Unable to convert to a client number", pos)

                            else:
                                put_simple(KEYWORDS.get(normalize_ident(full), TokenKind.identifier), full)

                            pos = scan

        # An open multiline string swallows the line break.
        if not self._in_multiline_string:
            put_simple(TokenKind.END_LINE, "")

        return result

    def tokenize(self, contents: str, filename: str | None = None) -> list[Token]:
        """Tokenize a whole program."""
        result: list[Token] = []

        for line_num, line in enumerate(contents.splitlines()):
            toks: list[Token] = self.tokenize_line(line, line_num + 1, filename=filename)

            # splitlines drops the newline. A multiline string needs it.
            if self._in_multiline_string:
                self._multiline_buffer += "\n"

            # Nothing but END_LINE means a blank line.
            elif len(toks) == 1:
                continue

            result.extend(toks)

        if self._in_multiline_string:
            # Left mid-string. The next program is not part of it.
            unclosed: str = self._multiline_buffer
            self._in_multiline_string = False
            self._multiline_buffer = ""
            raise TokenizerError(f"Unclosed multiline string: {unclosed} {self._multiline_start_line_info}")

        return result
