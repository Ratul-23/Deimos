"""Token kinds and their spellings."""

from collections.abc import Sequence
from enum import Enum, auto


class TokenKind(Enum):
    """Every token kind."""

    player_num = auto()
    player_all = auto()
    string = auto()
    number = auto()
    contains = auto()
    percent = auto()
    path = auto()
    logical_to = auto()
    logical_on = auto()
    logical_off = auto()
    boolean_true = auto()
    boolean_false = auto()

    greater = auto()
    less = auto()
    equals = auto()

    keyword_block = auto()
    keyword_call = auto()
    keyword_loop = auto()
    keyword_while = auto()
    keyword_until = auto()
    keyword_times = auto()
    keyword_if = auto()
    keyword_elif = auto()
    keyword_else = auto()
    keyword_except = auto()
    keyword_mass = auto()
    keyword_mob = auto()
    keyword_quest = auto()
    keyword_icon = auto()
    keyword_ifneeded = auto()
    keyword_completion = auto()
    keyword_xyz = auto()
    keyword_not = auto()
    keyword_return = auto()
    keyword_break = auto()
    keyword_mixin = auto()
    keyword_and = auto()
    keyword_or = auto()
    keyword_any_player = auto()
    keyword_starttimer = auto()
    keyword_resettimer = auto()
    keyword_endtimer = auto()
    keyword_startcounter = auto()
    keyword_resetcounter = auto()
    keyword_endcounter = auto()
    keyword_addone = auto()
    keyword_minusone = auto()
    keyword_same_any = auto()
    keyword_isbetween = auto()
    keyword_con = auto()

    command_kill = auto()
    command_sleep = auto()
    command_log = auto()
    command_goto = auto()
    command_sendkey = auto()
    command_waitfor_dialog = auto()
    command_waitfor_battle = auto()
    command_waitfor_zonechange = auto()
    command_waitfor_free = auto()
    command_waitfor_window = auto()
    command_usepotion = auto()
    command_buypotions = auto()
    command_relog = auto()
    command_click = auto()
    command_clickwindow = auto()
    command_teleport = auto()
    command_friendtp = auto()
    command_entitytp = auto()
    command_plus_teleport = auto()
    command_minus_teleport = auto()
    command_tozone = auto()
    command_load_playstyle = auto()
    command_set_yaw = auto()
    command_nav = auto()
    command_setdeck = auto()
    command_getdeck = auto()
    command_select_friend = auto()
    command_autopet = auto()
    command_set_goal = auto()
    command_set_quest = auto()
    command_set_zone = auto()
    command_toggle_combat = auto()
    command_restart_bot = auto()
    command_restart_client = auto()
    command_move_cursor = auto()
    command_move_cursor_window = auto()

    command_expr_window_visible = auto()
    command_expr_window_disabled = auto()
    command_expr_window_text = auto()
    command_expr_window_num = auto()

    command_expr_in_zone = auto()
    command_expr_in_combat = auto()
    command_expr_in_range = auto()

    command_expr_same_zone = auto()
    command_expr_same_quest = auto()
    command_expr_same_xyz = auto()
    command_expr_same_yaw = auto()
    command_expr_same_place = auto()

    command_expr_has_dialogue = auto()
    command_expr_has_xyz = auto()
    command_expr_has_quest = auto()
    command_expr_has_yaw = auto()

    command_expr_health = auto()
    command_expr_health_above = auto()
    command_expr_health_below = auto()
    command_expr_mana = auto()
    command_expr_mana_above = auto()
    command_expr_mana_below = auto()
    command_expr_energy = auto()
    command_expr_energy_above = auto()
    command_expr_energy_below = auto()
    command_expr_bagcount = auto()
    command_expr_bagcount_above = auto()
    command_expr_bagcount_below = auto()
    command_expr_gold = auto()
    command_expr_gold_above = auto()
    command_expr_gold_below = auto()
    command_expr_potion_count = auto()
    command_expr_potion_countabove = auto()
    command_expr_potion_countbelow = auto()
    command_expr_playercount = auto()
    command_expr_playercountabove = auto()
    command_expr_playercountbelow = auto()
    command_expr_counter = auto()
    command_expr_timer = auto()

    command_expr_tracking_quest = auto()
    command_expr_tracking_goal = auto()
    command_expr_quest_changed = auto()
    command_expr_goal_changed = auto()
    command_expr_zone_changed = auto()

    command_expr_loading = auto()
    command_expr_item_dropped = auto()
    command_expr_duel_round = auto()
    command_expr_account_level = auto()
    command_expr_any_player_list = auto()

    colon = auto()
    comma = auto()

    plus = auto()
    minus = auto()
    star = auto()
    slash = auto()
    modulo = auto()

    slash_slash = auto()
    star_star = auto()

    paren_open = auto()
    paren_close = auto()
    square_open = auto()
    square_close = auto()
    curly_open = auto()
    curly_close = auto()

    identifier = auto()

    END_LINE = auto()


# Matched against normalize_ident output, so lowercase and underscore free.
_SYNTAX_KEYWORDS: tuple[tuple[str, TokenKind], ...] = (
    ("contains", TokenKind.contains),
    ("to", TokenKind.logical_to),
    ("on", TokenKind.logical_on),
    ("off", TokenKind.logical_off),
    ("true", TokenKind.boolean_true),
    ("false", TokenKind.boolean_false),
    ("block", TokenKind.keyword_block),
    ("call", TokenKind.keyword_call),
    ("loop", TokenKind.keyword_loop),
    ("while", TokenKind.keyword_while),
    ("until", TokenKind.keyword_until),
    ("times", TokenKind.keyword_times),
    ("if", TokenKind.keyword_if),
    ("elif", TokenKind.keyword_elif),
    ("else", TokenKind.keyword_else),
    ("except", TokenKind.keyword_except),
    ("mass", TokenKind.keyword_mass),
    ("closestmob", TokenKind.keyword_mob),
    ("mob", TokenKind.keyword_mob),
    ("quest", TokenKind.keyword_quest),
    ("questpos", TokenKind.keyword_quest),
    ("questposition", TokenKind.keyword_quest),
    ("icon", TokenKind.keyword_icon),
    ("ifneeded", TokenKind.keyword_ifneeded),
    ("completion", TokenKind.keyword_completion),
    ("xyz", TokenKind.keyword_xyz),
    ("not", TokenKind.keyword_not),
    ("return", TokenKind.keyword_return),
    ("exitblock", TokenKind.keyword_return),
    ("break", TokenKind.keyword_break),
    ("exitloop", TokenKind.keyword_break),
    ("mixin", TokenKind.keyword_mixin),
    ("and", TokenKind.keyword_and),
    ("or", TokenKind.keyword_or),
    ("any", TokenKind.keyword_any_player),
    ("anyplayer", TokenKind.keyword_any_player),
    ("anyclient", TokenKind.keyword_any_player),
    ("starttimer", TokenKind.keyword_starttimer),
    ("createtimer", TokenKind.keyword_starttimer),
    ("settimer", TokenKind.keyword_starttimer),
    ("resettimer", TokenKind.keyword_resettimer),
    ("endtimer", TokenKind.keyword_endtimer),
    ("canceltimer", TokenKind.keyword_endtimer),
    ("stoptimer", TokenKind.keyword_endtimer),
    ("startcounter", TokenKind.keyword_startcounter),
    ("createcounter", TokenKind.keyword_startcounter),
    ("setcounter", TokenKind.keyword_startcounter),
    ("resetcounter", TokenKind.keyword_resetcounter),
    ("endcounter", TokenKind.keyword_endcounter),
    ("cancelcounter", TokenKind.keyword_endcounter),
    ("stopcounter", TokenKind.keyword_endcounter),
    ("addone", TokenKind.keyword_addone),
    ("minusone", TokenKind.keyword_minusone),
    ("sameany", TokenKind.keyword_same_any),
    ("sameanyplayer", TokenKind.keyword_same_any),
    ("sameanyclient", TokenKind.keyword_same_any),
    ("isbetween", TokenKind.keyword_isbetween),
    ("between", TokenKind.keyword_isbetween),
    ("var", TokenKind.keyword_con),
    ("con", TokenKind.keyword_con),
    ("set", TokenKind.keyword_con),
    ("setvar", TokenKind.keyword_con),
)

_STATEMENT_COMMANDS: tuple[tuple[str, TokenKind], ...] = (
    ("kill", TokenKind.command_kill),
    ("killbot", TokenKind.command_kill),
    ("stop", TokenKind.command_kill),
    ("stopbot", TokenKind.command_kill),
    ("end", TokenKind.command_kill),
    ("exit", TokenKind.command_kill),
    ("sleep", TokenKind.command_sleep),
    ("wait", TokenKind.command_sleep),
    ("delay", TokenKind.command_sleep),
    ("log", TokenKind.command_log),
    ("debug", TokenKind.command_log),
    ("print", TokenKind.command_log),
    ("walkto", TokenKind.command_goto),
    ("goto", TokenKind.command_goto),
    ("sendkey", TokenKind.command_sendkey),
    ("press", TokenKind.command_sendkey),
    ("presskey", TokenKind.command_sendkey),
    ("waitfordialog", TokenKind.command_waitfor_dialog),
    ("waitfordialogue", TokenKind.command_waitfor_dialog),
    ("waitforcombat", TokenKind.command_waitfor_battle),
    ("waitforbattle", TokenKind.command_waitfor_battle),
    ("waitforzonechange", TokenKind.command_waitfor_zonechange),
    ("waitforfree", TokenKind.command_waitfor_free),
    ("waitforwindow", TokenKind.command_waitfor_window),
    ("waitforpath", TokenKind.command_waitfor_window),
    ("usepotion", TokenKind.command_usepotion),
    ("buypotions", TokenKind.command_buypotions),
    ("refillpotions", TokenKind.command_buypotions),
    ("buypots", TokenKind.command_buypotions),
    ("refillpots", TokenKind.command_buypotions),
    ("relog", TokenKind.command_relog),
    ("logoutandin", TokenKind.command_relog),
    ("click", TokenKind.command_click),
    ("clickwindow", TokenKind.command_clickwindow),
    ("teleport", TokenKind.command_teleport),
    ("tp", TokenKind.command_teleport),
    ("setpos", TokenKind.command_teleport),
    ("friendtp", TokenKind.command_friendtp),
    ("friendteleport", TokenKind.command_friendtp),
    ("entitytp", TokenKind.command_entitytp),
    ("entityteleport", TokenKind.command_entitytp),
    ("plustp", TokenKind.command_plus_teleport),
    ("plusteleport", TokenKind.command_plus_teleport),
    ("minustp", TokenKind.command_minus_teleport),
    ("minusteleport", TokenKind.command_minus_teleport),
    ("tozone", TokenKind.command_tozone),
    ("loadplaystyle", TokenKind.command_load_playstyle),
    ("turncam", TokenKind.command_set_yaw),
    ("setcamyaw", TokenKind.command_set_yaw),
    ("nav", TokenKind.command_nav),
    ("navtp", TokenKind.command_nav),
    ("setdeck", TokenKind.command_setdeck),
    ("getdeck", TokenKind.command_getdeck),
    ("selectfriend", TokenKind.command_select_friend),
    ("choosefriend", TokenKind.command_select_friend),
    ("autopet", TokenKind.command_autopet),
    ("toggleautopet", TokenKind.command_autopet),
    ("loggoal", TokenKind.command_set_goal),
    ("logquest", TokenKind.command_set_quest),
    ("logzone", TokenKind.command_set_zone),
    ("togglecombat", TokenKind.command_toggle_combat),
    ("togglecombatmode", TokenKind.command_toggle_combat),
    ("rerun", TokenKind.command_restart_bot),
    ("restart", TokenKind.command_restart_bot),
    ("restartbot", TokenKind.command_restart_bot),
    ("restartclient", TokenKind.command_restart_client),
    ("relaunchclient", TokenKind.command_restart_client),
    ("cursor", TokenKind.command_move_cursor),
    ("movecursor", TokenKind.command_move_cursor),
    ("mousexy", TokenKind.command_move_cursor),
    ("movemouse", TokenKind.command_move_cursor),
    ("cursorwindow", TokenKind.command_move_cursor_window),
    ("mousewindow", TokenKind.command_move_cursor_window),
)

_EXPRESSION_COMMANDS: tuple[tuple[str, TokenKind], ...] = (
    ("windowvisible", TokenKind.command_expr_window_visible),
    ("windowdisabled", TokenKind.command_expr_window_disabled),
    ("windowtext", TokenKind.command_expr_window_text),
    ("windownum", TokenKind.command_expr_window_num),
    ("inzone", TokenKind.command_expr_in_zone),
    ("incombat", TokenKind.command_expr_in_combat),
    ("inrange", TokenKind.command_expr_in_range),
    ("samezone", TokenKind.command_expr_same_zone),
    ("samequest", TokenKind.command_expr_same_quest),
    ("samexyz", TokenKind.command_expr_same_xyz),
    ("sameyaw", TokenKind.command_expr_same_yaw),
    ("sameplace", TokenKind.command_expr_same_place),
    ("hasdialogue", TokenKind.command_expr_has_dialogue),
    ("hasxyz", TokenKind.command_expr_has_xyz),
    ("hasquest", TokenKind.command_expr_has_quest),
    ("hasyaw", TokenKind.command_expr_has_yaw),
    ("health", TokenKind.command_expr_health),
    ("healthabove", TokenKind.command_expr_health_above),
    ("healthbelow", TokenKind.command_expr_health_below),
    ("mana", TokenKind.command_expr_mana),
    ("manaabove", TokenKind.command_expr_mana_above),
    ("manabelow", TokenKind.command_expr_mana_below),
    ("energy", TokenKind.command_expr_energy),
    ("energyabove", TokenKind.command_expr_energy_above),
    ("energybelow", TokenKind.command_expr_energy_below),
    ("bagcount", TokenKind.command_expr_bagcount),
    ("bagcountabove", TokenKind.command_expr_bagcount_above),
    ("bagcountbelow", TokenKind.command_expr_bagcount_below),
    ("gold", TokenKind.command_expr_gold),
    ("goldabove", TokenKind.command_expr_gold_above),
    ("goldbelow", TokenKind.command_expr_gold_below),
    ("potioncount", TokenKind.command_expr_potion_count),
    ("potioncountabove", TokenKind.command_expr_potion_countabove),
    ("potioncountbelow", TokenKind.command_expr_potion_countbelow),
    ("playercount", TokenKind.command_expr_playercount),
    ("clientcount", TokenKind.command_expr_playercount),
    ("playercountabove", TokenKind.command_expr_playercountabove),
    ("clientcountabove", TokenKind.command_expr_playercountabove),
    ("playercountbelow", TokenKind.command_expr_playercountbelow),
    ("clientcountbelow", TokenKind.command_expr_playercountbelow),
    ("counter", TokenKind.command_expr_counter),
    ("timer", TokenKind.command_expr_timer),
    ("trackingquest", TokenKind.command_expr_tracking_quest),
    ("trackinggoal", TokenKind.command_expr_tracking_goal),
    ("questchanged", TokenKind.command_expr_quest_changed),
    ("goalchanged", TokenKind.command_expr_goal_changed),
    ("zonechanged", TokenKind.command_expr_zone_changed),
    ("loading", TokenKind.command_expr_loading),
    ("itemdropped", TokenKind.command_expr_item_dropped),
    ("duelround", TokenKind.command_expr_duel_round),
    ("combatround", TokenKind.command_expr_duel_round),
    ("fightround", TokenKind.command_expr_duel_round),
    ("accountlevel", TokenKind.command_expr_account_level),
    ("level", TokenKind.command_expr_account_level),
    ("anyplayerlist", TokenKind.command_expr_any_player_list),
    ("anyclientlist", TokenKind.command_expr_any_player_list),
)


def _merge_keywords(*groups: tuple[tuple[str, TokenKind], ...]) -> dict[str, TokenKind]:
    """Flatten keyword groups into one table."""
    result: dict[str, TokenKind] = {}

    for group in groups:
        for spelling, kind in group:
            if spelling != spelling.lower().replace("_", ""):
                raise ValueError(f"Keyword is not lowercase and underscore free: {spelling}")

            if spelling in result:
                raise ValueError(f"Keyword spelled more than once: {spelling}")

            result[spelling] = kind

    return result


KEYWORDS: dict[str, TokenKind] = _merge_keywords(_SYNTAX_KEYWORDS, _STATEMENT_COMMANDS, _EXPRESSION_COMMANDS)

# Tells a keyword from a name a script made up.
KEYWORD_KINDS: frozenset[TokenKind] = frozenset(KEYWORDS.values())


# Kinds with no word to spell them, named as a script would.
_KIND_NAMES: dict[TokenKind, str] = {
    TokenKind.player_num: "a client such as p1",
    TokenKind.player_all: "`p*`",
    TokenKind.string: "a quoted string",
    TokenKind.number: "a number",
    TokenKind.percent: "a percentage",
    TokenKind.path: "a path such as wizardcity/unicornway",
    TokenKind.greater: "`>`",
    TokenKind.less: "`<`",
    TokenKind.equals: "`=`",
    TokenKind.colon: "`:`",
    TokenKind.comma: "`,`",
    TokenKind.plus: "`+`",
    TokenKind.minus: "`-`",
    TokenKind.star: "`*`",
    TokenKind.slash: "`/`",
    TokenKind.modulo: "`%`",
    TokenKind.slash_slash: "`//`",
    TokenKind.star_star: "`**`",
    TokenKind.paren_open: "`(`",
    TokenKind.paren_close: "`)`",
    TokenKind.square_open: "`[`",
    TokenKind.square_close: "`]`",
    TokenKind.curly_open: "`{`",
    TokenKind.curly_close: "`}`",
    TokenKind.identifier: "a name",
    TokenKind.END_LINE: "the end of the line",
}


def _spellings_by_kind() -> dict[TokenKind, list[str]]:
    """Every spelling of each kind."""
    result: dict[TokenKind, list[str]] = {}

    for spelling, kind in KEYWORDS.items():
        result.setdefault(kind, []).append(spelling)

    return result


_SPELLINGS: dict[TokenKind, list[str]] = _spellings_by_kind()


def describe(kind: TokenKind) -> str:
    """A kind's name for an error."""
    # First spelling only. Listing aliases buries the point.
    if kind in _SPELLINGS:
        return f"`{_SPELLINGS[kind][0]}`"

    return _KIND_NAMES.get(kind, kind.name)


def describe_any(kinds: Sequence[TokenKind | str]) -> str:
    """Several kinds named for an error."""
    named: list[str] = []

    for kind in kinds:
        if isinstance(kind, TokenKind):
            named.append(describe(kind))
        else:
            named.append("a window path" if kind == "window_path" else kind)

    return " or ".join(named)
