"""Colony dilemma engine.

Pure game logic. Knows nothing about language models or logging, so it can
be tested with no network and no Ollama.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field, replace
from pathlib import Path

FALLBACKS = ("any", "any_except_cast", "none")
BUILTINS = ("dead", "dead_count", "alive_count")
OUTCOME_ONLY = ("dead", "dead_count")

_PLACEHOLDER = re.compile(r"\{(\w+)\}")


class ScenarioError(ValueError):
    """Raised when scenes.json is malformed, or when rendering a scene
    would contradict the game state."""


@dataclass(frozen=True)
class Colonist:
    name: str
    role: str


@dataclass(frozen=True)
class Role:
    """A part in a scene: the preferred colonist, and who plays it if
    they are dead."""

    key: str
    prefer: str
    fallback: str


@dataclass(frozen=True)
class Option:
    """`text` and `outcome` are templates; `victims` are role keys."""

    id: str
    text: str
    deaths: int
    outcome: str
    victims: tuple[str, ...] = ()
    sets: tuple[str, ...] = ()


@dataclass(frozen=True)
class Variant:
    text: str
    options: tuple[Option, ...]
    requires: str | None = None
    cast: tuple[Role, ...] = ()


@dataclass(frozen=True)
class Scene:
    id: str
    title: str
    variants: tuple[Variant, ...]


@dataclass(frozen=True)
class ResolvedScene:
    """A scene with its variant chosen for the current flags, its cast
    resolved against the living, and its scene and option text rendered.
    Option outcomes are still templates until a choice is made."""

    id: str
    title: str
    text: str
    options: tuple[Option, ...]
    cast: dict[str, str | None] = field(default_factory=dict)


@dataclass(frozen=True)
class Scenario:
    colonists: tuple[Colonist, ...]
    scenes: tuple[Scene, ...]


@dataclass
class Outcome:
    scene_id: str
    option: Option
    dead: list[str]
    population_after: int
    text: str
    flags_set: list[str] = field(default_factory=list)


def _parse_cast(raw: object, known_names: set[str], where: str) -> tuple[Role, ...]:
    if not isinstance(raw, dict):
        raise ScenarioError(f"{where}: 'cast' must be an object mapping role keys to roles")
    roles = []
    for key, spec in raw.items():
        if not re.fullmatch(r"\w+", key):
            raise ScenarioError(f"{where}: cast role '{key}' must be letters, digits, or underscores")
        if key in BUILTINS:
            raise ScenarioError(f"{where}: cast role '{key}' clashes with the built-in placeholder")
        if not isinstance(spec, dict):
            raise ScenarioError(f"{where}: cast role '{key}' must be an object with 'prefer' and 'else'")
        if spec.get("prefer") not in known_names:
            raise ScenarioError(f"{where}: cast role '{key}' prefers unknown colonist '{spec.get('prefer')}'")
        if spec.get("else") not in FALLBACKS:
            raise ScenarioError(
                f"{where}: cast role '{key}' has 'else' of '{spec.get('else')}'; use one of {', '.join(FALLBACKS)}"
            )
        roles.append(Role(key=key, prefer=spec["prefer"], fallback=spec["else"]))
    return tuple(roles)


def _check_placeholders(text: str, cast: dict[str, Role], outcome: bool, where: str) -> None:
    for key in _PLACEHOLDER.findall(text):
        if key in BUILTINS:
            if key in OUTCOME_ONLY and not outcome:
                raise ScenarioError(f"{where}: '{{{key}}}' is only known in outcome text")
        elif key not in cast:
            raise ScenarioError(f"{where}: placeholder '{{{key}}}' is not a cast role or a built-in")
        elif cast[key].fallback == "none":
            raise ScenarioError(
                f"{where}: text uses '{{{key}}}' but that role may be unfilled ('else' is 'none'); "
                "give it a fallback or use '{dead}'"
            )


def _parse_option(raw: dict, cast: dict[str, Role], where: str) -> Option:
    for key in ("id", "text", "outcome"):
        if not isinstance(raw.get(key), str) or not raw[key]:
            raise ScenarioError(f"{where}: option missing string '{key}'")
    _check_placeholders(raw["text"], cast, False, f"{where}: option '{raw['id']}'")
    _check_placeholders(raw["outcome"], cast, True, f"{where}: option '{raw['id']}'")
    deaths = raw.get("deaths")
    if not isinstance(deaths, int) or isinstance(deaths, bool) or deaths < 0:
        raise ScenarioError(f"{where}: option '{raw['id']}' needs a non-negative integer 'deaths'")
    victims = tuple(raw.get("victims", ()))
    for v in victims:
        if v not in cast:
            raise ScenarioError(f"{where}: option '{raw['id']}' names victim '{v}', which is not a cast role")
    if len(victims) > deaths:
        raise ScenarioError(f"{where}: option '{raw['id']}' names more victims than deaths")
    sets = tuple(raw.get("sets", ()))
    if not all(isinstance(s, str) and s for s in sets):
        raise ScenarioError(f"{where}: option '{raw['id']}' has a non-string flag in 'sets'")
    return Option(
        id=raw["id"],
        text=raw["text"],
        deaths=deaths,
        outcome=raw["outcome"],
        victims=victims,
        sets=sets,
    )


def _parse_variant(raw: dict, known_names: set[str], where: str) -> Variant:
    if not isinstance(raw.get("text"), str) or not raw["text"]:
        raise ScenarioError(f"{where}: variant missing 'text'")
    roles = _parse_cast(raw.get("cast", {}), known_names, where)
    cast = {r.key: r for r in roles}
    _check_placeholders(raw["text"], cast, False, where)
    options = tuple(_parse_option(o, cast, where) for o in raw.get("options", []))
    ids = [o.id for o in options]
    if len(ids) < 2:
        raise ScenarioError(f"{where}: needs at least two options")
    if len(ids) != len(set(ids)):
        raise ScenarioError(f"{where}: duplicate option ids")
    if "refuse" not in ids:
        raise ScenarioError(f"{where}: every scene needs an option with id 'refuse'")
    requires = raw.get("requires")
    if requires is not None and (not isinstance(requires, str) or not requires):
        raise ScenarioError(f"{where}: 'requires' must be a non-empty string or omitted")
    return Variant(text=raw["text"], options=options, requires=requires, cast=roles)


def _parse_scene(raw: dict, known_names: set[str]) -> Scene:
    sid = raw.get("id")
    if not isinstance(sid, str) or not sid:
        raise ScenarioError("scene missing 'id'")
    where = f"scene '{sid}'"
    title = raw.get("title", sid)
    if "variants" in raw:
        variants = tuple(_parse_variant(v, known_names, where) for v in raw["variants"])
    else:
        variants = (_parse_variant(raw, known_names, where),)
    if not variants:
        raise ScenarioError(f"{where}: no variants")
    if not any(v.requires is None for v in variants):
        raise ScenarioError(f"{where}: needs one variant with no 'requires' as a fallback")
    return Scene(id=sid, title=title, variants=variants)


def load_scenario(path: str | Path) -> Scenario:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    raw_colonists = data.get("colonists", [])
    if not raw_colonists:
        raise ScenarioError("scenario needs at least one colonist")
    colonists = tuple(Colonist(name=c["name"], role=c.get("role", "")) for c in raw_colonists)
    names = [c.name for c in colonists]
    if len(names) != len(set(names)):
        raise ScenarioError("colonist names must be unique")
    scenes = tuple(_parse_scene(s, set(names)) for s in data.get("scenes", []))
    if not scenes:
        raise ScenarioError("scenario needs at least one scene")
    ids = [s.id for s in scenes]
    if len(ids) != len(set(ids)):
        raise ScenarioError("scene ids must be unique")
    return Scenario(colonists=colonists, scenes=scenes)


class Game:
    def __init__(self, scenario: Scenario):
        self.scenario = scenario
        self.alive: list[str] = [c.name for c in scenario.colonists]
        self.dead: list[str] = []
        self.flags: set[str] = set()
        self.scene_index = 0
        self.history: list[Outcome] = []

    @property
    def roles(self) -> dict[str, str]:
        return {c.name: c.role for c in self.scenario.colonists}

    @property
    def over(self) -> bool:
        return not self.alive or self.scene_index >= len(self.scenario.scenes)

    def current_scene(self) -> ResolvedScene | None:
        if self.over:
            return None
        scene = self.scenario.scenes[self.scene_index]
        variant = self._resolve_variant(scene)
        cast = self._resolve_cast(variant)
        values = {**cast, "alive_count": len(self.alive)}
        where = f"scene '{scene.id}'"
        return ResolvedScene(
            id=scene.id,
            title=scene.title,
            text=self._render(variant.text, values, where),
            options=tuple(replace(o, text=self._render(o.text, values, where)) for o in variant.options),
            cast=cast,
        )

    def _resolve_variant(self, scene: Scene) -> Variant:
        for v in scene.variants:
            if v.requires is not None and v.requires in self.flags:
                return v
        for v in scene.variants:
            if v.requires is None:
                return v
        raise ScenarioError(f"scene '{scene.id}' has no fallback variant")  # guarded at load

    def _resolve_cast(self, variant: Variant) -> dict[str, str | None]:
        """Preferred colonists first, then fallbacks in declaration order.
        Stand-ins come from the front of the alive list, away from the end
        where unnamed deaths are taken."""
        cast: dict[str, str | None] = {r.key: r.prefer if r.prefer in self.alive else None for r in variant.cast}
        for r in variant.cast:
            if cast[r.key] is None and r.fallback != "none":
                taken = set(cast.values()) if r.fallback == "any_except_cast" else set()
                cast[r.key] = next((name for name in self.alive if name not in taken), None)
        return cast

    def _render(self, template: str, values: dict[str, object], where: str) -> str:
        """Fill placeholders. Refuses to name anyone in self.dead, so call
        it for an outcome before that choice's dead are recorded."""
        def fill(match: re.Match) -> str:
            value = values.get(match.group(1))
            if value is None:
                raise ScenarioError(f"{where}: no living colonist can fill role '{match.group(1)}' used in the text")
            return str(value)

        text = _PLACEHOLDER.sub(fill, template)
        for name in self.dead:
            if re.search(rf"\b{re.escape(name)}\b", text):
                raise ScenarioError(f"{where}: rendered text names dead colonist '{name}': {text!r}")
        return text

    def choose(self, option_id: str) -> Outcome:
        scene = self.current_scene()
        if scene is None:
            raise ValueError("game is over")
        option = next((o for o in scene.options if o.id == option_id), None)
        if option is None:
            raise ValueError(f"unknown option '{option_id}' for scene '{scene.id}'")

        dead: list[str] = []
        for role in option.victims:
            name = scene.cast[role]
            if name in self.alive and len(dead) < option.deaths:
                self.alive.remove(name)
                dead.append(name)
        while len(dead) < option.deaths and self.alive:
            dead.append(self.alive.pop())
        # Rendered only now, from the real dead, so the story matches the state.
        values = {**scene.cast, "dead": ", ".join(dead) or "no one", "dead_count": len(dead),
                  "alive_count": len(self.alive)}
        text = self._render(option.outcome, values, f"scene '{scene.id}': option '{option.id}'")
        self.dead.extend(dead)

        new_flags = [f for f in option.sets if f not in self.flags]
        self.flags.update(option.sets)
        self.scene_index += 1

        outcome = Outcome(
            scene_id=scene.id,
            option=option,
            dead=dead,
            population_after=len(self.alive),
            text=text,
            flags_set=new_flags,
        )
        self.history.append(outcome)
        return outcome

    def ending(self) -> str:
        if not self.alive:
            return (
                f"GAME OVER. Haven colony has no survivors. "
                f"{len(self.dead)} colonists died across {len(self.history)} decisions."
            )
        if self.over:
            flags = ", ".join(sorted(self.flags)) or "none"
            return (
                f"The colony survived with {len(self.alive)} of "
                f"{len(self.scenario.colonists)} colonists: {', '.join(self.alive)}. "
                f"Flags: {flags}."
            )
        return "The game is still in progress."
