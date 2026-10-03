import json
import re
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from game import Game, ScenarioError, load_scenario  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]


def scenario_dict(**overrides):
    base = {
        "colonists": [
            {"name": "Ann", "role": "medic"},
            {"name": "Bo", "role": "farmer"},
            {"name": "Cy", "role": "teen"},
            {"name": "Di", "role": "elder"},
        ],
        "scenes": [
            {
                "id": "one",
                "title": "One",
                "text": "First choice.",
                "cast": {"medic": {"prefer": "Ann", "else": "none"}},
                "options": [
                    {"id": "kind", "text": "Be kind", "deaths": 0, "sets": ["kind"], "outcome": "ok"},
                    {"id": "harsh", "text": "Be harsh", "deaths": 1, "victims": ["medic"], "outcome": "{dead} dies"},
                    {"id": "refuse", "text": "Decline", "deaths": 2, "outcome": "two die"},
                ],
            },
            {
                "id": "two",
                "title": "Two",
                "variants": [
                    {
                        "requires": "kind",
                        "text": "Kind branch.",
                        "options": [
                            {"id": "a", "text": "A", "deaths": 0, "outcome": "ok"},
                            {"id": "refuse", "text": "Decline", "deaths": 0, "outcome": "ok"},
                        ],
                    },
                    {
                        "text": "Default branch.",
                        "options": [
                            {"id": "b", "text": "B", "deaths": 10, "outcome": "wipe"},
                            {"id": "refuse", "text": "Decline", "deaths": 0, "outcome": "ok"},
                        ],
                    },
                ],
            },
        ],
    }
    base.update(overrides)
    return base


def write(tmp_path, data):
    p = tmp_path / "s.json"
    p.write_text(json.dumps(data))
    return p


def test_load_valid_scenario(tmp_path):
    sc = load_scenario(write(tmp_path, scenario_dict()))
    assert [c.name for c in sc.colonists] == ["Ann", "Bo", "Cy", "Di"]
    assert len(sc.scenes) == 2


def test_scene_without_refuse_is_rejected(tmp_path):
    data = scenario_dict()
    data["scenes"][0]["options"].pop()  # drop refuse
    with pytest.raises(ScenarioError, match="refuse"):
        load_scenario(write(tmp_path, data))


def test_unknown_victim_is_rejected(tmp_path):
    data = scenario_dict()
    data["scenes"][0]["options"][1]["victims"] = ["Nobody"]
    with pytest.raises(ScenarioError, match="Nobody"):
        load_scenario(write(tmp_path, data))


def test_named_victims_die_first_then_from_end(tmp_path):
    game = Game(load_scenario(write(tmp_path, scenario_dict())))
    out = game.choose("refuse")  # 2 deaths, no named victims
    assert out.dead == ["Di", "Cy"]
    assert game.alive == ["Ann", "Bo"]


def test_named_victim_counts_toward_deaths(tmp_path):
    game = Game(load_scenario(write(tmp_path, scenario_dict())))
    out = game.choose("harsh")
    assert out.dead == ["Ann"]
    assert game.alive == ["Bo", "Cy", "Di"]


def test_flags_select_variant(tmp_path):
    game = Game(load_scenario(write(tmp_path, scenario_dict())))
    game.choose("kind")
    assert "kind" in game.flags
    scene = game.current_scene()
    assert scene.text == "Kind branch."
    assert [o.id for o in scene.options] == ["a", "refuse"]


def test_default_variant_when_flag_missing(tmp_path):
    game = Game(load_scenario(write(tmp_path, scenario_dict())))
    game.choose("harsh")
    assert game.current_scene().text == "Default branch."


def test_deaths_capped_and_game_over(tmp_path):
    game = Game(load_scenario(write(tmp_path, scenario_dict())))
    game.choose("harsh")
    out = game.choose("b")  # 10 deaths, only 3 alive
    assert len(out.dead) == 3
    assert game.alive == []
    assert game.over
    assert game.current_scene() is None
    assert "game over" in game.ending().lower()


def test_survive_all_scenes(tmp_path):
    game = Game(load_scenario(write(tmp_path, scenario_dict())))
    game.choose("kind")
    game.choose("a")
    assert game.over
    assert game.current_scene() is None
    assert "survived" in game.ending().lower()


def test_invalid_option_id_rejected(tmp_path):
    game = Game(load_scenario(write(tmp_path, scenario_dict())))
    with pytest.raises(ValueError):
        game.choose("nope")


def casting_dict(cast, text="Now.", options=None):
    """Scene 'before' can kill Ann; scene 'now' is the one under test."""
    data = scenario_dict()
    data["scenes"] = [
        {
            "id": "before",
            "text": "Before.",
            "cast": {"ann": {"prefer": "Ann", "else": "none"}},
            "options": [
                {"id": "spare", "text": "Spare", "deaths": 0, "outcome": "ok"},
                {"id": "kill_ann", "text": "Kill", "deaths": 1, "victims": ["ann"], "outcome": "ok"},
                {"id": "refuse", "text": "Decline", "deaths": 0, "outcome": "ok"},
            ],
        },
        {
            "id": "now",
            "text": text,
            "cast": cast,
            "options": options or [
                {"id": "go", "text": "Go", "deaths": 0, "outcome": "ok"},
                {"id": "refuse", "text": "Decline", "deaths": 0, "outcome": "ok"},
            ],
        },
    ]
    return data


def casting_game(tmp_path, first_choice, cast, **kwargs):
    game = Game(load_scenario(write(tmp_path, casting_dict(cast, **kwargs))))
    game.choose(first_choice)
    return game


def go_option(**fields):
    return [
        {"id": "go", "text": "Go", "deaths": 0, "outcome": "ok", **fields},
        {"id": "refuse", "text": "Decline", "deaths": 0, "outcome": "ok"},
    ]


def test_cast_uses_preferred_colonist_when_alive(tmp_path):
    game = casting_game(tmp_path, "spare", {"lead": {"prefer": "Ann", "else": "any"}}, text="{lead} speaks.")
    scene = game.current_scene()
    assert scene.text == "Ann speaks."
    assert scene.cast == {"lead": "Ann"}


def test_cast_any_falls_back_to_first_living_colonist(tmp_path):
    game = casting_game(tmp_path, "kill_ann", {"lead": {"prefer": "Ann", "else": "any"}}, text="{lead} speaks.")
    assert game.current_scene().text == "Bo speaks."


def test_cast_any_except_cast_skips_colonists_in_other_roles(tmp_path):
    cast = {
        "lead": {"prefer": "Ann", "else": "any_except_cast"},
        "second": {"prefer": "Bo", "else": "any"},
    }
    game = casting_game(tmp_path, "kill_ann", cast, text="{lead} and {second}.")
    scene = game.current_scene()
    assert scene.text == "Cy and Bo."
    assert scene.cast == {"lead": "Cy", "second": "Bo"}


def test_cast_none_leaves_role_unfilled(tmp_path):
    game = casting_game(tmp_path, "kill_ann", {"lead": {"prefer": "Ann", "else": "none"}})
    assert game.current_scene().cast == {"lead": None}


def test_option_text_and_alive_count_are_rendered(tmp_path):
    game = casting_game(tmp_path, "kill_ann", {"lead": {"prefer": "Ann", "else": "any"}},
                        text="{alive_count} remain.", options=go_option(text="Send {lead}"))
    scene = game.current_scene()
    assert scene.text == "3 remain."
    assert scene.options[0].text == "Send Bo"


def test_victim_role_dies_then_rest_from_end(tmp_path):
    game = casting_game(tmp_path, "spare", {"lead": {"prefer": "Ann", "else": "none"}},
                        options=go_option(deaths=2, victims=["lead"]))
    assert game.choose("go").dead == ["Ann", "Di"]


def test_unfilled_victim_role_is_replaced_from_end(tmp_path):
    game = casting_game(tmp_path, "kill_ann", {"lead": {"prefer": "Ann", "else": "none"}},
                        options=go_option(deaths=2, victims=["lead"]))
    assert game.choose("go").dead == ["Di", "Cy"]
    assert game.alive == ["Bo"]


def test_outcome_is_rendered_from_the_real_dead(tmp_path):
    game = casting_game(tmp_path, "kill_ann", {"lead": {"prefer": "Ann", "else": "none"}},
                        options=go_option(deaths=1, victims=["lead"],
                                          outcome="{dead} died. {dead_count} gone, {alive_count} left."))
    out = game.choose("go")
    assert out.dead == ["Di"]
    assert out.text == "Di died. 1 gone, 2 left."


def test_outcome_with_no_deaths_says_no_one(tmp_path):
    game = casting_game(tmp_path, "spare", {}, options=go_option(outcome="Lost: {dead} ({dead_count})."))
    assert game.choose("go").text == "Lost: no one (0)."


def test_outcome_may_name_a_cast_victim_who_just_died(tmp_path):
    game = casting_game(tmp_path, "kill_ann", {"lead": {"prefer": "Ann", "else": "any"}},
                        options=go_option(deaths=1, victims=["lead"], outcome="{lead} is gone."))
    out = game.choose("go")
    assert out.dead == ["Bo"]
    assert out.text == "Bo is gone."


def test_unknown_placeholder_is_rejected(tmp_path):
    data = casting_dict({"lead": {"prefer": "Ann", "else": "any"}}, text="{leader} speaks.")
    with pytest.raises(ScenarioError, match="leader"):
        load_scenario(write(tmp_path, data))


def test_unknown_placeholder_in_outcome_is_rejected(tmp_path):
    data = casting_dict({}, options=go_option(outcome="{ghost} waves."))
    with pytest.raises(ScenarioError, match="ghost"):
        load_scenario(write(tmp_path, data))


def test_dead_placeholder_outside_outcome_is_rejected(tmp_path):
    data = casting_dict({}, options=go_option(text="Mourn {dead}"))
    with pytest.raises(ScenarioError, match="only .* outcome"):
        load_scenario(write(tmp_path, data))


def test_text_cannot_use_role_that_may_be_unfilled(tmp_path):
    data = casting_dict({"lead": {"prefer": "Ann", "else": "none"}}, text="{lead} speaks.")
    with pytest.raises(ScenarioError, match="lead.*unfilled"):
        load_scenario(write(tmp_path, data))


def test_unknown_preferred_colonist_is_rejected(tmp_path):
    data = casting_dict({"lead": {"prefer": "Zed", "else": "any"}})
    with pytest.raises(ScenarioError, match="Zed"):
        load_scenario(write(tmp_path, data))


def test_unknown_fallback_is_rejected(tmp_path):
    data = casting_dict({"lead": {"prefer": "Ann", "else": "whoever"}})
    with pytest.raises(ScenarioError, match="whoever"):
        load_scenario(write(tmp_path, data))


def test_role_key_cannot_shadow_builtin(tmp_path):
    data = casting_dict({"dead": {"prefer": "Ann", "else": "any"}})
    with pytest.raises(ScenarioError, match="dead"):
        load_scenario(write(tmp_path, data))


def test_text_using_role_nobody_can_fill_raises(tmp_path):
    cast = {
        "lead": {"prefer": "Ann", "else": "any_except_cast"},
        "b": {"prefer": "Bo", "else": "none"},
        "c": {"prefer": "Cy", "else": "none"},
        "d": {"prefer": "Di", "else": "none"},
    }
    game = casting_game(tmp_path, "kill_ann", cast, text="{lead} speaks.")
    with pytest.raises(ScenarioError, match="lead"):
        game.current_scene()


def test_rendering_a_dead_name_raises(tmp_path):
    game = casting_game(tmp_path, "kill_ann", {}, text="Ann waves.")
    with pytest.raises(ScenarioError, match="dead colonist 'Ann'"):
        game.current_scene()


def test_real_scenes_file_loads():
    sc = load_scenario(ROOT / "scenes.json")
    assert len(sc.colonists) == 10
    assert len(sc.scenes) == 6
    for scene in sc.scenes:
        for variant in scene.variants:
            ids = [o.id for o in variant.options]
            assert "refuse" in ids
            assert len(ids) == len(set(ids))


def names_in(text, names):
    return {n for n in names if re.search(rf"\b{re.escape(n)}\b", text)}


def fork(game):
    twin = Game(game.scenario)
    twin.alive, twin.dead, twin.flags = list(game.alive), list(game.dead), set(game.flags)
    twin.scene_index = game.scene_index
    return twin


def walk(game, path=()):
    """Play every option from here down, checking each text shown on the
    way. Returns the number of complete paths."""
    if game.over:
        return 1
    scene = game.current_scene()
    dead_before = list(game.dead)
    for text in (scene.text, *(o.text for o in scene.options)):
        assert not names_in(text, dead_before), (path, scene.id, text)
    paths = 0
    for option in scene.options:
        branch = fork(game)
        out = branch.choose(option.id)
        here = path + (option.id,)
        assert not names_in(out.text, dead_before), (here, out.text)
        # The story names exactly who died: no one missing, no one extra.
        assert names_in(out.text, branch.dead) == set(out.dead), (here, out.text)
        paths += walk(branch, here)
    return paths


def test_real_scenes_never_name_the_dead_on_any_path():
    assert walk(Game(load_scenario(ROOT / "scenes.json"))) > 1000


def test_real_scenes_reported_bug_stays_fixed():
    """Ruth dies in 'medicine'; 'water' must name who really dies next, and
    'aftermath' must not be led by a dead radio operator."""
    game = Game(load_scenario(ROOT / "scenes.json"))
    game.choose("return")
    assert game.choose("medic_first").dead == ["Ruth"]
    water = game.choose("tell_dez")
    assert water.dead == ["Sol"]
    assert "Sol" in water.text and "Ruth" not in water.text
    aftermath = game.current_scene()
    # Not the dead, not the medic, and not Dez, who was told and has nothing to find.
    assert aftermath.cast["finder"] not in (None, "Sol", "Ruth", "Mara", "Dez")
    assert "Sol" not in aftermath.text
    assert aftermath.cast["finder"] in aftermath.text
