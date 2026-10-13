"""What Slate's compile sends to the model (app/slate.py): python3 tests/test_slate_compile.py. No GPU, no network."""
import os
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
os.environ.setdefault("STUDIO_HOME", str(Path(tempfile.mkdtemp()) / "TensorFoldStudio"))
sys.path.insert(0, str(ROOT / "app"))
import negation as N  # noqa: E402
import server  # noqa: E402  (attaches Slate to the Studio's prompt builders)
import slate  # noqa: E402


def film(**shot) -> dict:
    """One scene with one shot: Leo and Mia on the walkway, as a production run wrote them."""

    one = {"id": "shot-1", "name": "1.4 First Smiles", "intent": "0:12.1-0:16.1 (2 bars, 5 s). The audience sees the first spark.",
           "spec": {"durationSec": 5, "fps": 24, "aspectRatio": "16:9", "size": "MS", "angle": "eye level", "lens": "50mm", "movement": "locked"},
           "prompt": "# Subject\nLeo hands Mia the stack of books and they smile.\n# Composition\nMedium two-shot in profile.\n# Lighting\nLate-morning sun through leaves.",
           "beatSheet": None, "mutedLines": [], "takes": [],
           "tf": {"start": "text", "cast": ["char-leo", "char-mia"], "location": "loc-1", "lines": [], "soundscape": "Leaf rustle and distant birdsong.", "music": "N/A", "cue": "", "seed": 42}}
    one.update(shot)
    return {"id": "p-1", "name": "Hesitation", "defaults": {"durationSec": 8}, "tf": {"res": "480p", "quality": "fh8"},
            "characters": [{"id": "char-leo", "name": "Leo", "age": "21", "gender": "male", "hair": "short sandy-blond hair", "clothing": "a rust-orange work jacket"},
                           {"id": "char-mia", "name": "Mia", "age": "20", "gender": "female", "hair": "long pale-blonde hair", "clothing": "an emerald-green cardigan"}],
            "locations": [{"id": "loc-1", "name": "Campus Walkway", "description": "A paved walkway under old trees"}],
            "scenes": [{"id": "scene-1", "name": "Scene 1", "shots": [one]}]}


def test_a_shots_intent_stays_out_of_the_prompt():
    made = slate.compile_shot(film(), "shot-1")
    assert "0:12.1" not in made["prompt"] and "audience" not in made["prompt"] and "2 bars" not in made["prompt"]
    assert "Leo hands Mia the stack of books" in made["prompt"]  # what happens is the Subject section's
    # with a beat sheet the beats are the action, and the intent is still a note
    beats = slate.compile_shot(film(beatSheet=[{"from": 0, "to": 2, "text": "Leo holds out the books."}, {"from": 2, "to": 5, "text": "Mia takes them and smiles."}]), "shot-1")
    assert "First, leo holds out the books. Finally, mia takes them and smiles." in beats["prompt"] and "audience" not in beats["prompt"]
    # an empty shot still compiles, and says so
    empty = slate.compile_shot(film(prompt=""), "shot-1")
    assert "audience" not in empty["prompt"] and any("no prompt yet" in w for w in empty["warnings"])


def test_a_compiled_prompt_forbids_nothing_by_name():
    """Slate's own words (a locked camera, a shot without lines) say what is there, so the negation check has nothing
    to list in a prompt compiled from a clean shot, with none of the app's sentences passed over."""

    had, N.OWN = N.OWN, ()
    try:
        made = slate.compile_shot(film(), "shot-1")
        prompt = made["prompt"]
        assert "Static shot on a locked, still camera." in prompt and "does not move" not in prompt
        assert "Mouths stay closed." in prompt and "Only ambient sound is heard." in prompt
        assert "Nobody speaks" not in prompt and "No voices" not in prompt
        assert N.find(prompt.replace("non_diegetic_music: N/A", "")) == [] and N.check(prompt)["negations"] == []
        assert not any("no negative prompt" in w for w in made["warnings"])
        for move in slate.MOVES.values():
            assert N.find(move) == [], move
        # a shot with nobody cast and no person in its words: no mouth is named into it
        empty = film(prompt="# Subject\nWaves roll onto an empty beach at dusk.", tf={"start": "text", "cast": [], "location": "", "lines": [], "soundscape": "", "music": "N/A", "seed": 1})
        assert "Mouths" not in slate.compile_shot(empty, "shot-1")["prompt"]
        # the director's own negation is still said, never rewritten
        told = slate.compile_shot(film(prompt="# Subject\nLeo walks. No cars anywhere."), "shot-1")
        assert "No cars anywhere." in told["prompt"] and any("No cars" in w for w in told["warnings"])
    finally:
        N.OWN = had


if __name__ == "__main__":
    for name, test in list(globals().items()):
        if name.startswith("test_"):
            test()
            print("ok", name)
