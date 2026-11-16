"""Checks for the thin Gymnasium adapter (no model). Usage: python test_pchie_gym.py --server PATH"""
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import pchie_gym  # noqa: E402

passed = failed = 0


def check(ok, what):
    global passed, failed
    if ok:
        passed += 1
    else:
        failed += 1
        print("FAIL:", what)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--server", required=True)
    a = ap.parse_args()
    names = pchie_gym.list_plugs(a.server)
    check(sorted(names) == sorted(pchie_gym.PLUGS.values()), "list: every plug is offered")

    env = pchie_gym.PchieEnv("lithokraft.maze.b1", server=a.server, max_steps=5)
    check(env.action_names == ["north", "east", "south", "west", "pickup", "use"], "maze: research action order")
    obs, info = env.reset(seed=6)
    check(env.observation_space.contains(obs), "maze: observation fits its space")
    check("from your start" in obs["text"] and "layout" not in obs["text"], "maze: own senses only, no grading")
    trace = []
    for k in range(5):
        obs, r, term, trunc, inf = env.step(k % 4)
        trace.append((obs["text"], r, term, trunc, inf["state"]))
    check(trace[-1][3] is True, "maze: max_steps truncates")
    g = env.grade()
    check(set(g) == {"success", "progress", "target"} and g["target"] == 17, "maze: grade after the episode (seed 6 distance 17)")
    obs2, info2 = env.reset(seed=6)
    trace2 = []
    for k in range(5):
        o, r, term, trunc, inf = env.step(k % 4)
        trace2.append((o["text"], r, term, trunc, inf["state"]))
    check(trace == trace2 and info["state"] == info2["state"], "maze: same seed, same episode")
    _, info3 = env.reset()
    check(info3["seed"] == 7, "maze: the next reset without a seed moves to the next seed")
    try:
        env.step(99)
        check(False, "maze: an action out of range is refused")
    except pchie_gym.PlugError:
        check(True, "maze: an action out of range is refused")
    env.close()

    c = pchie_gym.PchieEnv("craftax_like.grid", server=a.server)
    obs, _ = c.reset(seed=5)
    check(len(obs["glyphs"].split("\n")) == 5 and "@" in obs["glyphs"], "craftax-like: 5x5 glyph view")
    check(c.action_space.n == 8, "craftax-like: 8 actions")
    c.close()
    try:
        pchie_gym.PchieEnv("nethack", server=a.server)
        check(False, "an unknown plug is refused")
    except pchie_gym.PlugError:
        check(True, "an unknown plug is refused")
    check(pchie_gym.register_all(a.server) == pchie_gym.HAVE_GYMNASIUM, "register_all matches whether Gymnasium is installed")
    print(f"gym adapter: {passed} passed, {failed} failed (gymnasium {'installed' if pchie_gym.HAVE_GYMNASIUM else 'not installed, stand-ins used'})")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
