"""Thin Gymnasium adapter for the Bolt PCHIE harness game plugs.

Off the official path and never in a hashed path: it drives ``pchie_plug_server`` over stdin/stdout
(one JSON object per line) and only passes observations, actions and rewards through. Nothing here
writes a chain, reads a key, or opens a network connection.

    import pchie_gym
    env = pchie_gym.PchieEnv("lithokraft.maze.b1", server="path/to/pchie_plug_server.exe")
    obs, info = env.reset(seed=6)
    obs, reward, terminated, truncated, info = env.step(1)

Works with Gymnasium when it is installed (``PchieEnv`` is then a ``gymnasium.Env`` and
``register_all()`` registers ids such as ``Pchie/LithoKraftMazeB1-v1``). Without Gymnasium the same
class still runs with small stand-ins, so the adapter can be tested anywhere.

MIT License. Copyright (c) 2026 Jaydee Wetwork / Wet Wafer Works.
"""
from __future__ import annotations

import json
import os
import string
import subprocess
from typing import Any, Dict, Optional, Tuple

try:  # pragma: no cover - depends on the machine
    import gymnasium as _gym
    from gymnasium import spaces as _spaces

    HAVE_GYMNASIUM = True
    _EnvBase = _gym.Env
except ImportError:  # small stand-ins with the same shape
    HAVE_GYMNASIUM = False

    class _EnvBase:  # type: ignore[no-redef]
        metadata: Dict[str, Any] = {}

        def reset(self, *, seed=None, options=None):
            return None

        def close(self):
            return None

    class _Discrete:
        def __init__(self, n: int):
            self.n = int(n)

        def contains(self, x) -> bool:
            return isinstance(x, int) and 0 <= x < self.n

    class _Text:
        def __init__(self, max_length: int, min_length: int = 0, charset: str = string.printable):
            self.max_length, self.min_length, self.charset = max_length, min_length, set(charset)

        def contains(self, x) -> bool:
            return isinstance(x, str) and self.min_length <= len(x) <= self.max_length and all(c in self.charset for c in x)

    class _Dict:
        def __init__(self, spaces: Dict[str, Any]):
            self.spaces = spaces

        def contains(self, x) -> bool:
            return isinstance(x, dict) and all(k in x and s.contains(x[k]) for k, s in self.spaces.items())

    class _SpacesShim:
        Discrete = _Discrete
        Text = _Text
        Dict = _Dict

    _spaces = _SpacesShim()  # type: ignore[assignment]

PLUGS = {
    "Pchie/LithoKraftMazeB1-v1": "lithokraft.maze.b1",
    "Pchie/LithoKraftMazeStageA-v1": "lithokraft.maze.stagea",
    "Pchie/LithoLatticeTestbedStub-v0": "litholattice.testbed.stub",
    "Pchie/CraftaxLikeGrid-v1": "craftax_like.grid",
}

_CHARSET = string.printable


def default_server() -> str:
    """PCHIE_PLUG_SERVER, or pchie_plug_server(.exe) next to this file."""
    env = os.environ.get("PCHIE_PLUG_SERVER")
    if env:
        return env
    here = os.path.dirname(os.path.abspath(__file__))
    exe = "pchie_plug_server.exe" if os.name == "nt" else "pchie_plug_server"
    return os.path.join(here, exe)


class PlugError(RuntimeError):
    pass


class PchieEnv(_EnvBase):
    """One harness plug as a Gymnasium environment.

    Observation: {"text": str, "glyphs": str}. Action: an index into ``action_names``.
    ``terminated`` is the game's own end (goal or a fall); ``truncated`` is ``max_steps``.
    Grading (``grade()``) is for after the episode and is never part of the observation.
    """

    metadata = {"render_modes": ["ansi"]}

    def __init__(self, plug: str = "lithokraft.maze.b1", server: Optional[str] = None,
                 max_steps: Optional[int] = None, render_mode: Optional[str] = None):
        self.plug = plug
        self.render_mode = render_mode
        self.max_steps = max_steps
        self._proc = subprocess.Popen([server or default_server()], stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                                      stderr=subprocess.DEVNULL, text=True, encoding="utf-8", bufsize=1)
        info = self._call({"cmd": "make", "plug": plug})["info"]
        self.plug_info = info
        self.action_names = list(info["actions"])
        self.action_space = _spaces.Discrete(len(self.action_names))
        self.observation_space = _spaces.Dict({
            "text": _spaces.Text(max_length=4096, min_length=0, charset=_CHARSET),
            "glyphs": _spaces.Text(max_length=4096, min_length=0, charset=_CHARSET),
        })
        self._seed: Optional[int] = None
        self._episodes = 0
        self._steps = 0
        self._last: Dict[str, Any] = {}

    # -- protocol --
    def _call(self, msg: Dict[str, Any]) -> Dict[str, Any]:
        if self._proc is None or self._proc.poll() is not None:
            raise PlugError("plug server is not running")
        assert self._proc.stdin is not None and self._proc.stdout is not None
        self._proc.stdin.write(json.dumps(msg) + "\n")
        self._proc.stdin.flush()
        line = self._proc.stdout.readline()
        if not line:
            raise PlugError("plug server closed")
        out = json.loads(line)
        if not out.get("ok"):
            raise PlugError(out.get("err", "plug error"))
        return out

    @staticmethod
    def _obs(o: Dict[str, Any]) -> Dict[str, str]:
        return {"text": o["text"], "glyphs": o["glyphs"]}

    # -- Gymnasium API --
    def reset(self, *, seed: Optional[int] = None, options: Optional[Dict[str, Any]] = None):
        if HAVE_GYMNASIUM:
            super().reset(seed=seed)
        if seed is not None:
            self._seed, self._episodes = int(seed), 0
        base = 1 if self._seed is None else self._seed
        episode_seed = base + self._episodes
        self._episodes += 1
        self._steps = 0
        r = self._call({"cmd": "reset", "seed": episode_seed})
        self._last = r
        return self._obs(r["obs"]), {"seed": episode_seed, "state": r["state"], "plug": self.plug}

    def step(self, action) -> Tuple[Dict[str, str], int, bool, bool, Dict[str, Any]]:
        a = int(action)
        if not self.action_space.contains(a):
            raise PlugError(f"action {action!r} is outside 0..{len(self.action_names) - 1}")
        r = self._call({"cmd": "step", "action": a})
        self._last = r
        self._steps += 1
        truncated = self.max_steps is not None and self._steps >= self.max_steps and not r["terminated"]
        info = {"action": self.action_names[a], "action_ok": r["action_ok"], "success": r["success"], "state": r["state"]}
        return self._obs(r["obs"]), int(r["reward"]), bool(r["terminated"]), bool(truncated), info

    def render(self):
        o = self._last.get("obs", {})
        return (o.get("text", "") + ("\n" + o["glyphs"] if o.get("glyphs") else "")) if o else ""

    def grade(self) -> Dict[str, Any]:
        r = self._call({"cmd": "grade"})
        return {"success": r["success"], "progress": r["progress"], "target": r["target"]}

    def close(self):
        if self._proc is not None:
            try:
                if self._proc.poll() is None:
                    self._call({"cmd": "close"})
            except (PlugError, OSError, ValueError):
                pass
            try:
                self._proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self._proc.kill()
            self._proc = None


def list_plugs(server: Optional[str] = None):
    """Names the server offers."""
    p = subprocess.run([server or default_server()], input='{"cmd":"list"}\n{"cmd":"close"}\n', capture_output=True,
                       text=True, timeout=30)
    return json.loads(p.stdout.splitlines()[0])["plugs"]


def register_all(server: Optional[str] = None) -> bool:
    """Registers every plug with Gymnasium. False when Gymnasium is not installed."""
    if not HAVE_GYMNASIUM:
        return False
    for env_id, plug in PLUGS.items():
        if env_id not in _gym.registry:
            _gym.register(id=env_id, entry_point="pchie_gym:PchieEnv", kwargs={"plug": plug, "server": server})
    return True
