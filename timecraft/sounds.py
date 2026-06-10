# sounds.py
#
# Procedural sound synthesis for TimeCraft.
#
# All sounds are generated from pure Python (wave + struct + math) on first
# launch and cached as .wav files in a sounds/ subdirectory.  No external
# assets or dependencies are required.
#
# Usage:
#   from sounds import SoundManager
#   sm = SoundManager()                 # generates wavs if needed, loads all
#   sm.play('stone_break')              # fire and forget
#   sm.play('wood_place')

import wave
import struct
import math
import os
import pyglet.media

SAMPLE_RATE = 22050
MAX_AMP     = 32767

# ---------------------------------------------------------------------------
# Synthesis primitives
# ---------------------------------------------------------------------------

def _sine(freq, t):
    return math.sin(2 * math.pi * freq * t)

def _noise(t, seed=0):
    """Deterministic pseudo-noise via overlapping sine products."""
    return (math.sin(t * 2000.3 + seed) *
            math.sin(t * 3700.7 + seed * 0.3) *
            math.sin(t * 1200.1))

def _write_wav(path, samples):
    with wave.open(path, 'w') as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(SAMPLE_RATE)
        for s in samples:
            s = max(-MAX_AMP, min(MAX_AMP, int(s * MAX_AMP)))
            w.writeframes(struct.pack('<h', s))

# ---------------------------------------------------------------------------
# Per-material synthesisers
# Break = louder, slightly longer.  Place = shorter, softer.
# ---------------------------------------------------------------------------

def _stone(duration, volume, seed=0):
    """Hard click: broadband noise burst with fast exponential decay."""
    n = int(SAMPLE_RATE * duration)
    out = []
    for i in range(n):
        t = i / SAMPLE_RATE
        env = math.exp(-t * 40)
        s   = _noise(t, seed) * env * volume
        s  += _sine(120, t) * math.exp(-t * 60) * 0.3 * volume
        out.append(s)
    return out

def _dirt(duration, volume, seed=0):
    """Soft thud: low freq sine + noise."""
    n = int(SAMPLE_RATE * duration)
    out = []
    for i in range(n):
        t = i / SAMPLE_RATE
        env = math.exp(-t * 25)
        s   = _sine(80, t) * env * volume * 0.7
        s  += _noise(t, seed) * env * volume * 0.4
        out.append(s)
    return out

def _wood(duration, volume, seed=0):
    """Woody clunk: mid-range resonance."""
    n = int(SAMPLE_RATE * duration)
    out = []
    for i in range(n):
        t = i / SAMPLE_RATE
        env = math.exp(-t * 20)
        s   = _sine(200, t) * env * volume * 0.5
        s  += _sine(320, t) * math.exp(-t * 30) * volume * 0.3
        s  += _noise(t, seed) * math.exp(-t * 50) * volume * 0.2
        out.append(s)
    return out

def _glass(duration, volume, seed=0):
    """Bright tink: high harmonics with slow decay."""
    n = int(SAMPLE_RATE * duration)
    out = []
    for i in range(n):
        t = i / SAMPLE_RATE
        env = math.exp(-t * 12)
        s   = _sine(1200, t) * env * volume * 0.6
        s  += _sine(2400, t) * math.exp(-t * 20) * volume * 0.25
        s  += _sine(600,  t) * math.exp(-t * 8)  * volume * 0.15
        out.append(s)
    return out

def _leaf(duration, volume, seed=0):
    """Soft rustle: noise with gentle attack."""
    n = int(SAMPLE_RATE * duration)
    out = []
    for i in range(n):
        t = i / SAMPLE_RATE
        env = math.exp(-t * 30) * (1 - math.exp(-t * 200))
        s   = _noise(t, seed) * env * volume * 0.8
        s  += _noise(t, seed + 1) * env * volume * 0.2
        out.append(s)
    return out

def _water(duration, volume, seed=0):
    """Splash: pitch-dropping sine + modulated noise."""
    n = int(SAMPLE_RATE * duration)
    out = []
    for i in range(n):
        t    = i / SAMPLE_RATE
        env  = math.exp(-t * 18) * (1 - math.exp(-t * 100))
        freq = 600 * math.exp(-t * 5)
        s    = _sine(freq, t) * env * volume * 0.5
        s   += _noise(t, seed) * env * volume * 0.5
        out.append(s)
    return out

# ---------------------------------------------------------------------------
# Sound definitions
# (name, synth_fn, duration, volume, seed)
# ---------------------------------------------------------------------------

_SOUND_DEFS = {
    'stone_break': (_stone, 0.10, 0.70, 0),
    'stone_place': (_stone, 0.06, 0.45, 1),
    'dirt_break':  (_dirt,  0.12, 0.55, 2),
    'dirt_place':  (_dirt,  0.08, 0.40, 3),
    'wood_break':  (_wood,  0.14, 0.65, 4),
    'wood_place':  (_wood,  0.09, 0.45, 5),
    'glass_break': (_glass, 0.20, 0.55, 6),
    'glass_place': (_glass, 0.12, 0.35, 7),
    'leaf_break':  (_leaf,  0.10, 0.40, 8),
    'leaf_place':  (_leaf,  0.07, 0.28, 9),
    'water_place': (_water, 0.15, 0.40, 10),
    'water_break': (_water, 0.18, 0.50, 11),
    'sand_break':  (_dirt,  0.10, 0.45, 12),
    'sand_place':  (_dirt,  0.07, 0.32, 13),
}


def generate_sounds(sounds_dir):
    """Write all .wav files to *sounds_dir*, skipping any that already exist."""
    os.makedirs(sounds_dir, exist_ok=True)
    for name, (fn, dur, vol, seed) in _SOUND_DEFS.items():
        path = os.path.join(sounds_dir, f'{name}.wav')
        if not os.path.exists(path):
            _write_wav(path, fn(dur, vol, seed))


# ---------------------------------------------------------------------------
# SoundManager
# ---------------------------------------------------------------------------

class SoundManager:
    """Loads and plays TimeCraft's procedural sounds.

    All .wav files are synthesised on first use and cached in a sounds/
    directory next to this module.  Playback errors are silently swallowed
    so a missing audio device never crashes the game.
    """

    def __init__(self, sounds_dir=None):
        if sounds_dir is None:
            sounds_dir = os.path.join(os.path.dirname(__file__), 'sounds')
        self.sounds_dir = sounds_dir
        self._sounds = {}
        self._enabled = True
        self._load()

    def _load(self):
        try:
            generate_sounds(self.sounds_dir)
            for name in _SOUND_DEFS:
                path = os.path.join(self.sounds_dir, f'{name}.wav')
                self._sounds[name] = pyglet.media.load(path, streaming=False)
        except Exception:
            # Audio backend unavailable (headless, no device) — disable silently
            self._enabled = False

    def play(self, name):
        """Play a sound by name.  Safe to call even if audio is unavailable."""
        if not self._enabled:
            return
        source = self._sounds.get(name)
        if source is None:
            return
        try:
            source.play()
        except Exception:
            pass

    @property
    def enabled(self):
        return self._enabled

    def available_sounds(self):
        return list(self._sounds.keys())
