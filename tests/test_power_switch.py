# tests/test_power_switch.py
"""
Unit tests for bot_state.BotState — the /on, /off switch.

Persistence is covered for both backends: the local JSON file (real file via
tmp_path) and a fake Firestore client. No network anywhere.
"""

from bot_state import BotState


class _FakeDoc:
    """Firestore DocumentSnapshot stand-in."""

    def __init__(self, data=None):
        self._data = data

    @property
    def exists(self) -> bool:
        return self._data is not None

    def to_dict(self) -> dict:
        return self._data


class _FakeDocumentRef:
    def __init__(self, storage: dict, key: str):
        self._storage = storage
        self._key = key

    def get(self) -> _FakeDoc:
        return _FakeDoc(self._storage.get(self._key))

    def set(self, data: dict) -> None:
        self._storage[self._key] = dict(data)


class _FakeCollection:
    def __init__(self, storage: dict):
        self._storage = storage

    def document(self, name: str) -> _FakeDocumentRef:
        return _FakeDocumentRef(self._storage, name)


class _FakeFirestore:
    """Minimal Firestore client: collection() → document() → get/set."""

    def __init__(self):
        self._storage: dict = {}

    def collection(self, name: str) -> _FakeCollection:
        return _FakeCollection(self._storage)


def test_defaults_to_enabled_on_first_run(tmp_path):
    state = BotState(firebase_db=None, file_path=str(tmp_path / "s.json"))
    assert state.is_enabled() is True


def test_set_enabled_persists_to_file(tmp_path):
    path = str(tmp_path / "s.json")

    state = BotState(firebase_db=None, file_path=path)
    state.set_enabled(False)
    assert state.is_enabled() is False

    # A fresh instance (simulated restart) reads back "off".
    reloaded = BotState(firebase_db=None, file_path=path)
    assert reloaded.is_enabled() is False


def test_set_enabled_idempotent(tmp_path):
    path = str(tmp_path / "s.json")
    state = BotState(firebase_db=None, file_path=path)
    state.set_enabled(True)  # already True — no-op, must not break anything
    assert state.is_enabled() is True


def test_firestore_round_trip(tmp_path):
    db = _FakeFirestore()
    path = str(tmp_path / "s.json")

    state = BotState(firebase_db=db, file_path=path)
    state.set_enabled(False)

    # Simulate a restart: new BotState with the SAME Firestore but a MISSING
    # local file (Railway wipes the container filesystem).
    reloaded = BotState(firebase_db=db, file_path=str(tmp_path / "gone.json"))
    assert reloaded.is_enabled() is False


def test_local_file_fallback_when_firestore_has_no_doc(tmp_path):
    path = str(tmp_path / "s.json")

    BotState(firebase_db=None, file_path=path).set_enabled(False)

    # Firestore holds nothing yet → state falls back to the local file.
    state = BotState(firebase_db=_FakeFirestore(), file_path=path)
    assert state.is_enabled() is False


def test_toggle_back_on(tmp_path):
    path = str(tmp_path / "s.json")
    state = BotState(firebase_db=None, file_path=path)
    state.set_enabled(False)
    state.set_enabled(True)
    assert BotState(firebase_db=None, file_path=path).is_enabled() is True
