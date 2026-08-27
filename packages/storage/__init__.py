"""Persistence layer — MongoDB, and nothing clever.

Storage holds what the deterministic engines produced: users, their scan
history, their medicine cabinet. It never holds a fact of its own — a record
read back from Mongo carries the same provenance it was written with.
"""

from .mongo import MongoStore

__all__ = ["MongoStore"]
