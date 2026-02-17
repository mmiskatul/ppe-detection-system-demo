from functools import lru_cache
from typing import Optional

from pymongo import MongoClient

from .config import settings


@lru_cache
def get_mongo_client() -> Optional[MongoClient]:
    if not settings.mongodb_uri:
        return None
    return MongoClient(settings.mongodb_uri)


def get_collection():
    client = get_mongo_client()
    if client is None:
        return None
    return client[settings.mongodb_db][settings.mongodb_collection]
