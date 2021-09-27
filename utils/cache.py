import functools
from diskcache import Cache

CACHES = []
def cache(name, directory='.cache'):
    """Keep a cache of previous function calls that persists on disk"""

    def decorator_cache(func):
        @functools.wraps(func)
        def wrapper_cache(*args, **kwargs):
            cache_key = args + tuple(kwargs.items())
            if cache_key not in wrapper_cache.cache:
                wrapper_cache.cache[cache_key] = func(*args, **kwargs)
            return wrapper_cache.cache[cache_key]

        wrapper_cache.cache = Cache(directory + '/' + name)
        CACHES.append(wrapper_cache.cache)

        def cache_clear():
            """Clear the cache"""
            wrapper_cache.cache.clear()

        def cache_close():
            """Close the diskcache"""
            wrapper_cache.cache.close()

        wrapper_cache.cache_clear = cache_clear
        wrapper_cache.cache_close = cache_close

        return wrapper_cache

    return decorator_cache
