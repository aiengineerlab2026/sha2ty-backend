"""In-process event bus. Handlers run in order inside the request; an error propagates to the caller.

ponytail: in-process and synchronous; move to a queue (Supabase Realtime, Pub/Sub) when handlers must
outlive the request or run on another service.
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Awaitable, Callable
from typing import Any

Handler = Callable[[Any, Any], Awaitable[None]]  # (event, app.state)


class EventBus:
    def __init__(self) -> None:
        self._handlers: defaultdict[type, list[Handler]] = defaultdict(list)

    def subscribe(self, *event_types: type) -> Callable[[Handler], Handler]:
        def register(handler: Handler) -> Handler:
            for event_type in event_types:
                self._handlers[event_type].append(handler)
            return handler

        return register

    async def publish(self, event: object, state: Any) -> None:
        for handler in self._handlers[type(event)]:
            await handler(event, state)


bus = EventBus()
