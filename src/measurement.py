"""The result of timing one request: latency + tokens, and throughput derived from them.

Shared by every slice that measures a request: slice 1 times a single one
(src/timed_request.py); slice 2 fires many at once (src/concurrent_requests.py).
It lives in its own module so that neither slice "owns" the type the other depends on —
both import it from here as a peer.
"""

from dataclasses import dataclass


@dataclass(frozen=True)
class Measurement:
    """One request's result: how long it took and how many tokens it produced."""

    latency_s: float  # end-to-end wall-clock time for the whole request
    completion_tokens: int  # tokens the model generated (the output, not the prompt)

    @property
    def tokens_per_second(self) -> float:
        """Output generation throughput: tokens produced per second."""
        return self.completion_tokens / self.latency_s
