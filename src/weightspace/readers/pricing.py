"""Quoted list-price estimates with explicit, nonrefundable reservations."""

from dataclasses import dataclass
from decimal import Decimal, InvalidOperation

from .data import fields, integer


_BUCKETS = ("input", "cache_read", "cache_write", "cache_write_long", "output")
_MILLION = Decimal(1_000_000)


def dollars(value, name):
    try:
        if type(value) not in (str, float, int):
            raise ValueError
        amount = Decimal(str(value))
        if not amount.is_finite() or not Decimal(0) <= amount <= 1_000_000:
            raise ValueError
    except (InvalidOperation, ValueError):
        raise ValueError(f"{name} must be a finite dollar amount from 0 to 1000000") from None
    return amount


@dataclass(frozen=True)
class RateCard:
    standard: dict
    threshold: int | None
    long_context: dict | None

    @classmethod
    def read(cls, record, context_window):
        def rates(table):
            fields(table, _BUCKETS)
            result = {bucket: dollars(table[bucket], f"{bucket} rate") for bucket in _BUCKETS}
            if result["input"] <= 0 or result["output"] <= 0:
                raise ValueError("input and output rates must be positive")
            return result

        fields(record, ("source", "as_of", "standard"), ("long_context", "long_context_above_input_tokens"))
        if not isinstance(record["source"], str) or not record["source"].startswith("https://"):
            raise ValueError("rate card must cite an HTTPS source")
        if not isinstance(record["as_of"], str) or not record["as_of"]:
            raise ValueError("rate card must give its as_of date")
        standard = rates(record["standard"])
        if "long_context" not in record and "long_context_above_input_tokens" not in record:
            return cls(standard, None, None)
        if "long_context" not in record or "long_context_above_input_tokens" not in record:
            raise ValueError("long context price and threshold must be configured together")
        threshold = integer(record["long_context_above_input_tokens"], 1, context_window, "long context threshold")
        long_context = rates(record["long_context"])
        if any(long_context[bucket] < standard[bucket] for bucket in _BUCKETS):
            raise ValueError("long context rates must be at least the corresponding standard rates")
        return cls(standard, threshold, long_context)

    def _rates(self, input_tokens):
        return self.long_context if self.threshold is not None and input_tokens > self.threshold else self.standard

    def reservation(self, input_tokens, output_tokens):
        selected = self._rates(input_tokens)
        input_rate = max(selected[bucket] for bucket in _BUCKETS if bucket != "output")
        return (input_tokens * input_rate + output_tokens * selected["output"]) / _MILLION

    def estimate(self, usage):
        selected = self._rates(usage["input_total"])
        return sum(usage[bucket] * selected[bucket] for bucket in _BUCKETS) / _MILLION
