"""
Tests for ``dgssp.runtime.describe_account``.

Run against a stub service, because the point of the helper is that it survives
whatever the live API does or does not expose: a plan without ``usage()``, a
backend whose ``status()`` raises, an account payload containing a token. None
of those may crash a run or leak a secret into a results file.
"""

from __future__ import annotations

from dgssp.runtime import describe_account, print_account_summary


class _Status:
    def __init__(self, operational=True, pending_jobs=3):
        self.operational = operational
        self.pending_jobs = pending_jobs


class _Properties:
    last_update_date = "2026-08-30T12:00:00+00:00"


class _Backend:
    def __init__(self, name, num_qubits, family=None, raise_status=False):
        self.name = name
        self.num_qubits = num_qubits
        self.processor_type = {"family": family, "revision": 3} if family else None
        self._raise_status = raise_status

    def status(self):
        if self._raise_status:
            raise RuntimeError("provider is down")
        return _Status()

    def properties(self):
        return _Properties()


class _Service:
    channel = "ibm_quantum_platform"
    active_instance = "crn:v1:bluemix:public:quantum-computing:us-east:a/xyz::"

    def __init__(self, *, usage_raises=False):
        self._usage_raises = usage_raises

    def active_account(self):
        return {
            "channel": "ibm_quantum_platform",
            "token": "SECRET-TOKEN-DO-NOT-LEAK",
            "instance": self.active_instance,
        }

    def instances(self):
        return [{"crn": self.active_instance, "plan": "open", "name": "default"}]

    def usage(self):
        if self._usage_raises:
            raise RuntimeError("usage is not available on this plan")
        return {"period": {"remainingSeconds": 480}}

    def backends(self):
        return [
            _Backend("ibm_torino", 133, family="Heron"),
            _Backend("ibm_broken", 127, family="Eagle", raise_status=True),
        ]


def test_describe_account_reports_plan_instances_and_backends():
    """The fields that decide the hardware plan are all present."""
    info = describe_account(_Service())

    assert info["channel"] == "ibm_quantum_platform"
    assert info["instances"][0]["plan"] == "open"
    assert info["usage"] == {"period": {"remainingSeconds": 480}}

    names = [row["name"] for row in info["backends"]]
    assert names == ["ibm_torino", "ibm_broken"]

    torino = info["backends"][0]
    assert torino["num_qubits"] == 133
    assert torino["processor_type"]["family"] == "Heron"
    assert torino["simulator"] is False
    assert torino["operational"] is True
    assert torino["pending_jobs"] == 3
    assert torino["calibration_date"] == "2026-08-30T12:00:00+00:00"


def test_token_never_appears_in_the_summary():
    """A summary may be saved to paper/hardware/, so it must carry no secret."""
    info = describe_account(_Service())

    assert "token" not in info["active_account"]
    assert "SECRET-TOKEN-DO-NOT-LEAK" not in repr(info)


def test_unavailable_api_is_reported_not_raised():
    """A plan that does not expose usage yields an error string, not a crash."""
    info = describe_account(_Service(usage_raises=True))
    assert "error" in info["usage"]
    assert "not available on this plan" in info["usage"]["error"]


def test_a_backend_whose_status_fails_is_still_listed():
    """One unreachable device must not hide the rest of the fleet."""
    info = describe_account(_Service())
    broken = info["backends"][1]
    assert broken["name"] == "ibm_broken"
    assert broken["operational"] is None
    assert broken["num_qubits"] == 127


def test_print_account_summary_returns_the_same_payload(capsys):
    """The printing wrapper is a view, not a second source of truth."""
    info = print_account_summary(_Service())
    out = capsys.readouterr().out

    assert info == describe_account(_Service())
    assert "ibm_torino" in out
    assert "Heron" in out
    assert "SECRET-TOKEN-DO-NOT-LEAK" not in out
