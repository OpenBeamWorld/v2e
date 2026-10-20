# SPDX-FileCopyrightText: 2026 OpenBeam, LLC
# SPDX-License-Identifier: Apache-2.0
"""The harness: negotiate a plan or a refusal, execute it through a connector, audit the result.

``negotiate`` compares what a scene needs (``requires``, its actors and their control modes) with
what a connector declares in its manifest, and returns either a :class:`~openbeam_v2e.backends.base.Plan`
or a :class:`~openbeam_v2e.backends.base.Refusal` that lists every reason. A refusal is the honest
answer when a connector cannot run a scene: it lacks a required effect, channel kind or
observable; the scene has more controlled actors or seatable vehicles than the connector allows;
the requested timing regime is unavailable or its scene conditions are not met; or a controlled
actor has no program. A substitute mechanism is used only if the caller names it *and* the
connector declares it, and the run is then labelled ``emulated``.

``execute`` writes the plan into the run directory before anything is installed, attaches a
fresh call ledger, installs, opens the scene, spawns every actor at rest, runs, replays the
ledger audit and writes ``run.json``. ``audit_run`` re-audits a finished run directory offline.
``repeat_runs`` measures repeatability (the evidence behind a regime's ``repeatability`` value).
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
import statistics
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

from openbeam_v2e.backends.base import (
    PHYSICS_EFFECTS,
    RENDER_PACED,
    STEPPED,
    Backend,
    BackendManifest,
    ChannelKind,
    CompiledScene,
    ControllerHost,
    Effect,
    Layer,
    Plan,
    Refusal,
    RefusalReason,
    Repeatability,
    RunResult,
    RunSpec,
    Source,
    canonical_json,
    check_program,
    sha256_json,
    to_jsonable,
)
from openbeam_v2e.backends.ledger import CallLedger, audit_ledger, read_ledger

__all__ = ["SceneNeeds", "scene_needs", "negotiate", "effects_for", "execute", "audit_run", "repeat_runs",
           "load_run", "RunDirectoryError"]

PLAN_FILE = "plan.json"
RUN_FILE = "run.json"
INSTALL_FILE = "install.json"
LEDGER_DIR = "ledger"

#: Scene conditions the harness itself can check. Connector-specific conditions (such as a cloud
#: session kind) are folded into ``RegimeSupport.available`` by the connector.
KNOWN_CONDITIONS = ("player_vehicle",)

_HOST_PREFERENCE = (ControllerHost.CLIENT_LOCKSTEP, ControllerHost.ENGINE_SCRIPT, ControllerHost.VEHICLE_SCRIPT)


class RunDirectoryError(RuntimeError):
    pass


@dataclasses.dataclass(frozen=True)
class SceneNeeds:
    actors: tuple[str, ...]
    controlled: tuple[str, ...]
    seatable: tuple[str, ...]
    effects: tuple[str, ...]
    observables: tuple[str, ...]
    channels: tuple[tuple[str, str | None], ...]   # (channel id, kind or None)
    controlled_count: int
    seatable_count: int
    regime: str | None
    modes: Mapping[str, str]


def _mode(actor: Mapping[str, Any]) -> str:
    state = actor.get("initial_state") or {}
    return str(state["mode"]) if "mode" in state else "at_rest"


def scene_needs(scene: Mapping[str, Any]) -> SceneNeeds:
    """Read what a ``v2e.scene/1`` document needs from a connector (plain mapping, no engine terms)."""
    if not isinstance(scene, Mapping) or "actors" not in scene:
        raise ValueError("a scene needs an 'actors' list")
    dynamic = [a for a in scene["actors"] if a.get("class") != "static"]
    ids = tuple(str(a["id"]) for a in dynamic)
    controlled = tuple(str(a["id"]) for a in dynamic if a.get("control", "passive") != "passive")
    seatable = tuple(str(a["id"]) for a in dynamic if a.get("seatable"))
    req = scene.get("requires") or {}
    channels = []
    for item in req.get("channels", ()):
        if isinstance(item, str):
            channels.append((item, None))
        else:
            channels.append((str(item["id"]), item.get("kind")))
    modes = {str(a["id"]): _mode(a) for a in dynamic}
    return SceneNeeds(
        actors=ids,
        controlled=controlled,
        seatable=seatable,
        effects=tuple(req.get("effects", ())),
        observables=tuple(req.get("observables", ("position", "attitude"))),
        channels=tuple(channels),
        controlled_count=max(len(controlled), int(req.get("controlled_actors") or 0)),
        seatable_count=max(len(seatable), int(req.get("seatable_vehicles") or 0)),
        regime=req.get("regime"),
        modes=modes,
    )


def effects_for(manifest: BackendManifest, effects: Sequence[str],
                substitutes: Mapping[str, str] | None = None) -> tuple[list[Effect], list[RefusalReason]]:
    """Map required effect names to provenance records from a manifest, or refusal reasons."""
    caps = manifest.capabilities
    test_double = manifest.kind == "test_double"
    out: list[Effect] = []
    refusals: list[RefusalReason] = []
    substitutes = dict(substitutes or {})
    for name in effects:
        source = Source.TEST_DOUBLE if test_double else Source.NATIVE
        if name in PHYSICS_EFFECTS:
            if name == "deformation" and not caps.soft_body:
                refusals.append(RefusalReason("missing_effect", f"{manifest.backend} simulates rigid bodies only; "
                                                                "'deformation' needs a soft-body engine"))
                continue
            if name == "detachment" and caps.detachment == "none":
                refusals.append(RefusalReason("missing_effect", f"{manifest.backend} cannot detach parts"))
                continue
            out.append(Effect(name, source, f"{manifest.backend}.physics/{name}", Layer.ENGINE_CORE))
            continue
        channels = [c for c in caps.channels if c.effect == name]
        if channels:
            channel = channels[0]
            out.append(Effect(name, source, channel.mechanism or f"{manifest.backend}.channel/{channel.id}",
                              channel.layer))
            continue
        declared = caps.extras.get(f"substitute.{name}")
        if name in substitutes and declared and declared == substitutes[name]:
            out.append(Effect(name, Source.TEST_DOUBLE if test_double else Source.NATIVE_SUBSTITUTE, declared,
                              Layer.ENGINE_CORE))
            continue
        if name in substitutes:
            refusals.append(RefusalReason("substitute_not_declared",
                                          f"{manifest.backend} declares no substitute {substitutes[name]!r} "
                                          f"for effect {name!r}"))
            continue
        refusals.append(RefusalReason("missing_effect", f"{manifest.backend} has no channel or mechanism for "
                                                        f"effect {name!r}"))
    return out, refusals


def negotiate(scene: Mapping[str, Any], manifest: BackendManifest, *, opt_in: Mapping[str, Any] | None = None,
              programs: Mapping[str, Mapping[str, Any]] | None = None, lockstep: bool = False) -> Plan | Refusal:
    """Return a :class:`Plan` for running ``scene`` on the connector described by ``manifest``, or a refusal.

    ``opt_in`` keys: ``regime`` (a regime name), ``host`` (a controller host), ``substitutes``
    (``{effect: mechanism}``), ``add_anchor`` (true lets a compiler add an off-camera seatable
    anchor vehicle, which counts toward the seatable-vehicle limit), ``conditions`` (scene
    conditions the caller vouches for). ``lockstep`` says whether the connector object implements
    the client-lockstep mixin.
    """
    opt_in = dict(opt_in or {})
    programs = dict(programs or {})
    needs = scene_needs(scene)
    caps = manifest.capabilities
    reasons: list[RefusalReason] = []
    notes: list[str] = []

    effects, effect_refusals = effects_for(manifest, needs.effects, opt_in.get("substitutes"))
    reasons += effect_refusals

    for channel_id, kind in needs.channels:
        channel = caps.channel(channel_id)
        if channel is None:
            reasons.append(RefusalReason("missing_channel", f"{manifest.backend} has no channel {channel_id!r}"))
        elif kind is not None and channel.kind is not ChannelKind(kind):
            reasons.append(RefusalReason("channel_kind", f"{channel_id} is {channel.kind.value} on "
                                                         f"{manifest.backend}, the scene needs {kind}"))

    missing_obs = sorted(set(needs.observables) - set(caps.observables))
    if missing_obs:
        reasons.append(RefusalReason("missing_observable", f"{manifest.backend} cannot observe {missing_obs}"))

    if needs.controlled_count > caps.max_controlled_actors:
        reasons.append(RefusalReason("actor_limit", f"{needs.controlled_count} controlled actor(s); "
                                                    f"{manifest.backend} allows {caps.max_controlled_actors}"))

    seatable_count = needs.seatable_count
    regime_name = opt_in.get("regime") or needs.regime
    if regime_name is None:
        available = [r.regime for r in caps.regimes if r.available]
        regime_name = STEPPED if STEPPED in available else (RENDER_PACED if RENDER_PACED in available
                                                            else (available[0] if available else None))
    support = caps.regime(regime_name) if regime_name else None
    if support is None or not support.available:
        reasons.append(RefusalReason("regime_unavailable", f"regime {regime_name!r} is not available on "
                                                           f"{manifest.backend}"))
    else:
        vouched = set(opt_in.get("conditions", ()))
        for condition in support.requires:
            if condition == "player_vehicle":
                if needs.seatable:
                    continue
                if opt_in.get("add_anchor"):
                    seatable_count += 1
                    notes.append("an off-camera anchor vehicle is added for the regime's player vehicle; it counts "
                                 "toward the seatable-vehicle limit")
                    continue
                reasons.append(RefusalReason("regime_condition",
                                             f"regime {regime_name!r} needs a player vehicle; add a seatable vehicle "
                                             f"or opt in to an off-camera anchor (add_anchor)"))
            elif condition not in vouched:
                reasons.append(RefusalReason("regime_condition", f"regime {regime_name!r} needs {condition!r}"))
        if support.repeatability is Repeatability.UNMEASURED:
            notes.append(f"repeatability of {regime_name!r} on {manifest.backend} is unmeasured: compare runs with "
                         f"N >= 9 repeats and robust intervals")
        elif support.repeatability is Repeatability.STATISTICAL:
            notes.append(f"{regime_name!r} is statistical on {manifest.backend}: use N >= 9 repeats per arm")

    if caps.max_seatable_vehicles is not None and seatable_count > caps.max_seatable_vehicles:
        reasons.append(RefusalReason("seatable_limit", f"{seatable_count} seatable vehicle(s); {manifest.backend} "
                                                       f"allows {caps.max_seatable_vehicles}"))

    host: ControllerHost | None = None
    if needs.controlled:
        wanted = opt_in.get("host")
        hosts = set(caps.controller_hosts)
        if not lockstep:
            hosts.discard(ControllerHost.CLIENT_LOCKSTEP)
        if wanted is not None:
            host = ControllerHost(wanted)
            if host not in hosts:
                reasons.append(RefusalReason("host_unavailable", f"controller host {host.value!r} is not available "
                                                                 f"on {manifest.backend}"))
        else:
            host = next((h for h in _HOST_PREFERENCE if h in hosts), None)
            if host is None:
                reasons.append(RefusalReason("no_controller_host",
                                             f"{manifest.backend} offers no controller host in this configuration, "
                                             f"so it cannot run controlled actors"))

    for actor in needs.controlled:
        program = programs.get(actor)
        if program is None:
            reasons.append(RefusalReason("missing_program", f"controlled actor {actor!r} has no controller program"))
            continue
        try:
            check_program(program)
        except ValueError as exc:
            reasons.append(RefusalReason("bad_program", f"{actor}: {exc}"))
            continue
        if program["actor"] != actor:
            reasons.append(RefusalReason("bad_program", f"program for {actor!r} names actor {program['actor']!r}"))
        if host is not None and program.get("host") not in (None, host.value):
            reasons.append(RefusalReason("bad_program", f"program for {actor!r} targets host {program.get('host')!r}; "
                                                        f"the plan uses {host.value!r}"))
    for actor in sorted(set(programs) - set(needs.controlled)):
        reasons.append(RefusalReason("program_for_passive_actor",
                                     f"{actor!r} is passive or unknown in the scene but has a program"))

    for actor, mode in needs.modes.items():
        if mode not in ("at_rest", "drive_in"):
            reasons.append(RefusalReason("initial_state", f"{actor!r}: initial state {mode!r} cannot be spawned "
                                                          f"at rest; start the scene earlier or declare a native "
                                                          f"launcher"))
        if mode == "drive_in" and actor not in needs.controlled:
            reasons.append(RefusalReason("initial_state", f"{actor!r} drives in but is passive"))

    if not caps.spawn_at_pose:
        notes.append(f"{manifest.backend} spawns at a position only in this configuration; the rotation residual is "
                     f"recorded and a run whose residual exceeds the tolerance is invalid")

    if reasons:
        return Refusal(manifest.backend, tuple(reasons))
    channel_ids = tuple(c.id for c in caps.channels)
    return Plan(
        backend=manifest.backend,
        regime=str(regime_name),
        host=host,
        actors=needs.actors,
        controlled=needs.controlled,
        seatable=needs.seatable,
        programs={a: dict(programs[a]) for a in needs.controlled},
        channels={a: channel_ids for a in needs.controlled},
        effects=tuple(effects),
        requires=dict(scene.get("requires") or {}),
        opt_in=opt_in,
        modes=dict(needs.modes),
        notes=tuple(notes),
    )


def _write_json(path: Path, data: Any) -> None:
    path.write_text(json.dumps(to_jsonable(data), indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _prepare_run_dir(run_dir: Path) -> Path:
    run_dir = Path(run_dir)
    if run_dir.exists() and any(run_dir.iterdir()):
        raise RunDirectoryError(f"run directory {run_dir} is not empty; every run gets a fresh directory")
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir


def execute(backend: Backend, compiled: CompiledScene, plan: Plan, spec: RunSpec, run_dir: Path,
            *, header: Mapping[str, Any] | None = None) -> RunResult:
    """Run ``plan`` through ``backend`` and return the audited result (also written to ``run.json``)."""
    if plan.backend != backend.name:
        raise ValueError(f"plan was negotiated for {plan.backend!r}, not {backend.name!r}")
    if spec.regime != plan.regime or spec.host != plan.host:
        raise ValueError("run spec regime and host must match the plan")
    if spec.t0_s is None and compiled.t0_s is not None:
        spec = dataclasses.replace(spec, t0_s=compiled.t0_s)
    run_dir = _prepare_run_dir(run_dir)
    _write_json(run_dir / PLAN_FILE, plan.to_json())          # the plan is on disk before any engine call
    manifest = backend.manifest()
    ledger = CallLedger(run_dir / LEDGER_DIR, header={
        "connector": backend.name,
        "manifest_sha256": sha256_json(manifest),
        "plan_sha256": sha256_json(plan.to_json()),
        **dict(header or {}),
    })
    backend.use_ledger(ledger)
    try:
        receipts = backend.install(compiled.packages)
        _write_json(run_dir / INSTALL_FILE, [to_jsonable(r) for r in receipts])
        backend.open_scene(compiled.level)
        refs = []
        for spawn in compiled.spawns:
            ref, _record = backend.spawn(compiled.object(spawn.object), spawn.pose, spawn.actor)
            refs.append(ref)
        spawned = {r.actor for r in refs}
        missing = sorted(set(plan.actors) - spawned)
        if missing:
            raise ValueError(f"compiled scene spawns no actor for {missing}")
        result = backend.run(refs, plan.programs, spec, run_dir)
    finally:
        ledger.close()
        backend.use_ledger(None)
    policy = backend.run_audit_policy()
    audit = audit_ledger(run_dir / LEDGER_DIR, policy)
    attestation = dataclasses.replace(result.attestation, audit=audit, ledger=run_dir / LEDGER_DIR,
                                      ledger_head_sha256=ledger.head_sha256)
    content = result.content or compiled.content
    t0 = spec.t0_s if spec.t0_s is not None else result.t0_s
    result = dataclasses.replace(result, attestation=attestation, content=tuple(content), t0_s=t0)
    _write_json(run_dir / RUN_FILE, result.to_json(relative_to=run_dir))
    return result


def load_run(run_dir: Path) -> RunResult:
    run_dir = Path(run_dir)
    path = run_dir / RUN_FILE
    if not path.is_file():
        raise RunDirectoryError(f"{run_dir} has no {RUN_FILE}")
    return RunResult.from_json(json.loads(path.read_text(encoding="utf-8")), base=run_dir)


def audit_run(run_dir: Path, policy_for: Callable[[str], Any] | None = None) -> dict[str, Any]:
    """Re-audit a run directory offline: ledger chain, allow list, blobs and the recorded rows digest."""
    from openbeam_v2e.backends import registry

    run_dir = Path(run_dir)
    result = load_run(run_dir)
    name = result.manifest.backend
    policy = (policy_for or registry.audit_policy_for)(name)
    audit = audit_ledger(result.attestation.ledger, policy)
    head = read_ledger(result.attestation.ledger)[-1]["sha256"]
    problems = []
    if audit != "clean":
        problems.append(audit)
    if head != result.attestation.ledger_head_sha256:
        problems.append("ledger head does not match the run record")
    recorded = result.attestation.audit
    if recorded != "clean":
        problems.append(f"the run recorded its own audit as: {recorded}")
    return {
        "run": str(run_dir),
        "connector": name,
        "audit": audit,
        "ledger_head_sha256": head,
        "matches_run_record": head == result.attestation.ledger_head_sha256,
        "valid": result.valid() and not problems,
        "claim_class": result.claim_class(),
        "problems": problems,
    }


def _rows_sha256(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _final_positions(path: Path) -> dict[str, tuple[float, float, float]]:
    last = None
    with Path(path).open("r", encoding="utf-8") as fh:
        for line in fh:
            if line.strip():
                last = json.loads(line)
    if last is None:
        return {}
    return {actor: tuple(state["p"]) for actor, state in last["actors"].items()}


def _robust(values: Sequence[float]) -> dict[str, float]:
    med = statistics.median(values)
    mad = statistics.median([abs(v - med) for v in values])
    spread = 3.0 * 1.4826 * mad
    return {"median": med, "mad": mad, "lo": med - spread, "hi": med + spread}


def repeat_runs(make_backend: Callable[[], Backend], compiled: CompiledScene, plan: Plan, spec: RunSpec,
                out_dir: Path, n: int = 9) -> dict[str, Any]:
    """Run the same plan ``n`` times in fresh run directories and report measured repeatability.

    ``bit_identical`` means every run produced byte-identical rows; otherwise the report is
    ``statistical`` with the median and the robust interval (median +- 3 * 1.4826 * MAD) of every
    actor's final position and of every event time.
    """
    if n < 2:
        raise ValueError("repeatability needs at least two runs")
    out_dir = _prepare_run_dir(out_dir)
    runs = []
    for index in range(n):
        backend = make_backend()
        try:
            result = execute(backend, compiled, plan, spec, out_dir / f"run_{index:03d}")
        finally:
            backend.close()
        runs.append(result)
    digests = [_rows_sha256(r.rows) for r in runs]
    identical = len(set(digests)) == 1
    finals = [_final_positions(r.rows) for r in runs]
    metrics: dict[str, Any] = {}
    for actor in sorted(finals[0]):
        for axis, label in enumerate("xyz"):
            metrics[f"{actor}.final_{label}_m"] = _robust([f[actor][axis] for f in finals if actor in f])
    for event in sorted(set().union(*(r.events.keys() for r in runs))):
        times = [r.events[event] for r in runs if event in r.events]
        metrics[f"event.{event}_s"] = {**_robust(times), "present_in": len(times)}
    manifest = runs[0].manifest
    report = {
        "schema": "v2e.bench.repeat/1",
        "connector": manifest.backend,
        "engine": manifest.engine,
        "regime": plan.regime,
        "n": n,
        "repeatability": (Repeatability.BIT_IDENTICAL if identical else Repeatability.STATISTICAL).value,
        "rows_sha256": digests,
        "valid_runs": sum(1 for r in runs if r.valid()),
        "claim_classes": sorted({r.claim_class() for r in runs}),
        "metrics": metrics,
        "plan_sha256": sha256_json(plan.to_json()),
    }
    report["report_sha256"] = hashlib.sha256(canonical_json(report)).hexdigest()
    _write_json(out_dir / "bench-repeat.json", report)
    return report
