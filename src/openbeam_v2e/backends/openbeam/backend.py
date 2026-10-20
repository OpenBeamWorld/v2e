# SPDX-FileCopyrightText: 2026 OpenBeam, LLC
# SPDX-License-Identifier: Apache-2.0
"""The OpenBeam connector: documented customer MCP tools only, authored content only.

What it does through the published tools:

* **install**: ``install_mod`` (vehicles and props) and ``upload_level`` (levels) from a URL you
  control plus its sha256, or from a path on the game's machine; idempotent by sha256 using
  ``list_mods``. The documented tools cannot upload local bytes, so a package without a URL or
  game path is refused.
* **open scene**: ``load_level`` with an explicit wait.
* **spawn**: ``spawn_vehicle`` at a position (the documented tool takes no rotation). The request
  is pre-compensated from the object's bounding box; the actual pose is read back with
  ``actor_poses`` and the residual is recorded. Without a verified yaw convention the orientation
  residual is recorded as unverified (180 degrees), which makes the run invalid for scoring.
  OpenBeam keeps at most two seatable vehicles; the connector refuses a third and checks after
  every spawn that no actor disappeared.
* **run**: passive scenes in the documented stepped mode (``sim_mode``, ``sim_step``), with rows
  from ``actor_poses`` (position and velocity; the engine reports no full attitude through the
  documented tools) and optional frames from ``camera_frame``. Controlled actors need an
  engine-side host (:class:`~openbeam_v2e.backends.openbeam.scripts.ScriptHost`); none is bundled.

Coordinates: OpenBeam reports positions with Y up. V2E's level frame is Z up. The connector maps
``(x, y, z)`` in V2E to ``(x, z, -y)`` in OpenBeam and back with ``(x, y, z) -> (x, -z, y)``.
"""

from __future__ import annotations

import hashlib
import json
import math
import time
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

from openbeam_v2e.backends.base import (
    NO_RECORDING,
    RENDER_PACED,
    STEPPED,
    STEPPED_CAPTURE,
    STREAM_CAPTURE,
    ActorRef,
    Attestation,
    Backend,
    BackendError,
    BackendManifest,
    Capabilities,
    ControllerHost,
    Effect,
    InstallReceipt,
    LevelRef,
    ObjectRef,
    Package,
    Plan,
    Pose,
    RefusedCall,
    RegimeSupport,
    Repeatability,
    RunResult,
    RunSpec,
    Source,
    SpawnRecord,
    StateRow,
    canonical_json,
    sha256_json,
)
from openbeam_v2e.backends.ledger import AuditPolicy
from openbeam_v2e.backends.lockstep import EventDetector, RowWriter
from openbeam_v2e.backends.openbeam.scripts import ScriptHost, ScriptPolicy, ScriptReference
from openbeam_v2e.backends.openbeam.tools import GAME_TOOLS, PUBLIC_TOOLS, TOOL_ARGS, GameClient, GuardedClient

__all__ = ["OpenBeamBackend", "zup_to_yup", "yup_to_zup", "PHYSICS_DT_S", "MAX_SEATABLE_VEHICLES"]

PHYSICS_DT_S = 0.0005            # documented: sim_step advances fixed 0.5 ms steps
MAX_SIM_STEPS_PER_CALL = 20000   # documented upper bound for sim_step
MAX_SEATABLE_VEHICLES = 2
UNVERIFIED_ORIENTATION_DEG = 180.0
LICENCE_NOTICE = "OpenBeam Cloud sessions are subject to OpenBeam's terms of service."


def zup_to_yup(p: Sequence[float]) -> list[float]:
    x, y, z = (float(c) for c in p)
    return [x, z, -y]


def yup_to_zup(p: Sequence[float]) -> list[float]:
    x, y, z = (float(c) for c in p)
    return [x, -z, y]


def _program_blob(program: Mapping[str, Any]) -> bytes:
    return canonical_json(program)


def _run_marker(plan: Plan | None, seed: int) -> str:
    digest = sha256_json(plan.to_json() if plan is not None else {})[:12]
    return f"v2e-{digest}-{int(seed)}"


def _audit_run_script(args: Mapping[str, Any], row: Mapping[str, Any], scripts: ScriptPolicy) -> str | None:
    code = args.get("code")
    if not isinstance(code, str) or scripts.match(code) is None:
        return f"call {row.get('seq')}: run_script body is not a rendering of an audited snippet"
    if hashlib.sha256(code.encode("utf-8")).hexdigest() not in (row.get("blobs") or []):
        return f"call {row.get('seq')}: run_script body was not stored as a blob"
    return None


def _audit_game_call(args: Mapping[str, Any], row: Mapping[str, Any], scripts: ScriptPolicy) -> str | None:
    inner = args.get("name")
    if inner not in GAME_TOOLS:
        return f"call {row.get('seq')}: cloud_game_call forwards {inner!r}, which is not a documented game tool"
    inner_args = dict(args.get("arguments") or {})
    extra = sorted(set(inner_args) - set(TOOL_ARGS[inner]))
    if extra:
        return f"call {row.get('seq')}: forwarded {inner} with undocumented argument(s) {extra}"
    if inner == "run_script":
        return _audit_run_script(inner_args, row, scripts)
    return None


def _audit_program(args: Mapping[str, Any], row: Mapping[str, Any]) -> str | None:
    if args.get("sha256") not in (row.get("blobs") or []):
        return f"call {row.get('seq')}: program {args.get('actor')!r} was not stored as a blob"
    return None


def make_audit_policy(scripts: ScriptPolicy) -> AuditPolicy:
    methods = {f"mcp/{tool}": TOOL_ARGS[tool] for tool in PUBLIC_TOOLS}
    methods["v2e.program"] = frozenset({"actor", "model", "sha256"})
    return AuditPolicy(
        connector="openbeam",
        methods=methods,
        checks={
            "mcp/run_script": lambda a, r: _audit_run_script(a, r, scripts),
            "mcp/cloud_game_call": lambda a, r: _audit_game_call(a, r, scripts),
            "v2e.program": _audit_program,
        },
    )


class OpenBeamBackend(Backend):
    """See the module docstring. ``client`` is any object with ``call(tool, arguments) -> ToolResult``."""

    name = "openbeam"

    def __init__(self, client: GameClient, *, engine: str = "OpenBeam", stepped_available: bool = True,
                 host: ScriptHost | None = None, accepted_script_reference: ScriptReference | None = None,
                 yaw_convention: Mapping[str, Any] | None = None, end_session: Callable[[], Any] | None = None,
                 load_wait_s: float = 180.0, spawn_wait_s: float = 60.0, sleep: Callable[[float], None] = time.sleep,
                 transport: str = "direct") -> None:
        self.scripts = ScriptPolicy()
        if host is not None:
            if accepted_script_reference is None or accepted_script_reference != host.reference:
                raise ValueError("an engine-script host is enabled only when its script reference is accepted "
                                 "explicitly (accepted_script_reference must equal host.reference)")
            self.scripts.add(host.snippets)
        self.client = client
        self.guard = GuardedClient(client, scripts=self.scripts)
        self.engine = engine
        self.stepped_available = stepped_available
        self.host = host
        self.yaw_convention = dict(yaw_convention) if yaw_convention else None
        self._end_session = end_session
        self.load_wait_s = load_wait_s
        self.spawn_wait_s = spawn_wait_s
        self.sleep = sleep
        self.transport = transport
        self._installed: dict[str, InstallReceipt] = {}
        self._level: LevelRef | None = None
        self._spawned: list[tuple[ActorRef, ObjectRef]] = []
        self._spawns: list[SpawnRecord] = []
        self._known_ids: set[str] = set()
        self._removed = 0
        self._entered = False
        self.ledger = None

    # ------------------------------------------------------------------ construction

    @classmethod
    def from_config(cls, config: Mapping[str, Any]) -> "OpenBeamBackend":
        """Create the connector from configuration (never from scene or agent input).

        Keys: ``cassette`` (replay a recorded cassette; offline), or ``session_id`` (a cloud
        session reached at its session endpoint), or ``account_session_id`` (a cloud session
        reached through the account endpoint's ``cloud_game_call``), or ``endpoint`` (the MCP URL
        of a game you run); plus ``base_url``, ``token_file``, ``training`` (the session is a
        training session, so stepped mode is available), ``engine`` and ``yaw_convention``.
        """
        from openbeam_v2e.backends.openbeam import transport as tr

        config = dict(config)
        options = {k: config[k] for k in ("engine", "yaw_convention") if k in config}
        if "cassette" in config:
            from openbeam_v2e.backends.openbeam.cassette import ReplayClient
            return cls(ReplayClient(Path(config["cassette"])), transport="cassette",
                       stepped_available=bool(config.get("training", True)), **options)
        token = tr.TokenStore(Path(config["token_file"]) if config.get("token_file") else None)
        base_url = config.get("base_url", tr.DEFAULT_BASE_URL)
        if "session_id" in config:
            client = tr.GameEndpoint(tr.session_endpoint(str(config["session_id"]), base_url), token)
            return cls(client, transport="session", stepped_available=bool(config.get("training", False)), **options)
        if "account_session_id" in config:
            client = tr.AccountGame(tr.CloudAccount(token, base_url=base_url), str(config["account_session_id"]))
            return cls(client, transport="account", stepped_available=bool(config.get("training", False)), **options)
        if "endpoint" in config:
            client = tr.GameEndpoint(str(config["endpoint"]), token)
            return cls(client, transport="local", stepped_available=True, **options)
        raise ValueError("OpenBeam connector configuration needs one of: cassette, session_id, account_session_id, "
                         "endpoint")

    def use_ledger(self, ledger: Any) -> None:
        self.ledger = ledger
        self.guard.ledger = ledger

    # ------------------------------------------------------------------ manifest and provenance

    def manifest(self) -> BackendManifest:
        host = self.host
        channels = tuple(host.channels) if host else ()
        observables = {"position", "velocity"} | (set(host.observables) if host else set())
        regimes = (
            RegimeSupport(RENDER_PACED, host is not None, (), Repeatability.UNMEASURED, None),
            RegimeSupport(STEPPED, self.stepped_available, ("player_vehicle",), Repeatability.UNMEASURED, None),
        )
        caps = Capabilities(
            channels=channels,
            soft_body=True,
            detachment="breakgroups",
            regimes=regimes,
            physics_dt_s=PHYSICS_DT_S,
            recordings=(STEPPED_CAPTURE, STREAM_CAPTURE, NO_RECORDING),
            spawn_at_pose=False,
            max_controlled_actors=MAX_SEATABLE_VEHICLES if host else 0,
            max_seatable_vehicles=MAX_SEATABLE_VEHICLES,
            controller_hosts=frozenset({ControllerHost.ENGINE_SCRIPT}) if host else frozenset(),
            observables=frozenset(observables),
            camera_runtime_fov=False,
            instance_mask=False,
        )
        return BackendManifest(backend="openbeam", kind="engine", engine=self.engine,
                               api="mcp (documented tools)", dialect="stock_jbeam/openbeam", capabilities=caps,
                               licence_notice=LICENCE_NOTICE)

    def mechanisms(self, plan: Plan) -> tuple[Effect, ...]:
        return tuple(Effect(e.name, Source.NATIVE, e.mechanism, e.layer) for e in plan.effects)

    @classmethod
    def audit_policy(cls) -> AuditPolicy:
        return make_audit_policy(ScriptPolicy())

    def run_audit_policy(self) -> AuditPolicy:
        return make_audit_policy(self.scripts)

    # ------------------------------------------------------------------ install and scene

    def install(self, packages: Sequence[Package]) -> list[InstallReceipt]:
        present = {}
        listing = self.guard.call("list_mods", {}).data
        for record in listing.get("mods", []) or []:
            if record.get("sha256"):
                present[str(record["sha256"]).lower()] = record
        receipts = []
        for pkg in packages:
            if not pkg.sha256:
                raise RefusedCall(f"package {pkg.name!r} has no sha256")
            sha = pkg.sha256.lower()
            if sha in present:
                record = present[sha]
                ids = tuple(record.get("vehicles") or record.get("vehicleIds") or record.get("levelIds") or ())
                receipt = InstallReceipt(pkg.name, sha, "already_present", tuple(str(i) for i in ids),
                                         {"filename": record.get("filename")})
            else:
                args: dict[str, Any] = {"sha256": sha}
                if pkg.url:
                    args["url"] = pkg.url
                elif pkg.game_path:
                    args["path"] = pkg.game_path
                else:
                    raise RefusedCall(f"package {pkg.name!r}: OpenBeam installs archives from a URL you control (with "
                                      f"its sha256) or from a path on the game's machine; the documented tools "
                                      f"cannot upload local files")
                if pkg.filename:
                    args["filename"] = pkg.filename
                tool = "upload_level" if pkg.kind == "level" else "install_mod"
                data = self.guard.call(tool, args).data
                refused = (data.get("cache") or {}).get("refused") or data.get("refused")
                if refused:
                    raise BackendError(f"OpenBeam refused entries of {pkg.name!r}: {refused}")
                if pkg.kind == "level":
                    record = data.get("level") or {}
                    ids = record.get("levelIds") or (data.get("cache") or {}).get("entries") or ()
                else:
                    ids = data.get("vehicles") or (data.get("cache") or {}).get("vehicles") or ()
                status = "already_present" if data.get("already_present") else "installed"
                receipt = InstallReceipt(pkg.name, sha, status, tuple(str(i) for i in ids),
                                         {"filename": (data.get("level") or data.get("mod") or {}).get("filename")})
            self._installed[pkg.name] = receipt
            receipts.append(receipt)
        return receipts

    def open_scene(self, level: LevelRef) -> None:
        args: dict[str, Any] = {"wait_seconds": self.load_wait_s}
        if level.terrain:
            args["terrain"] = level.terrain
        else:
            level_id = level.level_id
            if level_id is None and level.package in self._installed:
                ids = self._installed[level.package].engine_ids
                level_id = ids[0] if len(ids) == 1 else None
            if not level_id:
                raise BackendError("open_scene needs a level id or a terrain")
            args["level_id"] = level_id
            if level.filename:
                args["filename"] = level.filename
        if level.spawn:
            args["spawn"] = level.spawn
        data = self.guard.call("load_level", args).data
        if not data.get("loaded"):
            raise BackendError(f"level did not finish loading within {self.load_wait_s} s (pending="
                               f"{data.get('pending')}); check status and read_log")
        self._level = level
        self._spawned, self._spawns, self._known_ids = [], [], set()
        self._removed, self._entered = 0, False
        self._known_ids = {str(a.get("instance_id")) for a in self._poses()}

    def _poses(self) -> list[dict]:
        return list(self.guard.call("actor_poses", {}).data.get("actors", []) or [])

    def _engine_file(self, obj: ObjectRef) -> str:
        if obj.engine_file:
            return obj.engine_file
        receipt = self._installed.get(obj.package or "")
        if receipt is None:
            raise BackendError(f"object {obj.name!r}: its package {obj.package!r} is not installed")
        if len(receipt.engine_ids) == 1:
            return receipt.engine_ids[0]
        for candidate in receipt.engine_ids:
            if candidate == obj.name or candidate.startswith(obj.name):
                return candidate
        raise BackendError(f"object {obj.name!r}: cannot tell which of {list(receipt.engine_ids)} to spawn")

    def _yaw_deg(self, reported: Any) -> float | None:
        if self.yaw_convention is None or reported is None:
            return None
        value = float(reported)
        if self.yaw_convention.get("units", "deg") == "rad":
            value = math.degrees(value)
        return float(self.yaw_convention.get("sign", 1.0)) * value + float(self.yaw_convention.get("offset_deg", 0.0))

    def spawn(self, obj: ObjectRef, pose: Pose, actor: str) -> tuple[ActorRef, SpawnRecord]:
        if self._level is None:
            raise BackendError("open a scene first")
        seatable_now = sum(1 for ref, _ in self._spawned if ref.seatable)
        if obj.seatable and seatable_now >= MAX_SEATABLE_VEHICLES:
            raise RefusedCall(f"OpenBeam keeps at most {MAX_SEATABLE_VEHICLES} seatable vehicles; spawning another "
                              f"would remove the oldest")
        offset = [0.0, 0.0, 0.0]
        if obj.bbox_min_m and obj.bbox_max_m:
            lo, hi = obj.bbox_min_m, obj.bbox_max_m
            offset = [(lo[0] + hi[0]) / 2.0, (lo[1] + hi[1]) / 2.0, lo[2]]
        requested_engine = [pose.position_m[i] + offset[i] for i in range(3)]
        args: dict[str, Any] = {"file": self._engine_file(obj), "position": zup_to_yup(requested_engine),
                                "enter": bool(obj.seatable and not self._entered), "wait_seconds": self.spawn_wait_s}
        if obj.config:
            args["config"] = obj.config
        data = self.guard.call("spawn_vehicle", args).data
        if not data.get("spawned"):
            raise BackendError(f"spawn of {actor!r} did not complete (pending={data.get('pending')})")
        if args["enter"]:
            self._entered = True
        poses = self._poses()
        ids = {str(a.get("instance_id")) for a in poses}
        fresh = [a for a in poses if str(a.get("instance_id")) not in self._known_ids]
        missing = self._known_ids - ids
        if missing:
            self._removed += len(missing)
        if len(fresh) != 1:
            raise BackendError(f"spawn of {actor!r}: expected one new actor, found {len(fresh)}")
        new = fresh[0]
        handle = str(new.get("instance_id"))
        self._known_ids = ids
        engine_pos = yup_to_zup(new.get("position", [0.0, 0.0, 0.0]))
        actual_pos = tuple(engine_pos[i] - offset[i] for i in range(3))
        yaw = self._yaw_deg(new.get("yaw"))
        if yaw is None:
            actual = Pose(actual_pos)
            residual_deg = UNVERIFIED_ORIENTATION_DEG
            method = "spawn_vehicle(position)+bbox-compensation; orientation unverified"
        else:
            actual = Pose.from_yaw(actual_pos, yaw)
            residual_deg = pose.angle_to_deg(actual)
            method = "spawn_vehicle(position)+bbox-compensation; yaw read back from actor_poses"
        record = SpawnRecord(actor, pose, actual, pose.distance_to_m(actual), residual_deg, method)
        ref = ActorRef(actor, handle, obj.name, obj.seatable)
        self._spawned.append((ref, obj))
        self._spawns.append(record)
        return ref, record

    # ------------------------------------------------------------------ runs

    def run(self, actors: Sequence[ActorRef], programs: Mapping[str, Mapping[str, Any]], spec: RunSpec,
            run_dir: Path) -> RunResult:
        run_dir = Path(run_dir)
        plan = _read_plan(run_dir)
        if programs and (self.host is None or spec.host is not ControllerHost.ENGINE_SCRIPT):
            raise RefusedCall("controlled actors on OpenBeam need an engine-script host; none is configured")
        if not programs and spec.regime != STEPPED:
            raise RefusedCall("passive runs on OpenBeam use the documented stepped mode (simulated time comes "
                              "from the step count)")
        started = time.perf_counter()
        program_hashes = []
        for actor, program in sorted(programs.items()):
            blob = _program_blob(program)
            digest = self.ledger.store_blob(blob) if self.ledger is not None else hashlib.sha256(blob).hexdigest()
            if self.ledger is not None:
                self.ledger.record("v2e.program", {"actor": actor, "model": program.get("model"), "sha256": digest},
                                   blobs=[digest])
            program_hashes.append((actor, digest))
        marker = _run_marker(plan, spec.seed)
        self.guard.call("run_script", {"code": self.scripts.render("log_marker", marker=f"{marker}:begin")})
        if programs:
            rows_path, events, sim_s, frames = self._run_host(actors, programs, spec, run_dir, marker)
        else:
            rows_path, events, sim_s, frames = self._run_stepped_passive(actors, spec, run_dir)
        self.guard.call("run_script", {"code": self.scripts.render("log_marker", marker=f"{marker}:end")})
        manifest = self.manifest()
        attestation = Attestation(
            spawns=tuple(self._spawns), initial_velocity_mps=0.0, teleports=0, in_run_resets=0,
            removed_actors=self._removed, regime=spec.regime, programs=tuple(program_hashes),
            ledger=Path(self.ledger.directory) if self.ledger is not None else run_dir / "ledger",
            ledger_head_sha256=self.ledger.head_sha256 if self.ledger is not None else "", audit="pending")
        # content origins come from the compiler (the harness fills them from the compiled scene)
        return RunResult(manifest=manifest, plan=plan, effects=self.mechanisms(plan) if plan else (), content=(),
                         attestation=attestation, rows=rows_path, events=events,
                         t0_s=None if plan and any(m == "drive_in" for m in plan.modes.values()) else 0.0,
                         video=None, frames=frames, sim_s=sim_s, wall_s=round(time.perf_counter() - started, 3))

    def _state_rows(self, t: float, handles: Mapping[str, str]) -> dict[str, StateRow]:
        poses = {str(a.get("instance_id")): a for a in self._poses()}
        missing = [h for h in handles.values() if h not in poses]
        if missing:
            self._removed = max(self._removed, len(missing))
        states = {}
        for actor, handle in handles.items():
            raw = poses.get(handle)
            if raw is None:
                continue
            states[actor] = StateRow(actor=actor, t=t, p=tuple(yup_to_zup(raw.get("position", (0, 0, 0)))), q=None,
                                     v=tuple(yup_to_zup(raw.get("velocity", (0, 0, 0)))), w=None,
                                     extras={"yaw_reported": raw.get("yaw"), "asleep": raw.get("asleep")})
        return states

    def _run_stepped_passive(self, actors: Sequence[ActorRef], spec: RunSpec,
                             run_dir: Path) -> tuple[Path, dict[str, float], float, Path | None]:
        if not self.stepped_available:
            raise RefusedCall("stepped mode is available in training sessions or a game you run yourself")
        handles = {a.actor: a.handle for a in actors}
        offsets = {ref.actor: _reference_offset(obj) for ref, obj in self._spawned}
        hz = spec.telemetry_hz or 30.0
        per_row = max(1, int(round(1.0 / (hz * PHYSICS_DT_S))))
        total = max(1, int(round(spec.duration_s / PHYSICS_DT_S)))
        frames_dir = run_dir / "frames" if spec.record == STEPPED_CAPTURE else None
        if frames_dir is not None:
            frames_dir.mkdir()
        detector = EventDetector()
        rows_path = run_dir / "rows.jsonl"
        self.guard.call("sim_mode", {"mode": "stepped", "idle_release_s": 120})
        step = 0
        index = 0
        writer = RowWriter(rows_path, actors=list(handles), backend=self.name, engine=self.engine, regime=STEPPED,
                           dt_s=PHYSICS_DT_S, t0_s=0.0, attitude_observed=False,
                           notes="observed: position and velocity of each actor's reference point (no attitude "
                                 "through the documented tools)")
        try:
            while True:
                t = round(step * PHYSICS_DT_S, 9)
                self.guard.sim_t = t
                states = {}
                for actor, s in self._state_rows(t, handles).items():
                    off = offsets.get(actor, (0.0, 0.0, 0.0))
                    states[actor] = StateRow(actor, t, tuple(s.p[i] - off[i] for i in range(3)), None, s.v, None)
                writer.write(t, states, None, detector.update(t, states))
                if frames_dir is not None:
                    self._capture(frames_dir, index, spec)
                index += 1
                if step >= total:
                    break
                todo = min(per_row, total - step)
                while todo > 0:
                    chunk = min(todo, MAX_SIM_STEPS_PER_CALL)
                    self.guard.call("sim_step", {"steps": chunk})
                    todo -= chunk
                    step += chunk
        finally:
            writer.close()
            self.guard.sim_t = None
            self.guard.call("sim_mode", {"mode": "realtime"}, raise_on_error=False)
        telemetry = self.guard.call("telemetry", {}, raise_on_error=False)
        if not telemetry.is_error:
            (run_dir / "telemetry_end.json").write_text(json.dumps(telemetry.data, indent=2, sort_keys=True) + "\n",
                                                        encoding="utf-8")
        return rows_path, dict(detector.events), total * PHYSICS_DT_S, frames_dir

    def _capture(self, frames_dir: Path, index: int, spec: RunSpec) -> None:
        width = spec.cameras[0].width if spec.cameras else 640
        result = self.guard.call("camera_frame", {"max_width": int(width), "format": "png"})
        if result.images:
            mime, data = result.images[0]
            suffix = ".png" if "png" in mime else ".jpg"
            (frames_dir / f"{index:06d}{suffix}").write_bytes(data)
        with (frames_dir / "index.jsonl").open("a", encoding="utf-8") as fh:
            meta = {k: result.data.get(k) for k in ("sim_time", "width", "height", "fov_y_deg", "camera_pose")}
            fh.write(json.dumps({"index": index, **meta}, sort_keys=True) + "\n")

    def _run_host(self, actors: Sequence[ActorRef], programs: Mapping[str, Mapping[str, Any]], spec: RunSpec,
                  run_dir: Path, marker: str) -> tuple[Path, dict[str, float], float, Path | None]:
        host = self.host
        assert host is not None
        handles = {a.actor: a.handle for a in actors}
        hz = spec.telemetry_hz or 30.0
        for snippet_id, values in host.start(programs, handles, marker, spec.duration_s, hz):
            result = self.guard.call("run_script", {"code": self.scripts.render(snippet_id, **dict(values))})
            if result.data.get("succeeded") is False:
                raise BackendError(f"host snippet {snippet_id} did not succeed; read the script log")
        if spec.regime == STEPPED:
            if not self.stepped_available:
                raise RefusedCall("stepped mode is available in training sessions or a game you run yourself")
            self.guard.call("sim_mode", {"mode": "stepped", "idle_release_s": 120})
            try:
                todo = max(1, int(round(spec.duration_s / PHYSICS_DT_S)))
                while todo > 0:
                    chunk = min(todo, MAX_SIM_STEPS_PER_CALL)
                    self.guard.call("sim_step", {"steps": chunk})
                    todo -= chunk
            finally:
                self.guard.call("sim_mode", {"mode": "realtime"}, raise_on_error=False)
        else:
            deadline = time.monotonic() + spec.duration_s + 2.0
            while time.monotonic() < deadline:
                self.sleep(min(2.0, max(0.0, deadline - time.monotonic())))
                self.guard.call("status", {})
        rows: list[dict] = []
        page = 0
        while True:
            text = self.guard.call("read_log", {"log": "script", "grep": host.row_grep(marker, page),
                                                "lines": 2000}).text
            lines = [line for line in text.splitlines() if line.strip()]
            parsed = [r for r in (host.parse_row(line, handles) for line in lines) if r is not None]
            rows += parsed
            if len(parsed) < 1000:
                break
            page += 1
        rows.sort(key=lambda r: r["t"])
        detector = EventDetector()
        rows_path = run_dir / "rows.jsonl"
        with RowWriter(rows_path, actors=list(handles), backend=self.name, engine=self.engine, regime=spec.regime,
                       dt_s=PHYSICS_DT_S, t0_s=0.0) as writer:
            for row in rows:
                states = {a: StateRow(a, row["t"], tuple(s["p"]), tuple(s["q"]) if s.get("q") else None,
                                      tuple(s["v"]) if s.get("v") else None, tuple(s["w"]) if s.get("w") else None)
                          for a, s in row["actors"].items()}
                writer.write(row["t"], states, row.get("cmd"), detector.update(row["t"], states))
        sim_s = rows[-1]["t"] if rows else 0.0
        return rows_path, dict(detector.events), sim_s, None

    def reset_between_runs(self) -> None:
        if self._level is not None:
            self.open_scene(self._level)

    def close(self) -> None:
        if self._end_session is not None:
            try:
                self._end_session()
            finally:
                self._end_session = None
        close = getattr(self.client, "close", None)
        if callable(close):
            close()


def _read_plan(run_dir: Path) -> Plan | None:
    path = Path(run_dir) / "plan.json"
    if not path.is_file():
        return None
    return Plan.from_json(json.loads(path.read_text(encoding="utf-8")))


def _reference_offset(obj: ObjectRef) -> tuple[float, float, float]:
    if obj.bbox_min_m and obj.bbox_max_m:
        lo, hi = obj.bbox_min_m, obj.bbox_max_m
        return ((lo[0] + hi[0]) / 2.0, (lo[1] + hi[1]) / 2.0, lo[2])
    return (0.0, 0.0, 0.0)
