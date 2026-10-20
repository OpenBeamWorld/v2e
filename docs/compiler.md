# The compiler

```python
from openbeam_v2e.compile import compile_scene, CompileRefused

try:
    result = compile_scene("scene.json", "templates.json", "catalog.json", out="build/")
except CompileRefused as exc:
    for reason in exc.refusals:
        print(reason.code, reason.actor, reason.message)
```

`compile_scene(scene, templates=None, catalog=None, world=None, connector=None, out=None)`
takes objects or file paths. Without a catalog, the named connector gets one with no stock
content, so every actor gets generated content. With `out`, it writes `package.json` (a
`v2e.package/1` request) and the level geometry under `level/`.

## What the request holds

- **`level`**: one collision entry per static mesh of the world package (meshes with role
  `visual` are left out), each written as a Wavefront OBJ file in Z-up metres, with its bounds,
  role and SHA-256. Synthesised ground carries `observed: false`; scores never use it.
- **`content`**: one content-origin record per dynamic actor: the template, `generated` or
  `stock`, every parameter with value, sigma, status (`measured`, `assumed`, `catalog`) and
  source, per-dimension residuals for stock matches, and a hash of the parameters.
  Generated vehicles scale unmeasured values (wheelbase, track, mass, ...) from a round
  mid-size design point (4.75 x 1.80 x 1.45 m, 1500 kg); props get their mass from a density
  class unless the scene or the templates measured it.
- **`spawns`**: one pose per actor, always at rest (`at_rest: true`,
  `initial_velocity_mps: 0`; the schema has no field for anything else). Reconstructed poses
  that sink into the ground are lifted out and the lift is recorded.
- **`timeline.t0_s`**: the simulated time that corresponds to video frame 0.
- **`controllers`**: for each controlled actor, a time-indexed reference (path, arc length,
  times and speeds) that its program must follow; passive actors are marked `passive`.

## Actors that drive in

An actor with `mode: "drive_in"` spawns at the far end of an approach and accelerates at
2.0 m/s^2 to its observed speed, then holds it for 10 m before its frame-0 pose:
`L = v^2 / (2 a) + 10 m`. The approach continues the first half second of the observed path
backwards, as a straight line or, when that path bends, as an arc of the fitted radius. Where
the reconstructed ground stops, a flat slab of synthesised ground carries the approach.

A corridor along the approach (the actor's width plus 2 m on each side, 3 m high) must be
free of static geometry and of every other object's frame-0 hull; otherwise the scene is
refused with `corridor_blocked`.

When several actors drive in, the one that needs the most time to reach its frame-0 pose sets `t0`; the others start
moving later, so every one of them reaches its frame-0 pose at that same moment. A scene in
which nobody drives in gives the engine half a second to settle before frame 0. If some object
is in mid-air in the first frame, it was let go right then, so frame 0 is the spawn itself.

## Refusals

`seatable_limit`, `controlled_limit` (the catalog's engine limits), `template_unavailable`,
`no_stock_match`, `unknown_density_class`, `missing_dimensions`, `missing_static_mesh`,
`missing_world_object`, `pose_unknown`, `drive_in_too_slow`, `runup_too_curved`,
`corridor_blocked`, `moving_without_launcher` and `released_with_drive_in` (a released object
and a drive-in actor in one scene would need a delayed spawn, which connectors cannot express
yet). All refusals are collected and reported together.

## Connector dialects

A dialect turns the request into what one connector loads.
`openbeam_v2e.compile.dialects.build_compiled(package, connector, world=...)` finds it: a plugin
registered under the connector's name in the `openbeam_v2e.dialect_packs` entry point group,
or a built-in dialect (the `fake` test connector ships its own).
