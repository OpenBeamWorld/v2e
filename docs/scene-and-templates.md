# Scenes and templates

A scene (`v2e.scene/1`) says what happens in a clip without naming an engine. A template
mapping (`v2e.templates/1`) says which authored content stands in for each object. Both are
JSON; their JSON Schemas are in `openbeam_v2e.schema` (`openbeam_v2e.schema.export_json_schemas`
writes them as files for any JSON Schema tool), and `openbeam_v2e.schema.check_file(path)` validates
any V2E document, including the cross-reference checks below.

## Scene

```json
{"schema": "v2e.scene/1", "world": "world/",
 "actors": [
   {"id": "car", "class": "vehicle.car", "dims_m": {"length": [4.4, 0.05]}, "seatable": true,
    "control": "driver",
    "initial_state": {"mode": "drive_in", "frame0_pose": "track:car@0", "speed_mps": [12.0, 0.4]}},
   {"id": "cone", "class": "prop.cylinder", "control": "passive",
    "initial_state": {"mode": "at_rest", "pose": "track:cone@0"}},
   {"id": "wall", "class": "static", "mesh": 2, "role": "collision"}],
 "events": [{"name": "impact", "frame": [40, 41], "kind": "contact", "actors": ["car", "wall"],
             "evidence": "outline touches the wall"}],
 "phases": [{"name": "approach", "from": "start", "to": "impact"},
            {"name": "after_impact", "from": "impact", "to": "rest"}],
 "targets": [{"metric": "speed_at_event", "actor": "car", "event": "impact", "value": [12.0, 0.4],
              "source": "design_value"}],
 "requires": {"effects": ["drive", "brake", "steer", "contact"],
              "observables": ["position", "attitude"], "controlled_actors": 1}}
```

- **Actors.** Dynamic actors have a class `vehicle.<kind>` or `prop.<kind>`, optional measured
  dimensions and mass (`[value, sigma]`), a control mode (`driver`, `passive` or a
  plugin-declared `<plugin>.<name>`), an optional symmetry class (`none`, `box`, `cylinder`,
  `cylinder_symmetric`, `sphere`; defaults follow the class) and an initial state. Static actors
  name a mesh of the world package and its role (`collision`, `drivable`, `visual`). `object`
  maps an actor to a differently named object of the world package.
- **Initial states never give an actor speed.** `at_rest`: spawn motionless at `pose`.
  `drive_in`: spawn motionless at the start of a generated approach and let the controller
  reach `frame0_pose` at `speed_mps`. `moving`: a passive object already in motion at frame 0;
  it needs `launcher` (an engine-native mechanism from authored content), otherwise the
  compiler refuses the scene. Poses are explicit (`{"position_m": [...], "quat_xyzw": [...]}`)
  or references `track:<object>@<frame>` into the world package.
- **Events** are inclusive frame intervals with a kind (`contact`, `separation`, `rest`,
  `custom`; guessed from the name when missing) and the actors involved. Half the interval
  width, floored at half a frame, is the timing sigma.
- **Phases** run between events; `start`, `end` and an undeclared `rest` (when every scored
  object has come to rest in the reference) are reserved boundaries. Scores are per phase.
- **Targets** name a metric (see [scoring](scoring.md)), the actor, event or phase it applies
  to, a value with sigma and a `source`: `first_principles`, `design_value`,
  `published_measurement`, `surveyed_clip`, `independent_measurement` or `reconstruction`.
  Engine runs are never a source. `tolerance_sigma` (default 2) sets the pass band.
- **Requires** lists the effects, observables and actor counts a connector must offer.

The validator also checks what the schema language cannot: unique ids, phases that refer to
declared events, targets that refer to known actors and phases, no velocity-like fields in
initial states, drive-in only for controlled vehicles, and actor counts within `requires`.

## Templates

```json
{"schema": "v2e.templates/1",
 "defaults": {"vehicle.*": "vehicle.generic"},
 "objects": {
   "cone": {"template": "prop.cylinder", "density_class": "rubber"},
   "van": {"template": "vehicle.stock", "rule": "nearest_by_dims",
           "tolerance": {"length": 0.08, "width": 0.06}, "fallback": "vehicle.generic"},
   "crate": {"template": "prop.box",
             "overrides": {"mass_kg": {"value": 18.0, "sigma": 0.5, "status": "measured",
                                       "reason": "weighed before the test"}}}}}
```

| Template | Content |
| --- | --- |
| `vehicle.generic` | A vehicle generated from the measured dimensions (default for every vehicle class). |
| `vehicle.stock` | A stock vehicle from the connector's catalog: `nearest_by_dims` (weighted relative error over length, width, height and wheelbase, refused beyond the tolerances) or `content_id`. |
| `prop.box`, `prop.cylinder`, `prop.sphere`, `prop.mesh` | A prop sized by the measurements; mass from a density class times the volume unless measured. |
| `static.mesh` | Level collision from a static mesh. |

Lookup order for an actor: its own entry in `objects`, then `defaults` (exact class, then the
longest matching pattern), then the built-in defaults. Overrides carry a status (`measured` or
`assumed`; nothing is `fitted` in this release) and a reason, and are recorded in the run's
content-origin record.
