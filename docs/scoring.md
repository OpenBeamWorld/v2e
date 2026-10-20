# Scoring

```python
from openbeam_v2e.score import score_run, score_worlds, repeat_report

report = score_run("runs/one", scene="scene.json", out="runs/one/score.json")
```

`score_run(run, reference=None, scene=None, out=None)` accepts a connector run directory (its
rows are exported onto the reference timeline first), a world package directory or an
in-memory `World`. The reference defaults to the scene's world package. `score_worlds` does the
same on two `World` objects. The report is `v2e.score/1`; it works for any number and kind of
objects and has no scene-specific constants.

## What is measured

| Block | Contents |
| --- | --- |
| `phases[].objects.<actor>` | position error at the hull's bounding-box centre (RMS, max, mean, RMS z-score against the reference's per-frame sigma), attitude error modulo the object's symmetry (RMS, max, RMS z-score), speed error, coverage, and silhouette IoU |
| `events[]` | each scene event (or the reconstruction's events) with its reference interval, the run's frame (the connector's reported event of the same name, else a detected event of the same kind and actors), the delta in frames and seconds, the sigma, a z-score and `within_sigma` |
| `objects.<actor>.at_events` | position, speed, yaw, pitch and roll in both worlds at each event, with position, speed and attitude differences |
| `objects.<actor>.after_events` | after each contact: rebound speed, displacement and yaw change to rest, time to rest, rest pose |
| `objects.<actor>.rollover` | vehicles: cumulative roll about the initial travel direction, quarter turns, roll onset, rest attitude (`wheels_down`, `side`, `roof`) |
| `objects.<actor>.bounce` | props: ground-contact times, apex heights, rebound ratios, bounce count, rest time and pose |
| `alignment.<actor>` | drive-in actors: position, heading and speed residual at frame 0 |
| `targets[]` | each scene target with the run's value, z-score and pass/fail |
| `claim` | the run's claim class (`native`, `emulated`, `scripted`, `incomplete`, `test_double`), provenance signature, validity, effects and content |
| `summary` | worst position and attitude RMS, mean IoU, events found and in sigma, targets passed, and `objective` |

**Attitude modulo symmetry.** Footage cannot show rotations that map an object onto itself,
so the error is the smallest angle over the object's symmetry group: none for vehicles; for
boxes the half and quarter turns allowed by the measured extents (the full cube group when all
sides match); for cylinders only the axis direction (and not its sign when the ends match);
spheres are not scored.

**Silhouettes.** The run's hull at the run's pose is drawn through the reference camera of
each sampled frame and compared with the observed outline (`v2e/outlines.json`) or, without
one, with the reference hull at the reference pose. Next to the raw IoU the report gives the
shape limit, the IoU the run's content reaches when placed exactly at the reference pose, and
the ratio of the two, which separates pose error from template shape mismatch.

**Events from tracks.** Contacts use exact distances from hull vertices to the other body's
triangles (objects against every static mesh and every other object). A pair touching at the
first frame starts in contact, so an object resting on the ground has no ground "contact"
event. A hit and bounce can happen entirely between two video frames; the detector still
counts it when the sampled distance dips to a minimum that is closer than the tolerance plus
half of the distance moved between neighbouring frames. An object is at rest from the first
frame of its last still stretch (under 0.05 m/s and 3 deg/s, lasting at least 0.5 s).

**Target metrics.** `speed_at_event`, `yaw_at_event`, `time_of_event`, `rebound_speed`,
`displacement_after_event`, `yaw_change_after_event`, `time_to_rest_after_event`,
`cumulative_roll_deg`, `quarter_turns`, `roll_onset_time`, `first_contact_time`,
`first_apex_height`, `first_rebound_ratio`, `bounce_count`, `rest_time`, `trajectory_rms`,
`attitude_rms` and `silhouette_iou` (`openbeam_v2e.score.TARGET_METRICS`).

**Objective.** One number for search, lower is better: every error is divided by the
uncertainty it should be judged against, squared, and averaged. Position and attitude RMS use
the reference's median sigma (at least 5 cm and 2 deg), event timing uses the event's sigma,
and coverage gaps, objects missing from the run and target z-scores add their own terms.
Search ranks candidates by the median objective over repeats.

## Honesty rules

A reference that is itself an engine run, or engine-rendered synthetic truth, is labelled in
`warnings`: such comparisons support no accuracy claim. Claim classes are computed from the
run's effects, never typed in.

## Repeats

Engines whose runs are not bit-identical are compared on N repeats.
`repeat_report(reports)` (with `None` for failed runs) returns `v2e.score_repeat/1`: per metric
the median, the MAD scaled to a normal sigma, the robust interval `median +- 3 * 1.4826 * MAD`,
IQR, min and max, plus the failure fraction, and a warning when runs mix claim classes or
provenance signatures.
