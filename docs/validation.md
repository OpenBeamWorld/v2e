# Validation and remaining gates

This private pre-alpha has working offline integration, not release qualification.
Passing tests, completing a benchmark row, and meeting accuracy targets are different things.

## Track status

| Track | Verified offline | Still open |
| --- | --- | --- |
| Perception | CPU calibration, pan/tilt/zoom tracking, segmentation, pose fitting, uncertainty and world-package round trips; registered model replacement; optional adapters tested with stand-ins. | The default four-family benchmark exposes large reconstruction errors. No real-footage accuracy gate or downloaded-model acceptance has passed. |
| Backends and agents | Fake and cassette OpenBeam contracts; allowlisted calls and ledger audit; separate CLI and stdio MCP processes; all four generated families through the general compiler, run, export and score tools. | No live OpenBeam acceptance. Controlled OpenBeam runs need an accepted script host. Full attitude is not available from the documented tools; position-only logs cannot be exported as full-pose worlds. |
| Benchmark | Four generated families, deterministic truth generation, baseline and oracle report rows, explicit controller programs and reproducible walkthroughs. | Reconstruction quality, independent physical targets missed by the fake, the full 40-case grid, engine-rendered truth and real measured clips. |

The contract suite deliberately uses oracle worlds to isolate compiler/connector integration.
The reconstruction suite separately checks a smaller synthetic moving-car clip against stated
camera, trajectory, dimension and held-out silhouette tolerances. Neither replaces the harder
multi-family benchmark or independent real footage.

## Offline test checkpoint

On 20 October 2026, the full offline suite passed **851 tests** on each environment:

| Python | NumPy | SciPy | Result |
| --- | --- | --- | --- |
| 3.12.10 | 2.5.4 | 1.18.1 | 851 passed, 3 live tests deselected, no warnings |
| 3.10.20 | 2.2.6 | 1.15.3 | 851 passed, 3 live tests deselected, 276 numerical RuntimeWarnings |

The warnings are unsuppressed and remain under investigation, including matrix products in the
renderer and least-squares solver. The preceding 761-test checkpoint had 246 warnings; the joint
dimension checkpoint has 30 more. The earlier increase of 60 since the 739-test checkpoint was
separately reproduced by ten new rendered pose fixtures at existing renderer matrix products.
The physics safeguards add 27 stage checks and one scaffold-header check, with no additional warnings.
The sampled-contact fix adds 34 geometry/scorer checks and one scaffold-header check.
Offline CI also tests Python
3.10 and 3.12 and builds and installs a wheel; passing that job is not live-engine acceptance.

## Default benchmark checkpoint

Measured with the CPU baseline, tripod cameras, 320 x 180 frames and a 0.002 s fake-physics
step. All eight rows completed, but the **reconstruction quality gate is not met**:

| Family | Reconstructed path RMS (m) | Mean attitude error (deg) | Recreated path RMS (m) | Targets met |
| --- | --- | --- | --- | --- |
| Barrier crash | 0.17 | 2.4 | 2.30 | 0/3 |
| Ramp jump | 3.74 | 68.1 | 24.76 | 0/4 |
| Rollover | 2.18 | 65.4 | 15.64 | 1/2 |
| Drop and bounce | 9.15 | 52.3 | 8.58 | 0/3 |

These measurements include the physics safeguards below, following the dimension, segmentation,
rotated-line driver and image-border pose fixes. Vehicle results are identical to the preceding
dimension checkpoint. Drop reconstruction slightly **worsens from 9.14688 to 9.14821 m**, and its
recreated path improves only from 8.60 to 8.58 m. Baseline physical targets **regress from 2/12 to
1/12** after rejecting an incompatible ballistic correction; that regression is not hidden by the
passing tests. The earlier crash/rollover recreation regressions from 0.89/12.18 to 2.30/15.64 m
also remain. Heading ambiguity, pre-visibility extrapolation, ground-constrained airborne vehicles
and free-body scale/depth are unresolved. This is a bounded correctness fix, not accuracy acceptance.

The oracle rows bypass perception. Their recreated path errors are 0.19, 0.12, 0.08 and
less than 0.01 m respectively, with silhouette IoU of 0.91, 0.92, 0.96 and 1.00. The ramp
still misses all four design-speed targets even with oracle perception: the fake vehicle
loses speed on the ramp. These are diagnostic failures, not targets to relax until green.

Named event-time targets read the run's event map; an absent event remains unmet, never a zero timing error.
The benchmark's dense-track metrics and the package scorer's frame-sampled metrics are still
separate implementations: matching names does not make their values equivalent. In particular,
the package scorer samples impact speed at the event frame, which may already include contact
deceleration. The four benchmark-only targets (`apex_rise`, `airborne_time`, `landing_distance`,
`roll_onset_allowed`) are not evaluated by the general scorer.

All these results are **synthetic self-consistency**, labelled `test_double`. They are not
evidence of accuracy on real videos, soft-body physics fidelity or native OpenBeam behaviour.

## Segmentation diagnostic

A paired check on the default clips compared the previous baseline with the prompt-guided
background and backward-tracking fixes. Both variants received the same RGB frames, sparse
annotations and previously estimated camera track. Truth masks were used only to score outputs.
An absent mask scores zero overlap on every truth-visible frame; interpolated poses do not
count as mask observations. The overlap below is the mean intersection-over-union (IoU).

| Object | Visible frames | Observed before → after | Mean IoU before → after |
| --- | --- | --- | --- |
| Crash car | 84 | 5 → 84 | 0.037 → 0.976 |
| Jump car | 84 | 79 → 84 | 0.884 → 0.953 |
| Rollover car | 80 | 67 → 79 | 0.787 → 0.955 |
| Drop ball | 105 | 31 → 105 | 0.249 → 0.933 |
| Drop box | 105 | 20 → 71 | 0.172 → 0.501 |
| Drop cylinder | 105 | 21 → 105 | 0.174 → 0.947 |

Neither variant emitted masks on frames where truth was entirely out of view. The drop box
still loses 34 visible frames; more observations alone do not establish accurate pose, size or
recreation. Regression tests separately cover signed backward prediction, a stopping object,
image-edge preservation, scaled automatic tracking, and refusal to turn exclusion boxes into
observations or invent a background that was never uncovered.

A subsequent later-anchor recovery check raises the drop box from **71/105 to 105/105** observed
visible frames and mean IoU from **0.501 to 0.783**, with the other five tracks effectively
unchanged and no new masks on wholly invisible frames. The three visibility-gap unit cases
isolate association with an RGB colour-threshold foreground stand-in; the benchmark diagnostic
uses the actual CPU background model. Neither receives dense truth masks as model inputs.
Additional regressions protect forward observations, annotation pixel ownership (including
three overlapping tracks), and prompts outside the requested sampling grid. This closes the
box-mask coverage loss in this default synthetic clip, not general occlusion or pose accuracy.
The drop path RMS changes only from 9.17 to 9.15 m, its recreated path worsens slightly from
8.56 to 8.60 m, and physical target acceptance was **2/12** overall at that checkpoint.

The first end-to-end rerun exposed a separate driver bug: a rotated run-up used global-Y line
error, so the crash recreation never reached its frame-0 trigger. The cruise driver now uses
signed perpendicular distance to the line through its configured frame-0 position and heading.
Rotation/translation checks and two obstacle-free drive-in checks cover this without changing
actuator permissions, reference trajectories or targets. All eight rows completed after the fix;
the inaccurate reconstructed heading remains inaccurate, not replaced with the oracle heading.

## Image-border pose and heading diagnostic

The silhouette fitter formerly treated unseen hull edges outside the image as fitting errors,
pulling partly visible vehicles into the frame and distorting their headings. It now compares
the hull's image-clipped projection. Ground fits start at the first untruncated fitting mask and
revisit earlier masks backwards, with a configurable weak prior on the adjacent fitted heading.
This prior does not constrain heading to translational velocity or force positive entry speed.

Twenty-two added checks cover all four borders, clipping geometry, offscreen penalties, forward
and reversing motion, rotated entry tracks across the angle wrap, overriding the prior with
visible evidence, invalid prior settings and the no-complete-mask fallback. The rendered pose
fixtures isolate that stage using exact masks/cameras and class dimension priors; single-frame
fit checks also start from perturbed fixture poses. Eight substantive border/entry cases fail
against the previous committed baseline. Held-out masks remain excluded from fitting and seed
selection, and entirely hidden frames do not become observations.

In the preceding image-border benchmark, both versions used the same rendered RGB hashes, sparse prompts and
estimated cameras. Crash frame-0 yaw improves from -77.9 to -6.2 degrees, and rollover from
131.3 to 4.4 degrees; rollover now infers a positive 16.05 m/s entry instead of an at-rest
recreation. But its estimated frame-0 lateral position is still wrong by several metres.
At that checkpoint the jump had a bad -96.4-degree entry heading, negative signed entry speed,
and an at-rest recreation under the existing benchmark rules. No signed-speed clamp was added or
changed. Crash's maximum dimension error also **worsened from 0.20 to 0.51 m**, despite much
better path/attitude and silhouette fits. Local covariance and a high overlap do not resolve
ambiguous outlines or a ground model fitted to an airborne vehicle. These regressions remain
documented; the dimension checkpoint below supersedes those measurements, never with oracle rows.

## Dimension measurement and model compatibility

Holding a wrong-size fitted pose fixed can lock its size bias into subsequent measurement.
Ground dimension fits now jointly optimize the selected frames' x/y positions and yaw, then
refit the full track. Those refits reconsider both front/back image hypotheses; they never
choose a body heading from the sign of travel speed. Free-body dimension fits still hold
poses fixed, and explicitly label their covariance as conditional on those poses.

Measurement frames must pass a ground-fit outline RMS limit: `dims_max_rms_sigma` defaults
to twice the assumed boundary noise (2 px with the default 1 px sigma), or accepts null to
disable the gate. Time-spread selection operates only on compatible fitting frames, never
held-out masks. If none pass at the initial size, a provisional conditional-size step refits
poses and retries the same gate. That proposal is not itself a dimension measurement; if no
frame passes, or the final solve fails, the previous dimensions and uncertainty are retained.
Diagnostics record selected/rejected frames, the limit, bootstrap attempts and solver outcome.

This is a necessary compatibility check, **not proof of ground contact**. It can still reject
valid objects with poor starting size/template, or admit a wrong motion model that happens to
fit an outline. Joint covariance includes local size/pose coupling, not calibrated uncertainty
over incorrect motion models, template mismatch or competing modes. Free-body scale/depth
ambiguity remains unresolved. Neither a small covariance nor high IoU establishes accuracy.

Twenty-six new checks isolate dimensions with exact rasterized masks/cameras, non-default
sizes, perturbed pose starts for direct solver checks, and separate held-out views for track
checks. The sparse-view fixture has constant angular rate so its held-out interpolation is
specified. The two non-default-size track cases fail the preceding committed baseline's size
tolerance. Control-flow checks cover noise scaling, excluded held-out masks, rejected/nonfinite
fits, failed/no-evidence fallbacks and configuration. These are not video-perception acceptance.

The dimension checkpoint used identical RGB hashes, sparse prompts, estimated cameras and segmentation
to the previous checkpoint. Crash width changes from 1.29 to 2.08 m (truth 1.80 m), reducing maximum
dimension error from **0.51 to 0.28 m**; rollover's maximum dimension error improves **0.43 to 0.22 m**.
Jump's maximum size error slightly **worsens from 0.37 to 0.38 m**, and mean attitude error also
worsens. Its entry heading changes to a still-wrong 62.4 degrees and positive 18.98 m/s signed
speed, so its recreation now moves but still misses the ramp targets. Crash's entry heading is
8.3 degrees. Recreated crash/rollover paths both regress as shown above. No signed-speed rule,
physical target or tolerance was changed. Acceptance at that checkpoint was **2/12**, not passed.

## Motion evidence and optional physics safeguards

A stage-isolated diagnostic compared ground and six-degree-of-freedom fits with the same saved
estimated masks/cameras and fixed inferred dimensions. Candidates used fitting frames only;
held-out masks scored predictions, and dense truth was read only after every candidate track and
selection manifest was saved. This is not a full video-inference benchmark or validated classifier.
Thirty of 56 stronger complete-outline free candidates put the hull more than 0.1 m below ground.
Simple ground/free switches still produced below-ground smoothed tracks even when those per-frame
penetrations were rejected. Better outlines alone do not resolve contact, scale/depth or heading.
No automatic motion switch was adopted; the vehicle ground constraint remains unchanged.

The investigation found a separate physics-stage defect: missing flight-event pairs caused an
unconditional whole-clip ballistic fit, including for a resting object. The corrected stage uses
all sustained-clearance intervals, requires at least three fitted/observed/non-held-out poses,
screens the weighted residual per degree of freedom (default 4), and rejects ground penetration.
Rejected proposals preserve poses, velocities and residuals. Accepted proposals refresh the final
silhouette IoUs; held-out masks never select a proposal, and earlier IoUs remain in diagnostics.
Separate flights and clip boundaries no longer require an invented event pair.

Clearance is not proof of gravity-only motion. The consistency screen is an uncalibrated heuristic
for correlated smoothed poses, and the model acts on the body origin, not a measured centre of mass.
Input covariance is retained and explicitly labelled, not replaced by a falsely precise conditional
ballistic covariance. Rotation, support and other forces remain model-mismatch risks.

Twenty-seven new checks isolate this stage using supplied synthetic poses, masks and cameras
through registered stand-ins, not RGB perception. Seven pipeline regression cases fail the previous
committed baseline, covering no-flight and uncertain-clearance fallbacks, supported objects, multiple
flights/boundary pairing, insufficient fitting evidence and stale final residuals. These include
numerical, control-flow and reporting checks, not seven accuracy acceptances. The baseline moves
a resting free object's origin about 0.46 m below ground; the corrected pipeline leaves it untouched.

The final full benchmark again uses identical RGB hashes, estimated cameras, segmentation counts
and provenance. It rejects the box's ballistic proposal (weighted residual/dof 4.91) and the ball's
(13.39, also penetrating ground by 0.33 m), while accepting the cylinder's (0.95). Sizes, attitudes,
observation flags and input covariance are unchanged; vehicle positions and IoUs are identical.
The ball's uncorrected entry height changes its recreated dense-benchmark first-contact time from
0.660 to 0.704 s, outside the unchanged 0.639 s target's 0.033 s tolerance. Drop loses its one target
pass, reducing the baseline total to **1/12**; oracle remains **8/12**. No threshold was tuned to
recover the pass. At that checkpoint the package scorer reported 1.60 s for that same named ball
event, not 0.704 s. The sampled-contact diagnostic below corrects that particular missed bounce;
neither the dense nor corrected sampled value meets the target.

## Sampled-contact diagnostic

The contact detector's broad-phase search returned infinity for distances beyond its search
cutoff. Its between-frame local-minimum test required finite neighbouring samples, so a fast
approach/rebound could lose the very samples needed to detect its first contact. The detector now
queries uncapped vertex-to-triangle distances only for search-censored neighbours of candidate
minima. Distant pairs retain the broad-phase shortcut. Unknown poses remain unknown, and neither
the contact tolerance nor the between-frame cap has been increased. The search margin also
respects the contact tolerance when it exceeds the between-frame cap.

Thirty-four checks use closed-form sampled gaps with box, sphere and cylinder meshes, varied
speeds/frame rates/sampling offsets, moving body pairs and rigid coordinate changes. They also
cover near misses, unknown neighbours, clip boundaries, initial support, named/reported events,
distinct successive contacts and absent-event targets. Twenty-three fail on the preceding
committed baseline. These isolate geometry/scoring, not pose reconstruction or footage accuracy.

For the same saved synthetic drop run, the package scorer now detects the ball at frame 21
(0.700 s), rather than the later bounce at frame 48 (1.600 s). The dense benchmark still reports
0.704 s; both remain outside the unchanged 0.639 s target's 0.033 s tolerance. The dense target
result stays **1/12** overall, oracle **8/12**. No reconstructed pose, recreated trajectory,
target, event name or event-matching rule was changed. Frame-local contact remains a heuristic,
not continuous collision detection: brief contacts outside the cap or near unknown/boundary
samples can still be missed, and a sufficiently close near miss can satisfy that heuristic.
General event matching still uses the nearest compatible detected event when no named report
exists; it does not guarantee one-to-one ordinal matching for arbitrary overlapping intervals.
Impact-speed sampling and other metric differences remain separate open work.

## Reproduce the checks

```sh
python -m pip install -e '.[test]' build
python -m pytest -q
python -m pytest -q tests/contract/test_offline_pipeline.py
python -m openbeam_v2e.bench synth run --out runs/validation --width 320 --height 180
python -m build --wheel --sdist
```

The benchmark writes JSON, CSV and Markdown reports. Inspect their errors and target results:
exit status zero and `status: ok` only mean its steps completed. Live-engine tests remain
opt-in (`-m openbeam`); do not start paid or shared engine sessions just to make offline CI pass.

## Next quality work

1. Evaluate temporally coherent, contact-constrained motion hypotheses for jumps and rollovers,
   including initial heading/position and pre-visibility extrapolation. Simple per-frame free-pose
   switching was rejected above; do not repeat it as acceptance or compensate with oracle poses.
2. Diagnose dimension bias and remaining mask-boundary/association errors (box mean IoU is still
   only 0.783); validate airborne props' scale/depth rather than trusting a broad size prior.
3. Add held-out benchmark regression thresholds only after documenting the intended operating
   conditions. Do not fit to oracle trajectories, replace bad baseline rows with oracle rows,
   or count missing events as zero error.
4. Then validate independently measured real clips and optional model weights. Qualify the
   live connector separately, including axes, spawn offsets, orientation conventions and host API.
