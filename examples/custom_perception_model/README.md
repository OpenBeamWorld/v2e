# Example: plug in your own perception model

`colour_key.py` is a complete perception model in about 90 lines: a segmenter that learns each
object's colour from its annotated outline and keeps that colour in every frame. It is registered
under the name `example.colour_key` and can replace the baseline segmenter in any
reconstruction, including the benchmark. The guide [Adding a perception model](../../docs/perception-models.md)
explains each part.

## Try it in Python

```python
import importlib.util
from openbeam_v2e.perception import registry

spec = importlib.util.spec_from_file_location("colour_key", "examples/custom_perception_model/colour_key.py")
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
registry.register(module.ColourKeySegmenter)

from openbeam_v2e.bench.runner import run_bench
report = run_bench("runs/colour-key", families=["drop_and_bounce"], reconstructions=("baseline",),
                   models={"masks": "example.colour_key"})
print(report.markdown())
```

`models={"masks": "example.colour_key"}` replaces only the segmentation stage; calibration,
camera tracking and pose fitting stay on the baseline. The report's `recon` column reads
`baseline+masks=example.colour_key`, so runs with different models sit side by side.

## Ship it as a package

Put the class in your own package and declare it under the `openbeam_v2e.perception` entry point
group; the registry finds it once the package is installed, and every tool can then name it:

```toml
# pyproject.toml of your package
[project.entry-points."openbeam_v2e.perception"]
colour_key = "my_package.colour_key:ColourKeySegmenter"
```

```sh
openbeam-v2e reconstruct clip/frames --config clip/reconstruct.json --model masks=example.colour_key --out clip/world
python -m openbeam_v2e.bench synth run --out runs/colour-key --model masks=example.colour_key
```

In a reconstruction configuration the same choice is `"segmentation": "example.colour_key"`,
or `{"model": "example.colour_key", "params": {"tolerance": 0.04}}` with parameters.
