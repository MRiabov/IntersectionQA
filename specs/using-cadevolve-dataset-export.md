# Using CADEvolve Dataset

CADEvolve is a synthetic CAD dataset built around executable CadQuery programs. The public release is distributed on Hugging Face as a WebDataset archive, so a consumer only needs the dataset artifact itself and a Python environment capable of reading tar members and executing CadQuery code.

For IntersectionQA, CADEvolve is the primary object source for released benchmark examples. Synthetic primitives should be kept to golden fixtures, smoke tests, and debugging examples rather than used as a separate full corpus before CADEvolve ingestion.

## Dataset at a glance

- Split: `train`
- Format: `webdataset`
- Size: about 2,004,385 rows
- Archive size: about 4.73 GB
- License: `apache-2.0`
- Paper: CADEvolve: Creating Realistic CAD via Program Evolution

The dataset card describes the release as a CAD reverse-engineering corpus. The paper abstract says the pipeline evolves a small set of primitives into about 8k parametric generators, then expands that into about 1.3M CadQuery scripts paired with rendered geometry.

## Suggested environment

For inspection only:

- `python`
- `datasets` or `huggingface_hub`
- `tarfile` from the Python standard library

For execution:

- `cadquery`
- a working geometry backend for CadQuery / OCP
- `numpy` if you want to read the generator embeddings

## Archive layout

The archive root has three top-level trees:

- `CADEvolve-G/` for generator artifacts and embeddings
- `CADEvolve-P/` for executable CadQuery programs
- `CADEvolve-C/` for executable CadQuery programs

Within those trees, the dataset card exposes these subfolders:

- `CADEvolve-G/parametric_generators.json`
- `CADEvolve-G/embeddings/`
- `CADEvolve-P/ABC-P/`
- `CADEvolve-P/CADEvolve-P-core/`
- `CADEvolve-P/ShapeNet-P/`
- `CADEvolve-C/ABC-C/`
- `CADEvolve-C/CADEvolve-C-core/`
- `CADEvolve-C/ShapeNet-C/`

The `.py` files are intended to be executed as Python source. Many rows define a geometry result in a variable named `result`, though some scripts may use other conventional names such as `shape`, `solid`, or `part`.

## Quirks and handling notes

- CADEvolve is distributed as a tar/WebDataset-style archive, but repeated local generation should not use the tar as the hot path.
- The Hugging Face viewer is useful for discovery. For actual code handling, download `cadevolve.tar` once, extract a deterministic executable subset, and materialize it into a local extracted source directory.
- The `CADEvolve-G`, `CADEvolve-P`, and `CADEvolve-C` trees are different kinds of content, not interchangeable subsets of the same representation.
- `CADEvolve-G` is for generator artifacts and embeddings.
- `CADEvolve-P` and `CADEvolve-C` are the executable CadQuery program trees you typically care about for geometry reconstruction.
- Archive paths usually include a leading `./`, so prefix filtering should account for that.
- Some scripts are tiny primitives, but others use more advanced CadQuery features such as sweeps, lofts, multiple workplanes, and helper functions.
- The `.py` files are arbitrary Python code, so they should be treated as untrusted input and executed in a separate process or sandbox.
- If a script does not place the final shape in `result`, inspect alternate output variables or intermediate functions before assuming it is invalid.

## Basic ways to use it

### 1. Browse or load the dataset directly

If you just need to inspect metadata, row counts, or the archive structure, the Hugging Face dataset page is the simplest entry point:

- [Hugging Face dataset card](https://huggingface.co/datasets/kulibinai/cadevolve)

You can also load it with the `datasets` library:

```python
from datasets import load_dataset

ds = load_dataset("kulibinai/cadevolve", split="train")
print(ds)
```

### 2. Download the archive and materialize a local source directory

If you need to run the CadQuery programs yourself, download `cadevolve.tar`
once, extract the executable member subset you intend to use into a local
directory, and then work from that directory. Keep the original archive member
path as provenance.

```python
from huggingface_hub import hf_hub_download
import tarfile

archive_path = hf_hub_download(
    repo_id="kulibinai/cadevolve",
    repo_type="dataset",
    filename="cadevolve.tar",
)

with tarfile.open(archive_path, "r:*") as tar:
    for member in tar:
        if not member.isfile():
            continue
        if not member.name.endswith(".py"):
            continue
        if not member.name.startswith("./CADEvolve-C/CADEvolve-C-core/"):
            continue

        source = tar.extractfile(member)
        if source is None:
            continue

        code = source.read().decode("utf-8", errors="replace")
        # Inspect before executing. CadQuery source is arbitrary Python.
```

IntersectionQA's loader does this automatically when
`smoke.use_extracted_source_cache` is enabled. The default extracted source
directory is:

```text
.cache/intersectionqa/cadevolve_sources/
```

This directory is a local build artifact. Public JSONL rows must continue to
store the original CADEvolve archive member path, not local filesystem paths.

After the bounded source directory is prepared, dataset generation should use
the directory directly instead of reopening the tar. Pass the exact extracted
directory with `--cadevolve-source-dir`. The archive is only needed to prepare
the directory. Generation intentionally does not auto-discover local caches,
because that would make default runs depend on machine-local state.

### 3. Execute scripts in a controlled environment

When you do execute source, keep the environment isolated and explicit:

```python
namespace = {"__name__": "__main__"}
exec(code, namespace, namespace)

shape = (
    namespace.get("result")
    or namespace.get("shape")
    or namespace.get("solid")
    or namespace.get("part")
)
```

If the script does not expose a usable shape, inspect its top-level variables or function returns. Some CADEvolve examples are straightforward procedural scripts; others rely on more advanced CadQuery constructs such as sweeps, lofts, multiple workplanes, or helper functions.

## What to expect from the code

CADEvolve is broader than a simple sketch-extrude dataset. Expect to see:

- multiple workplanes
- explicit `cq.Plane(...)` construction
- `sweep(...)`
- `loft(...)`
- `union(...)` and `cut(...)`
- `box(...)`, `circle(...)`, `rect(...)`, `polygon(...)`
- transformed workplanes and offset geometry

The generator set in `CADEvolve-G` is useful if you want retrieval, clustering, or embeddings-based sampling. The `P` and `C` trees are the executable CAD programs you would typically use for direct execution or translation.

## Practical workflow for a fresh project

1. Download `cadevolve.tar`.
2. Pick one subset directory such as `CADEvolve-C/CADEvolve-C-core/` or `CADEvolve-P/ShapeNet-P/`.
3. Sample a few `.py` files first.
4. Execute each file in a sandboxed Python process.
5. Export geometry to STEP if your downstream task needs geometric comparison or translation.
6. Keep a manifest of source member names, extracted file paths, and execution outcomes.

## Safety notes

- Treat every `.py` file as untrusted code.
- Do not run the dataset source in your main process.
- Prefer a temporary directory or isolated worker process for execution.
- Expect some programs to use CadQuery features that are not trivial to translate into other CAD kernels or DSLs.

## References

- [Hugging Face dataset card](https://huggingface.co/datasets/kulibinai/cadevolve)
- [Hugging Face README](https://huggingface.co/datasets/kulibinai/cadevolve/blob/main/README.md)
- [CADEvolve paper abstract](https://arxiv.org/abs/2602.16317)
