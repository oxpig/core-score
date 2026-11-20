# Changelog

## 0.6.0

- Supply verified SMILES-matched ligand connectivity to OpenStructure for both
  predictions and references, preserving observed atom correspondence and coordinates
  within SDF precision. Preserve the selected chemistry when separating ligand copies.
- Reuse the installed OpenStructure ligand CLI in persistent workers without caching
  structures or scores. Discover the OST environment through its executable instead
  of depending on project-specific environments or paths.
- Bound each OST comparison attempt to 120 seconds by default, terminate its process
  group on timeout, and retry the identical command up to twice with fresh output files.
  Record timeouts and fallback diagnostics, and clean up workers on completion/failure.
- Limit PoseBusters energy calculations to one thread without changing its thresholds
  or conformer seed. Preserve unknown results and explain missing checks in CSV notes.
- Save ligand-preparation provenance and serialization checks in the output details.

The CORE definition and use of all poses are unchanged. Results can differ from 0.5.1
where coordinate-inferred connectivity was incorrect. Unverifiable chemistry now
produces an unresolved result rather than a score based on an unverified graph.
