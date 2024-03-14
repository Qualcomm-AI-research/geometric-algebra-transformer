# Changelog

## [1.2.2] - 2024-03-13

_Minor cleanup._

## [1.2.1] - 2024-03-13

### Changed

- Autocast normalization layers and attention features to fp32 in mixed-precision training

### Added

- Add `CrossAttention` layer

### Fixed

- Fix bug in attention layers that lead to crashes when using an xformers backend and not using multi-query attention

## [1.2.0] - 2024-02-19

### Added

- Expose `join_reference` kwarg in `GATr.forward()`, `AxialGATr.forward()`
- Add utility function `gatr.utils.compile_linear.compile_equi_linear_submodules()`

### Removed

- Remove option to *not* provide a position in `embed_oriented_plane()`, `embed_reflection()`

## [1.1.1] - 2024-01-10

_Minor cleanup._

## [1.1.0] - 2024-01-02

### Changed

- Change dimension collapse behaviour in `AxialGATr` and `BaselineAxialTransformer`

### Added

- Add hooks functionality and `register_hook()` method to `BaseExperiment`

### Fixed

- Fix critical bug in `embed_3d_object_two_vec()` that lead to wrong results when any tensor dimension was 3
- Fix various minor issues in `BaseExperiment`
- Improve logging in `BaseExperiment`

## [1.0.0] - 2023-10-19

_First release._
