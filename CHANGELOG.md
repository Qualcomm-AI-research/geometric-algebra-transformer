# Changelog

## [1.4.2] - 2024-08-27

### Fixed

- Fix bug in `gatr.baselines.transformer` where the `normalized_shape` parameter incorrectly normalized over all dimensions except the first. Thanks to [@spinjo](https://github.com/spinjo) for identifying and providing the fix

## [1.4.1] - 2024-08-22

### Fixed

- Replace legacy call to deprecated `torch.LongTensor` with `torch.tensor`. Thanks to [@Ruibin-Liu](https://github.com/Ruibin-Liu) for providing a fix

## [1.4.0] - 2024-08-01

### Changed

- Equivariance-breaking functions have been replaced with equivariant counterparts, breaking backwards compatibility with `ArteryGATrWrapper` models trained with old GATr versions
- New argument `checkpoint` in `GATr` constructor for more fine-grained control over checkpointing behaviour

### Added

- Add embeddings for rays in Plücker coordinates
- Add `nominal_flops_per_token` property to linear layers that counts FLOPs

### Deprecated

- Equivariance-breaking functions now raise warnings
- Argument `checkpoint_blocks` in `GATr` constructor is deprecated in favour of `checkpoint`

### Fixed

- Fix bug in `compile_equi_linear()` that made autodiff through compiled linear layers incorrect

## [1.3.0] - 2024-04-18

### Changed

- Experimental support for torch.compile

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
