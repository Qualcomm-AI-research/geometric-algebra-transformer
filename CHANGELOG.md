# Changelog

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
