# Changelog

All notable changes to SmartFit Orientation are documented here.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/).

## [1.0.1] - 2026-02-19

### Added

- **Build-volume-aware orientation** — Candidate orientations are checked against the printer’s build volume (width × depth × height). Only orientations that fit are preferred; if none fit, the best printability orientation is used with a warning.
- **In-plane spin optimisation** — After choosing the best orientation, the plugin sweeps rotation angles on the build plate (0.5° steps) to find the angle that fits the plate with the most clearance.
- **Drop to build plate** — Model is translated so its bottom sits on the build plate (Y = 0).
- **Centre on build plate** — Model is centred on the build plate using Cura’s build volume (works for both centred and corner-origin printers).
- **User setting: Fast vs precise fit check** — In **Extensions → SmartFit Orientation → Modify Settings**, option “Use fast fit check” (checked by default): fast = 5° steps + early exit; unchecked = precise 0.5° full sweep. Help “?” tooltip explains the option.
- **NOTICE and license headers** — LGPL-3.0 compliance: NOTICE file, LICENSE, and source file headers with attribution (Jaime van Kessel, Christoph Schranz, nallath, SighClock).

### Changed

- Version numbering reset to 1.0.x for this fork. Based on CuraOrientationPlugin (nallath) and STL-Tweaker (Christoph Schranz), with SmartFit extensions by SighClock.

---

## [1.0.0] - 2026-02-19

- Initial SmartFit Orientation release (fork of CuraOrientationPlugin with build-volume fit, spin search, drop-to-plate, and centring).
