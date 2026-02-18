# SmartFit Orientation

A Cura plugin that automatically orients your model for optimal FDM printing **and** ensures it fits within your printer's build volume.

Based on [CuraOrientationPlugin](https://github.com/nallath/CuraOrientationPlugin) (LGPL-3.0). This fork adds build-volume-aware orientation and automatic spin optimisation.

## Installation

1. Download the [latest release](https://github.com/SighClock/SmartFitOrientation/releases) or clone this repo.
2. Copy the entire `SmartFitOrientation` folder (the one containing `plugin.json`, `__init__.py`, etc.) into Cura’s plugin directory:
   - **Windows:** `%APPDATA%\cura\5.x\plugins` (replace `5.x` with your Cura version, e.g. `5.6`)
   - **macOS:** `~/Library/Application Support/cura/5.x/plugins`
   - **Linux:** `~/.local/share/cura/5.x/plugins`
3. Restart Cura. The menu **Extensions → SmartFit Orientation** should appear.

You can also install via the [UltiMaker Cura Marketplace](https://marketplace.ultimaker.com/) if the plugin is published there.

**Compatibility:** Cura 3.5+ (SDK 5.0.0–8.0.0). Requires NumPy.

## What it does

- Calculates the best printable face orientation (minimising overhangs and support material)
- Checks every candidate orientation against your printer's actual build volume
- Sweeps in-plane rotation angles to find the spin that gives the most clearance on the build plate
- Drops the model onto the build plate and centres it automatically

## Usage

- **Extensions → SmartFit Orientation → Calculate fast optimal printing orientation** — quicker run, fewer candidates.
- **Calculate extended optimal printing orientation** — more thorough search (recommended).
- **Modify Settings** — enable “Automatically calculate the orientation for all loaded models” and choose whether to minimise support volume or supported area.

Select one or more models in the build plate, then run one of the orientation actions.

## Original research

More info on the underlying STL-Tweaker algorithm can be found [here](https://www.researchgate.net/publication/311765131_Tweaker_-_Auto_Rotation_Module_for_FDM_3D_Printing).

## Credits

- **[Christoph Schranz](https://github.com/ChristophSchranz)** — [STL-Tweaker / Tweaker-3](https://github.com/ChristophSchranz/Tweaker-3): auto-rotation algorithm and research; evolutionary-optimized parameters used in `MeshTweaker.py`.
- **Jaime van Kessel** — Original Cura orientation plugin and wrapper around STL-Tweaker (copyright in source files).
- **[nallath](https://github.com/nallath)** — [CuraOrientationPlugin](https://github.com/nallath/CuraOrientationPlugin): maintained Cura plugin this project is based on (forked from [iot-salzburg/STL-tweaker](https://github.com/iot-salzburg/STL-tweaker)).

## License

This plugin is licensed under the **GNU Lesser General Public License v3.0 (LGPL-3.0)**,
in compliance with the [upstream CuraOrientationPlugin](https://github.com/nallath/CuraOrientationPlugin).
See [LICENSE](LICENSE) for the full text.
