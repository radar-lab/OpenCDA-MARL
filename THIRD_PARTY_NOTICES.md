# Third-party notices

The code written for OpenCDA-MARL is licensed under the Apache License,
Version 2.0 ([LICENSE](LICENSE), [NOTICE](NOTICE)). This file lists what the
repository contains that is *not* covered by that licence, the paths each part
occupies and the terms it comes with. Where a path is listed here, the Apache
licence does not apply to it; the exceptions are the three mixed files at the
end of section 1, where it applies to the lines the OpenCDA-MARL authors wrote.
Everything not listed is OpenCDA-MARL's own work: the `opencda_marl` package (the bundled checkpoints, recordings,
configuration profiles and the intersection map included), `scripts/`, the
tests, and the documentation.

| Part | Where | Terms |
| --- | --- | --- |
| OpenCDA core (UCLA Mobility Lab) and files derived from it | section 1 | UCLA Academic Software License |
| CARLA code | section 2 | MIT License |
| CARLA map data and assets | section 3 | CC-BY, as the CARLA project states |
| CARLA Python API wheels | `dependencies/whl/` | MIT License |
| Ultralytics (optional, not included) | section 5 | AGPL-3.0, from its own source |
| Packages installed from their own sources | section 6 | their own licences |

## 1. OpenCDA (UCLA Mobility Lab)

Source: <https://github.com/ucla-mobility/OpenCDA>. This repository's history
contains the upstream history up to commit `72b17e7b` (2024-12-04), the last
upstream commit before the OpenCDA-MARL authors' own work began (`0f86f6c0`,
2025-09-15). The changes made to the vendored files are recorded in
[docs/opencda/marl_patches.md](docs/opencda/marl_patches.md).

The public `opencda-marl` wheel (`pixi run wheel --public`) omits every path listed
here; the full build, which includes them, stays in Git and is never uploaded to a
package index.

Paths under these terms:

- `opencda/`: the vendored core, with its assets and scenario scripts. The
  CARLA files inside it (section 2) keep their own MIT notices.
- `opencda.py`: the scenario launcher, extended with the MARL command-line
  options.
- `opencda_marl/_bundled/configs/opencda/`: the scenario configurations of
  OpenCDA's `config_yaml` folder, copied unchanged.
- `opencda_marl/assets/maps/IHP_map.net.xml` and
  `opencda_marl/assets/maps/2lane_freeway_simplified.net.xml`: SUMO networks
  converted from OpenCDA's OpenDRIVE maps.
- `test/opencda/`, except `test_customized_map_navigation.py`,
  `test_vehicle_manager_agent_selection.py` and `test_world_build_retry.py`
  (OpenCDA-MARL's own): OpenCDA's unit tests and their data.
- Mixed files, OpenCDA-MARL's own except for code adapted from OpenCDA:
  `opencda_marl/builtin/agents/vanilla_agent.py` (the method
  `white_list_match`, parts of `__init__`, `is_close_to_destination` and
  `update_information`), `opencda_marl/builtin/agents/marl_behavior_agent.py`
  (the method `set_destination`) and
  `opencda_marl/builtin/planning/debug_helper.py` (the function
  `draw_trajetory_points`). The rest of each file is OpenCDA-MARL's.

The terms of these files are those of UCLA's licence, reproduced here byte for
byte (the same text is kept beside the core as `opencda/LICENSE`):

```text
Academic Software License: © 2021 UCLA Mobility Lab (“Institution”).  Academic or nonprofit researchers are permitted to use this Software (as defined below) subject to Paragraphs 1-3:

Institution hereby grants to you free of charge, so long as you are an academic or nonprofit researcher, a nonexclusive license under Institution’s copyright ownership interest in this software and any derivative works made by you thereof (collectively, the “Software”) to use, copy, and make derivative works of the Software solely for educational or academic research purposes, in all cases subject to the terms of this Academic Software License. Except as granted herein, all rights are reserved by Institution, including the right to pursue patent protection of the Software.

Please note you are prohibited from further transferring the Software -- including any derivatives you make thereof -- to any person or entity. Failure by you to adhere to the requirements in Paragraphs 1 and 2 will result in immediate termination of the license granted to you pursuant to this Academic Software License effective as of the date you first used the Software.

IN NO EVENT SHALL INSTITUTION BE LIABLE TO ANY ENTITY OR PERSON FOR DIRECT, INDIRECT, SPECIAL, INCIDENTAL, OR CONSEQUENTIAL DAMAGES, INCLUDING LOST PROFITS, ARISING OUT OF THE USE OF THIS SOFTWARE, EVEN IF INSTITUTION HAS BEEN ADVISED OF THE POSSIBILITY OF SUCH DAMAGE. INSTITUTION SPECIFICALLY DISCLAIMS ANY AND ALL WARRANTIES, EXPRESS AND IMPLIED, INCLUDING, BUT NOT LIMITED TO, ANY IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE. THE SOFTWARE IS PROVIDED “AS IS.” INSTITUTION HAS NO OBLIGATION TO PROVIDE MAINTENANCE, SUPPORT, UPDATES, ENHANCEMENTS, OR MODIFICATIONS OF THIS SOFTWARE.

Commercial entities: please contact the UCLA Mobility Lab at jiaqima@ucla.edu for licensing opportunities.
```

The text grants academic or nonprofit researchers a licence to use, copy and
make derivative works of the software solely for educational or academic
research purposes, prohibits transferring the software, including derivatives,
to any person or entity, and directs commercial entities to contact the UCLA
Mobility Lab. Nothing in the Apache-2.0 licence of OpenCDA-MARL changes these
terms, and OpenCDA-MARL grants no right in the files above beyond what UCLA's
text grants. The headers inside these files are UCLA's own and are left as they
are: `License: TDG-Attribution-NonCommercial-NoDistrib` in most of them, and
`License: MIT` in eight files under `test/opencda/` (`mocked_carla.py` and seven
`test_*.py`). OpenCDA-MARL treats those eight as under UCLA's licence and does
not rely on the MIT header.

OpenCDA-MARL's code calls into this core: the CARLA backend needs it, and parts
of the SUMO benchmark import it as well. A running installation therefore uses
the OpenCDA code, and that use is subject to the terms above.

## 2. CARLA code (MIT)

These files carry CARLA's copyright notices (Computer Vision Center, Intel Labs)
and the MIT License:

- in `opencda/`:
  `co_simulation/sumo_integration/bridge_helper.py`,
  `co_simulation/sumo_integration/carla_simulation.py`,
  `co_simulation/sumo_integration/constants.py`,
  `co_simulation/sumo_integration/sumo_simulation.py`,
  `core/actuation/pid_controller.py`, `core/common/misc.py`,
  `core/plan/global_route_planner.py` and
  `core/plan/global_route_planner_dao.py`;
- in OpenCDA-MARL: `opencda_marl/builtin/agents/basic_agent.py` and
  `opencda_marl/builtin/planning/controller.py`, `local_planner.py` and
  `misc.py`.

Each file keeps its own notice (for example `Copyright (c) 2018-2020 CVC` and
`Copyright (c) 2018 Intel Labs`). Changes made to the four files in
`opencda_marl/` by the OpenCDA-MARL authors are under the same MIT terms.
`opencda/scenario_testing/utils/cosim_api.py` is modified from CARLA's
co-simulation code and is marked `License: MIT` by its author; it carries no
CARLA copyright line and is treated as MIT too. The
licence, as the CARLA project publishes it
(<https://github.com/carla-simulator/carla/blob/master/LICENSE>):

```text
MIT License

Copyright (c) 2024 Computer Vision Center (CVC) at the Universitat Autonoma de
Barcelona (UAB).

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```

## 3. CARLA map data and assets (CC-BY)

The CARLA project's README states that "CARLA specific assets are distributed
under the CC-BY License"; it names no version. The following derive from CARLA
0.9.16 map data and keep that attribution (CARLA team, Computer Vision Center
(CVC) at the Universitat Autonoma de Barcelona, <https://carla.org>):

- `opencda_marl/assets/maps/Town01.net.xml` to `Town07.net.xml`,
  `Town10HD.net.xml` and `OpenDriveMap.net.xml`: SUMO networks converted from
  CARLA's OpenDRIVE maps with `netconvert` by `scripts/convert_opendrive_maps.py`
  (changed from the originals);
- `opencda_marl/assets/maps/Town10HD.xodr`: copied from CARLA, as
  `opencda_marl/assets/maps/README.md` records (0.9.16 is the version that README
  names for the derived maps);
- in `opencda/assets/`, the `Town05` and `Town06` SUMO networks are also derived
  from CARLA town maps; they reach this repository through the OpenCDA core,
  whose history does not record the exact source.

Check the CARLA project's licence page for the version of CC-BY that applies
before reusing these files.

## 4. CARLA Python API wheels (MIT)

`dependencies/whl/` holds CARLA Python API wheels, kept so a clone can install
the CARLA client without a network. Their own metadata names the MIT License.
They are redistributed as obtained, except the CARLA 0.9.15 `cp312` Windows
wheel, which the OpenCDA-MARL authors built from CARLA's 0.9.15 source tag.
The Linux wheels bundle shared libraries (libpng, libtiff, libjpeg, libz,
liblzma, libjbig, libzstd and libwebp, depending on the wheel) that have their
own licences; the wheels carry no licence texts for them, so see each library's
own project.

## 5. Ultralytics (optional, not included)

The repository's pixi environment installs the Ultralytics package (AGPL-3.0)
from conda-forge. Perception with `--apply_ml` uses it, downloads the
`yolov8m.pt` weights on first use, and falls back to downloading the YOLOv5
source and weights through `torch.hub` when the package is missing, all under
Ultralytics' terms. None of this is part of this repository, of the
`opencda-marl` wheel or of the Apache-2.0 licence.

## 6. Packages installed from their own sources

PyTorch, NumPy, Eclipse SUMO, PettingZoo, Gymnasium and the other packages named
in `pyproject.toml` and `pixi.lock` are installed from PyPI or conda-forge under
their own licences and are not part of this repository. They carry terms of
their own: PyTorch's CUDA runtime libraries are under NVIDIA's terms, and the
SUMO project publishes Eclipse SUMO under EPL-2.0 or GPL-2.0-or-later.
