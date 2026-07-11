# Third-party notices

## Unitree G1 MuJoCo model and meshes

- Component: Unitree G1 29-DoF MJCF and 35 referenced STL meshes
- Upstream: <https://github.com/google-deepmind/mujoco_menagerie/tree/main/unitree_g1>
- Pinned Menagerie commit: `71f066ad0be9cd271f7ed58c030243ef157af9f4`
- Original Unitree model recorded by Menagerie: `g1_29dof_rev_1_0`,
  Unitree repository commit `c20ca8f1fe5e519474c6c8d10b1ce5c719dd7a65`
- Copyright: 2016–2023 HangZhou YuShu TECHNOLOGY CO., LTD.
  (Unitree Robotics)
- License: BSD-3-Clause

The verbatim license, upstream documentation, provenance, and checksums are
under `third_party/unitree_g1/`. The upstream `g1_mjx.xml` and imported meshes
are unmodified. The project-authored `robots/unitree_g1_tennis.xml` derivative
adds tennis-specific collision masks, a right-wrist racket mount, self-contact
pairs, paths, and simulation settings.

Neither the Unitree name nor contributor names may be used to endorse or
promote derived products without prior written permission. Inclusion here does
not imply endorsement by Unitree Robotics or MuJoCo Menagerie contributors.
