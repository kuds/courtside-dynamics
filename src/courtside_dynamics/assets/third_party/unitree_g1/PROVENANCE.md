# Unitree G1 asset provenance

Imported on 2026-07-11 from the `unitree_g1` subtree of
<https://github.com/google-deepmind/mujoco_menagerie> at commit
`71f066ad0be9cd271f7ed58c030243ef157af9f4`.

The imported model is `g1_mjx.xml`, the 29-DoF variant with simplified
collision proxies and tuned position-actuator defaults. The 35 STL files it
references, its verbatim `LICENSE`, and upstream README/changelog are retained.
Images, scene wrappers, the plain duplicate model, and dexterous-hand-only
assets were omitted because the packaged tennis adapter does not reference
them.

The Menagerie changelog records the source Unitree model as
`g1_29dof_rev_1_0` from Unitree repository commit
`c20ca8f1fe5e519474c6c8d10b1ce5c719dd7a65`. `SHA256SUMS` locks the exact
imported file set. Project changes are isolated in
`../../robots/unitree_g1_tennis.xml`; files in this directory are immutable
upstream copies.

The G1 hardware itself is proprietary. Redistribution of these simulation
assets under BSD-3-Clause does not make the physical robot open hardware and
does not imply Unitree endorsement.
