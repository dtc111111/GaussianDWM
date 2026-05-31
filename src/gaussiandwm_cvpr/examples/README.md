# GaussianDWM CVPR Dummy Data

These files are synthetic contract examples. They are only intended for
annotation parsing, tensor-shape checks, and script smoke tests. They do not
represent the paper dataset, model quality, or metric reproduction.

The QA item shows the processed annotation fields required by the CVPR
similarity selector path, including a `[512]` CLIP text-feature shard.

The world item shows the processed annotation fields required by the CVPR world
generation path, including six pseudo RGB/depth frames and a packed target
latent memmap shaped `[N,2,4,56,96]`.
