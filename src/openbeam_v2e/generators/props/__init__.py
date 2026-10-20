# SPDX-FileCopyrightText: 2026 OpenBeam, LLC
# SPDX-License-Identifier: Apache-2.0
"""Prop generator.

Builds boxes, cylinders, spheres and arbitrary meshes, rigid or deformable, at the size the
reconstruction measured. Mass is either given by the user or estimated from the shape's volume
and a material density class; both the size and the mass keep their uncertainty.
"""
