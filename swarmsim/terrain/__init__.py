"""Module 1 - procedural terrain.

    from swarmsim.terrain import TerrainParams, generate_and_export, World
    world = generate_and_export(TerrainParams(seed=3), "worlds/demo")
    world = World.load("worlds/demo")
"""
from .generator import export_scene, generate_and_export, generate_world
from .params import TerrainParams
from .textures import Palette
from .world import Building, Road, Tree, World

__all__ = ["TerrainParams", "World", "Building", "Road", "Tree", "Palette",
           "generate_world", "export_scene", "generate_and_export"]
