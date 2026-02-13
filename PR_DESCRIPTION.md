# Terrain Visualizer and Procedural Primitive Suite

This PR introduces an interactive terrain visualizer and expands the procedural terrain suite with several new primitives.

## Features

### Interactive Terrain Visualizer
*   **Real-time Parameter Tuning**: Dynamic GUI sliders allow for real-time adjustment of terrain parameters (e.g. noise scale, resolution, difficulty) with instantaneous visual feedback.
*   **Robot Context**: Support for spawning various robot models (Go1, G1, Yam) in their default standing poses to provide scale and context.
*   **Overview Mode**: An "All Terrains" preset to tile and compare the entire terrain suite simultaneously.
*   **Live Statistics**: Real-time display of polygon counts.

### New Procedural Terrain Types
Extends the `mjlab` terrain library with many new primitives, including:
*   **Heightfields**: Perlin noise, uniform random noise, and wave-based surfaces.
*   **Primitive terrains**: Open stairs (ascending/descending), sloped pyramids, nested rings, radial beams, stepping stones, and random spread boxes.

## How to Test
1. Run the visualizer script: `uv run python src/mjlab/scripts/visualize_terrain.py`
2. Access the interface at `http://localhost:8080`
3. Use the dropdowns and sliders to explore the different terrains and robot models.
