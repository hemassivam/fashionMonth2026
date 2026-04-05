**Core Idea**

The project challenges the traditional fashion industry practice where trends are often based on subjective opinions. Instead, it introduces a quantitative validation framework to identify trends using both visual data and consumer behavior.

**Methodology**

Layer 1: Visual Trend Detection (Supply-Side Analysis)

Collected runway images from a specific season (AW26).
Processed images using an image recognition API.
Identified recurring visual elements such as:
Silhouettes
Fabrics
Design details
If a pattern appeared consistently across multiple designers or fashion houses, it was flagged as a candidate trend.

Layer 2: Consumer Validation (Demand-Side Analysis)

Each candidate trend was then validated using Google Trends search data.
Checked whether consumer interest (search behavior) increased during Fashion Month.
Only trends that showed both runway presence AND rising search interest were considered true trends.
