## [Main] - 2026/11/02

### Removed

- MAJOR Remove the laser model `analytical_temperature` and the parameter subsection `laser/analytical`. Instead, a simulation case may prescribe the temperature field by attaching a field function `"prescribed_temperature"` for `"heat_transfer"` in `mp-melt-pool`; the heat equation is then not solved.
```diff
{
  "laser": {
-    "model": "analytical_temperature",
-    "analytical": {
-      "ambient temperature": 500,
-      "max temperature": 3500,
-      "temperature x to y ratio": 3
-    }
  }
}
```
