# MINIRANSAC
## Environment
IDE: JetBrains CLion
OS: macOS Sierra

## How to use
You can run the executable file as follow:

```
./ransac --filename sample_points.csv --threshold 0.1 --inliner 0.5
```

Where:
- filenams: is the "," separated text file containing list of points.
- threshold: the maximum distance to decide inliners.
- inliner: the fraction of inliners, must be 0 -> 1. If this value is negaive, Ransac will decide it for you.
