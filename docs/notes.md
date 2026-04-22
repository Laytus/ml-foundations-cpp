# Notes

## Build the program

From the project root:

```bash
cmake -S . -B build
cmake --build build
```

This will:
- configure the project with CMake
- generate the build files inside `build/`
- compile the executable

## Run the program

After building, run:

```bash
./build/ml_app
```

## Rebuild after code changes

If CMake is already configured, you usually only need:

```bash
cmake --build build
```

Then run again:

```bash
./build/ml_app
```

## Test strategy used for now

For now, testing is being done through `app/main.cpp`.

This means the workflow is:
1. add a small manual test in `main.cpp`
2. rebuild the project
3. run `./build/ml_app`
4. inspect the output

This is enough for the current learning phase, before adding a dedicated test framework.

## Current examples tested

So far, the program has been manually tested for:
- vector operations
- dot product
- mean
- CSV loading
- train/test split
- linear regression prediction
- MSE loss