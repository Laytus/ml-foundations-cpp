# CMakeLists Notes

`CMakeLists.txt`:

```cmake
cmake_minimum_required(VERSION 3.16)

project(ml_foundations_cpp LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Debug)
endif()

add_compile_options(-Wall -Wextra -Werror)

add_library(ml_core
    src/linear_regression/linear_regression.cpp
    src/logistic_regression/logistic_regression.cpp
    src/knn/knn.cpp
    src/decision_tree/decision_tree.cpp
)

target_include_directories(ml_core PUBLIC include)

add_executable(ml_app
    app/main.cpp
)

target_link_libraries(ml_app PRIVATE ml_core)
```

---

At a high level, this file says:

1. this is a C++ project
2. use C++17
3. default to Debug builds
4. compile with warnings enabled
5. build a reusable library called `ml_core`
6. build an executable called `ml_app`
7. link the executable against the library

---

## Explanation line by line

### 1. Minimum CMake version

```cmake
cmake_minimum_required(VERSION 3.16)
```

> this project requires at least CMake 3.16


### 2. Project declaration

```cmake
project(ml_foundations_cpp LANGUAGES CXX)
```

> This is a C++ project

#### Why `CXX`?

In CMake:

- `C` means C
- `CXX` means C++

This activates the C++ toolchain.

This name becomes the identity of the build configuration and appears in IDE integrations and generated project metadata.


### 3. C++ standard

```cmake
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
```

#### `set(CMAKE_CXX_STANDARD 17)`

> Compile using C++17

#### `set(CMAKE_CXX_STANDARD_REQUIRED ON)`

> Do not silently downgrade to an older standard

Without this, a compiler might try to compile with an older version if it cannot do 17.

#### `set(CMAKE_CXX_EXTENSIONS OFF)`

> Use standard C++17, not compiler-specific extensions

For example, this avoids switching to things like:

- `gnu++17`


### 4. Default build type

```cmake
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Debug)
endif()
```

> If no build type is specified manually, default to **Debug**

#### Why "Debug" by default?

Because during learning and implementation, we want:

- easier debugging
- no aggressive optimization
- predictable behavior

#### Later

When needed, we can build release explicitly with:

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
```


### 5. Warning flags

```cmake
add_compile_options(-Wall -Wextra -Werror)
```

This adds three important compiler flags:

- `-Wall`

Enable common warnings.

- `-Wextra`

Enable additional strict warnings.

- `-Werror`

Treat warnings as errors.


### 6. Create the core library

```cmake
add_library(ml_core
    src/linear_regression/linear_regression.cpp
    src/logistic_regression/logistic_regression.cpp
    src/knn/knn.cpp
    src/decision_tree/decision_tree.cpp
)
```

> This creates a *build target* called **`ml_core`**

It is a library made from the model source files.

#### Why make a library?

Instead of putting everything directly into `main.cpp`, we separate:

- **core ML code** → library
- **program entry point** → executable


### 7. Include directories

```cmake
target_include_directories(ml_core PUBLIC include)
```

> This tells CMake: *headers can be found under `include/`*

So a file like:

```cpp
#include "ml/linear_regression/linear_regression.hpp"
```

will resolve correctly.

#### Why `PUBLIC`?

Because:

- the library itseld needs those headers
- anything that links against the library also needs them

In this case, `ml_app` links against `ml_core`, so `PUBLIC` is the right choice.


### 8. Create the executable

```cmake
add_executable(ml_app
    app/main.cpp
)
```

> This creates the runnable program: **`ml_app`**

using:

> `app/main.cpp`

#### Why separate app from library?

Because `main.cpp` should only be:

- orchestration
- CLI entrypoint
- experiment runner

while the real ML logic lives in the library.


### 9. Link executable to the library

```cmake
target_link_libraries(ml_app PRIVATE ml_core)
```

> This says: *`ml_app` depends on `ml_core`*

So when we build `ml_app`, it includes the code from the ML library.

#### Why `PRIVATE`?

Because `ml_app` uses `ml_core` internally, but that dependency does not need to propagate further.

---

## Mental model of the build

```txt
           +------------------+
           |      ml_app      |
           |    main.cpp      |
           +--------+---------+
                    |
                    v
           +------------------+
           |     ml_core      |
           | model code       |
           | LR / Logistic    |
           | kNN / Tree       |
           +------------------+
```

---

## Important decisions behind this CMake design

### Decision 1 – one core library

We chose **one shared internal library** (`ml_core`) instead of many little libraries.

#### Why?

Because at this stage:

- simpler
- easier to maintain
- less ceremony

### Decision 2 – keep `main.cpp` separate

This enforces the habit that:

- logic belongs in modules
- `main` is just the entrypoint

### Decision 3 – C++17 strict warnings

This keeps the phase:

- modern
- stable
- disciplined

---

## How to build with this

Once this file is in place, run:

```bash
cmake -S . -B build
cmake --build build
```

And later run:

```bash
./build/ml_app
```

---

## One important warning

This `CMakeLists.txt` assumes that these files already exist and compile:

- `src/linear_regression/linear_regression.cpp`
- `src/logistic_regression/logistic_regression.cpp`
- `src/knn/knn.cpp`
- `src/decision_tree/decision_tree.cpp`

If some of those are currently empty or include missing headers, the build may fail.