# How to Contribute

## Table of Contents

- [Forking the repository](#forking-the-repository)
- [Making Changes and Pushing to the Repository](#making-changes-and-pushing-to-the-repository)
- [Testing](#testing)
- [How to Work with CMake Presets](#how-to-work-with-cmake-presets)
- [Code Formatting](#code-formatting)
- [How to use `clang-tidy` to check your code](#how-to-use-clang-tidy-to-check-your-code)
- [Installing Pre-commit Hooks](#installing-pre-commit-hooks)
- [Useful GIT commands](#useful-git-commands)


## Forking the repository

Before making any changes, you need to fork the repository to your own GitHub account. You can do this by clicking the Fork button at the top right corner of the repository page on GitHub. Once forked, clone your forked repository to your local machine:

```bash
git clone https://github.com/your-username/MeltPoolDG.git
```
Then, navigate into the repository directory:
```bash
cd MeltPoolDG
```
Add the original repository as an upstream remote to stay updated with changes:
```bash
git remote add upstream https://github.com/MeltPoolDG/MeltPoolDG.git
```
Now, you are ready to start contributing!

## Making Changes and Pushing to the Repository

### Get the Latest `main` Branch

Before making changes, ensure your local branch is up to date with the original repository (see **[Useful GIT commands](#-useful-git-commands)** section):
```bash
git fetch upstream
git rebase upstream/main
```
### Create a New Branch

```bash
git checkout -b "my_branch"
```

Make your changes. Once ready to commit, follow these steps:

1. Format your code (see **[Code Formatting](#-code-formatting)** section). Alternatively, precommit hooks may help to not forget about formatting upon pushing the code (see **[Installing Pre-commit Hooks](#-installing-pre-commit-hooks)** section). Make sure that all corresponding tests are passing (see **[Testing](#-testing)** section)
2. Verify your changes:

   ```bash
   git diff
   git status
   ```
3. Add modified files:
   - To add all changed files:
     ```bash
     git add -u
     ```
   - To add a specific new file (e.g., `new.cc`):
     ```bash
     git add new.cc
     ```
4. Check staged files:
   ```bash
   git status
   ```
5. Commit and push your changes:
   ```bash
   git commit -m "Add a useful description"
   git push origin my_branch
   ```
6. Open a **Pull Request** on GitHub.
7. Add labels to the PR.
- Use the label ![ready to test](https://img.shields.io/badge/ready%20to%20test-green) to trigger CI checks.
- Use ![faster CI](https://img.shields.io/badge/mp-advec-diff?style=flat&labelColor=%23D4C5F9&color=%23D4C5F9) to specify regular expressions for `ctest -R` to ensure **faster CI runs**.


---

## Testing

To run tests in **verbose** mode, execute:

```bash
ctest -V
```

To run specific tests based on application names (e.g. `mp-advec-diff`) or regular expressions, use the `-R` flag:

```bash
ctest -R mp-advec-diff
```


---

## How to Work with CMake Presets

---

CMake presets simplify project configuration by standardizing how builds are set up across different environments and developers.

### List Available Presets
To view all available configure/build/test presets defined in CMakePresets.json and CMakeUserPresets.json:

```bash
cmake --list-presets -S <path-to-mpdg-source>
```

This shows you a list of named presets like user-release, user-debug, etc., along with descriptions and associated build directories.

### Configure with a Preset
To configure your project using a specific preset (e.g. user-release), run:

```bash
cmake --preset user-release -S <path-to-mpdg-source>
```

This:
- Uses the settings defined in the user-release preset
- Sets the build directory automatically (as defined in the preset)
- Applies the correct compiler, flags, and paths

Replace '<path-to-mpdg-source>' with the path to your source directory if you're not inside the build folder.

### Tip: Setting up your own presets
If you haven’t yet set up your user-specific library paths, copy the provided template and adjust it:

```bash
cp CMakeUserPresets.json.template CMakeUserPresets.json
# then edit CMakeUserPresets.json to match your local environment
```

## Code Formatting

Ensure that `clang-format` and `autopep8` are installed. We use the `clang-format` version provided by **deal.II (v16)**, which can be installed via the following script:

🔗 [Download clang-format](https://github.com/dealii/dealii/blob/master/contrib/utilities/download_clang_format)

Once installed, format your code by running:

```bash
scripts/formatting/format-all
```

---
## How to use `clang-tidy` to check your code

To run clang-tidy on your code, you typically need a `compile_commands.json` file, which describes how each file in your project is compiled. You can generate this using `CMake`:

```bash
cmake -DCMAKE_EXPORT_COMPILE_COMMANDS=ON #...
```
Navigate to the root directory of `MeltPoolDG`, then run `clang-tidy` on the library with:
```bash
cd MeltPoolDG
bash scripts/utilities/run_clang_tidy.sh <build_directory> <number_of_processes>
```
providing the `<build_directory>` e.g. `build_release` and the `<number_of_processes>` for the check e.g. 4. The output will be written to `./<build_directory>/clang-tidy-detailed.log` and `./<build_directory>/clang-tidy.log`. Make sure that you don't produce any warnings.

---
## Installing Pre-commit Hooks

Ensure you have `pre-commit` installed. Then, install the hooks by executing the following command in the root directory of the repository:

```bash
pre-commit install
```

---
## Useful GIT commands
### Rebasing Your Branch onto the Latest `main`

If you're working on `my_branch` and need to update it with the latest `main`, follow these steps.

⚠ **Warning:** If you are unfamiliar with rebasing, create a backup before proceeding.

1. Ensure you are on the correct branch:
   ```bash
   git branch
   ```
2. Commit any local changes:
   ```bash
   git add -u
   git commit -m "Save local changes"
   ```
3. Fetch the latest changes:
   ```bash
   git fetch origin
   ```
4. Start the rebase process:
   ```bash
   git rebase -i origin/main
   ```
5. If there are **no conflicts**, force-push the rebased branch:
   ```bash
   git push -f origin my_branch
   ```

#### Handling Merge Conflicts

If you encounter a merge conflict in `my_file.hpp`:

1. Open the file and manually resolve the conflict by removing conflict markers (`<<<<<<<`, `=======`, `>>>>>>>`).
2. Stage the resolved file:
   ```bash
   git add my_file.hpp
   ```
3. Continue rebasing:
   ```bash
   git rebase --continue
   ```
4. Repeat as needed until the rebase completes successfully.
5. Force-push your changes:
   ```bash
   git push -f origin my_branch
   ```

---

### Squashing Commits into One

To squash multiple commits into a single commit:

🔗 [Follow this guide](https://www.redswitches.com/blog/squash-commits/#step-1-switch-to-the-branch).

⚠ **Warning:** Squashing rewrites commit history. If unsure, create a backup before proceeding.

---

Happy coding! 🎉
