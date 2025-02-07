# ✋ How to Contribute

## Testing

To run the tests in verbose mode, execute:

```bash
ctest -V
```

## Code Formatting

Make sure that `clang-format` is installed. To format all `*.cc` files in the `application` folder, navigate to the root directory of the project and execute:

```bash
clang-format -i application/*.cc
```

## Making Changes and Pushing to the Repository

### Get the latest `master` branch
```bash
git fetch origin
git rebase origin/master
```

### Create a new branch for your changes

```bash
git checkout -b "my_branch"
```

Make your changes accordingly. Once you are finished and ready to commit, follow these steps:

1. Format the files according to the **Code Formatting** section.
2. Verify your changes using:

```bash
git diff
git status
```

3. Add modified files:
- To add all tracked files that have changed:
```bash
git add -u
```
- To add a new or specific file (e.g., `new.cc`):
```bash
git add new.cc
```

4. Check the staged files:

```bash
git status
```

5. Commit and push your changes:

```bash
git commit -m "Add a useful description"
git push origin my_branch
```

6. Open a pull request on GitHub.

## Rebasing Your Changes to the Latest `main` Branch

If you are working on a branch `my_branch` and need to rebase onto the latest `main`, follow these steps.

**WARNING:** If you are an inexperienced Git user, make sure to create a backup before proceeding.

1. Ensure you are on the correct branch:

```bash
git branch
```

2. Commit your local changes:

```bash
git add -u
git commit -m "my changes"
```

3. Fetch the latest changes from the remote repository:

```bash
git fetch origin
```

4. Start the rebase process:

```bash
git rebase -i origin/main
```

If there are no merge conflicts, force-push the rebased commit history:

```bash
git push -f origin my_branch
```

### Handling Merge Conflicts

If there are merge conflicts in a file (e.g., `my_file_with_merge_conflicts.hpp`), resolve them by manually editing the file. Remove conflict markers (`<<<<<<< HEAD` and `>>>>>>>`) and ensure the code is correct.

Once resolved, continue rebasing:

```bash
git add my_file_with_merge_conflicts.hpp
git rebase --continue
```

Repeat this process until the rebase completes successfully. Finally, force-push your changes:

```bash
git push -f origin my_branch
```

## Squashing Commits into One

If you want to squash multiple commits into a single commit on `my_branch`, follow the steps outlined in [this guide](https://www.redswitches.com/blog/squash-commits/#step-1-switch-to-the-branch).

**WARNING:** If you are an inexperienced Git user, make sure to create a backup before proceeding.
