# ✋ How to Contribute

## 🌟 Forking the repository

Before making any changes, you need to fork the repository to your own GitHub account. You can do this by clicking the Fork button at the top right corner of the repository page on GitHub. Once forked, clone your forked repository to your local machine:

```bash
git clone https://github.com/your-username/MeltPoolDG-dev.git
```
Then, navigate into the repository directory:
```bash
cd MeltPoolDG-dev
```
Add the original repository as an upstream remote to stay updated with changes:
```bash
git remote add upstream https://github.com/MeltPoolDG/MeltPoolDG-dev.git
```
Now, you are ready to start contributing!

## 🧪 Testing

To run tests in **verbose** mode, execute:

```bash
ctest -V
```

To run specific tests based on application names (e.g. `mp-advec-diff`) or regular expressions, use the `-R` flag:

```bash
ctest -R mp-advec-diff
```

---

## 🎨 Code Formatting

Ensure that `clang-format` and `autopep8` are installed. We use the `clang-format` version provided by **deal.II (v16)**, which can be installed via the following script:

🔗 [Download clang-format](https://github.com/dealii/dealii/blob/master/contrib/utilities/download_clang_format)

Once installed, format your code by running:

```bash
scripts/formatting/format-all
```

---

## 🔧 Installing Pre-commit Hooks

Ensure you have `pre-commit` installed. Then, install the hooks by executing the following command in the root directory of the repository:

```bash
pre-commit install
```

---

## 🚀 Making Changes and Pushing to the Repository

### 📥 Get the Latest `master` Branch

Before making changes, ensure your local branch is up to date:

```bash
git clone git@github.com:MeltPoolDG/MeltPoolDG-dev.git
git fetch origin
git rebase origin/master
```
###

### 🌿 Create a New Branch

```bash
git checkout -b "my_branch"
```

Make your changes. Once ready to commit, follow these steps:

1. Format your code (see **Code Formatting** section).
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
   git push my_remote my_branch
   ```
6. Open a **Pull Request** on GitHub.
7. Add labels to the PR.
   - Use the label <span style="background-color: #8BD909; padding: 4px;">ready to test</span> to trigger CI checks.
   - Use **lilac-colored labels** <span style="background-color: #D4C5F9; padding: 4px;">labels</span> to specify regular expressions for **faster CI runs**.

---

## 🔄 Rebasing Your Branch onto the Latest `master`

If you're working on `my_branch` and need to update it with the latest `master`, follow these steps.

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
   git rebase -i origin/master
   ```
5. If there are **no conflicts**, force-push the rebased branch:
   ```bash
   git push -f origin my_branch
   ```

### ⚔️ Handling Merge Conflicts

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

## 🛠 Squashing Commits into One

To squash multiple commits into a single commit:

🔗 [Follow this guide](https://www.redswitches.com/blog/squash-commits/#step-1-switch-to-the-branch).

⚠ **Warning:** Squashing rewrites commit history. If unsure, create a backup before proceeding.

---

Happy coding! 🎉

