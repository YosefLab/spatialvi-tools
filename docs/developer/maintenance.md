# Maintenance Guide

## Releases

We follow [Semantic Versioning](https://semver.org). Each release is tagged `MAJOR.MINOR.PATCH`.

### Creating a release

1. Update `CHANGELOG.md` — move items from `[Unreleased]` to a new versioned section.
2. Bump the version in `pyproject.toml` and `src/spatialvi/__init__.py`.
3. Open a PR, merge, then create a GitHub release tag.

## Updating dependencies

Dependency bounds are declared in `pyproject.toml`. When a new major version of a dependency
is released:

1. Test against the new version locally.
2. Update the lower bound if new features are required, or add an upper bound if breaking
   changes exist.
3. Update `CHANGELOG.md`.

## Pre-commit hooks

The project uses [pre-commit](https://pre-commit.com) hooks for code quality. To update hook
versions:

```bash
pre-commit autoupdate
git add .pre-commit-config.yaml
git commit -m "chore: update pre-commit hooks"
```
