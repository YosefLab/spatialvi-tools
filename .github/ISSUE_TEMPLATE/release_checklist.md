---
name: Release checklist
about: Checklist for developers
title: ""
labels: releases
assignees: ""
---

- [ ] Bump version in `pyproject.toml`
- [ ] If patch release, backport version bump PR into the appropriate branch. Else, create a new
    branch off `main` with the appropriate rules
- [ ] Run the tutorials and verify they complete without errors
- [ ] Run the release workflow
- [ ] Check that the version updates correctly on [PyPI](https://pypi.org/project/spatialvi-tools/)
- [ ] (Optional) Post threads on Discourse and Twitter
