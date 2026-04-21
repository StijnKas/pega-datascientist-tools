# pega_io — TODO

This directory holds one Markdown file per open follow-up item for
the unified IO layer (`python/pdstools/pega_io/`).

## File format

Filename: `<short-slug>.md`. Concise, aim for under one screen:

```markdown
# <one-line title>

**Priority:** P1 / P2 / P3
**Touches:** `<files or areas>`

<problem statement>

## Approach

<concrete proposal>
```

## Listing open items

```
ls docs/plans/pega-io/
```

## Marking done

`git rm docs/plans/pega-io/<slug>.md` in the PR that resolves it.
