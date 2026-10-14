# Issue tracker: GitHub

Issues and specs for Matt Pocock engineering skills live in GitHub Issues for
`GuF-veg/pypto-lib-agent`. Use the `gh` CLI. Pass
`--repo GuF-veg/pypto-lib-agent` to issue, PR, and label commands.

## Conventions

- Create: `gh issue create --title "..." --body-file <file>`.
- Read: `gh issue view <number> --json number,title,body,labels,comments`.
- List: `gh issue list --state open --json number,title,body,labels,comments`; filter with `--label` and `--state` as needed.
- Comment: `gh issue comment <number> --body-file <file>`.
- Edit body: `gh issue edit <number> --body-file <file>`.
- Apply or remove labels: `gh issue edit <number> --add-label "..."` or `--remove-label "..."`.
- Close: `gh issue close <number>`; post any explanation as a comment first.
- Parent/child: `gh issue create --parent <parent> ...` or `gh issue edit <parent> --add-sub-issue <child>`. If sub-issues are unavailable, add the child to a task list in the parent body and put `Part of #<parent>` at the top of the child body.

Write multiline bodies to a temporary Markdown file and pass `--body-file`.
For older CLI versions, use `gh api` for native sub-issues or dependencies.
REST paths use `repos/GuF-veg/pypto-lib-agent/...`.

## Pull requests as a triage surface

**PRs as a request surface: no.**

GitHub shares issue and PR numbers. Resolve a bare `#42` with
`gh pr view 42`, then fall back to `gh issue view 42`.

## Skill instructions

When a skill says "publish to the issue tracker", create a GitHub issue.
When it says "fetch the relevant ticket", read the issue with the fields above.

## Wayfinding operations

- Map: a single issue labelled `wayfinder:map`, holding Notes, Decisions-so-far, and Fog.
- Child ticket: link it to the map as a sub-issue, using the parent/child convention above. Label it `wayfinder:<type>`, where type is `research`, `prototype`, `grilling`, or `task`.
- Blocking: use native issue dependencies: `gh issue edit <child> --add-blocked-by <blocker>`. If unavailable, put `Blocked by: #<number>, #<number>` at the top of the child body. A ticket is unblocked when every blocker is closed.
- Frontier: list the map's open children in map order. Skip assigned children and those with open blockers; the first remaining child wins. Native `issue_dependencies_summary.blocked_by` counts open blockers.
- Claim: `gh issue edit <number> --add-assignee @me`, the session's first write.
- Resolve: comment with the answer, close the child, and append a context pointer with its link to the map's Decisions-so-far.
