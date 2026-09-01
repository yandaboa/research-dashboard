# research-dashboard

Repo-agnostic ledger of research questions → commit links + experiments + assets, with stable
takeaways as a searchable knowledge base, wandb metric polling, and a local review GUI.

- **View**: `python3 ledger_serve.py` → http://127.0.0.1:8777 (question nav left, detail right,
  cropped live feed when nothing is selected).
- **Write** (agents): the `ledger_mcp.py` stdio MCP server (register in a repo's `.mcp.json`) —
  `ledger_guide`, `ledger_search`, `ledger_add_question`, `ledger_add_diff`, `ledger_add_experiment`,
  `ledger_add_asset`, `ledger_update`, `ledger_list`, `ledger_show`. CLI equivalent: `ledger.py -h`.
- **Metrics**: `fetch_metrics.py --sweep` pulls linked wandb metrics (stdlib GraphQL, `~/.netrc`
  auth) into `data/metrics/`; run it from cron every 5 minutes.
- **Data**: `data/{questions,diffs,experiments,patches,metrics}` — one JSON per entry, gitignored
  (`UWLAB_LEDGER_ROOT` overrides the location).

Used from [UWLab-patrick-private](https://github.com/yandaboa/UWLab-patrick-private) — see its
`CLAUDE.md` for the agent conventions (search first, commit before registering, telegraphic entries).
