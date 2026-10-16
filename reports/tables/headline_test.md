| Model | Precision@10% | Recall@10% | PR-AUC | ROC-AUC |
|---|---|---|---|---|
| Best baseline: 7-day mean OOS hours | 0.549 (0.543 to 0.555) | 0.159 (0.154 to 0.165) | 0.466 (0.462 to 0.471) | 0.636 (0.633 to 0.638) |
| Rule baseline: today's OOS hours (rule > 2h) | 0.576 (0.570 to 0.581) | 0.166 (0.162 to 0.169) | 0.470 (0.467 to 0.474) | 0.649 (0.647 to 0.651) |
| LightGBM, daily features | 0.607 (0.603 to 0.612) | 0.176 (0.172 to 0.180) | 0.532 (0.528 to 0.536) | 0.699 (0.696 to 0.701) |
| LightGBM, combined features | 0.615 (0.610 to 0.620) | 0.178 (0.174 to 0.182) | 0.538 (0.535 to 0.542) | 0.708 (0.705 to 0.710) |
| **Best model: LightGBM, full features** | 0.651 (0.646 to 0.656) | 0.189 (0.185 to 0.192) | 0.567 (0.563 to 0.571) | 0.729 (0.727 to 0.731) |
