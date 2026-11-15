| Methodology         | Base Model  | Prompt Approach | Operation Ratio ↓ (%) | Storage Ratio ↓ (%) | Relative Path Length ↓ (%) | Valid Path Ratio ↑ (%) |
| ------------------- | ----------- | --------------- | --------------------: | ------------------: | -------------------------: | ---------------------: |
| A*                  | -           | -               |                   100 |                 100 |                        100 |                    100 |
| Dynamic WA* (w = 2) | -           | -               |                     - |                   - |                          - |                      - |
| LLM                 | GPT-4o mini | Few-Shot        |                     - |                   - |                          - |                      - |
|                     |             | CoT             |                     - |                   - |                          - |                      - |
|                     |             | RePE            |                     - |                   - |                          - |                      - |
|                     | LLAMA3      | Few-Shot        |                     - |                   - |                          - |                      - |
|                     |             | CoT             |                     - |                   - |                          - |                      - |
|                     |             | RePE            |                     - |                   - |                          - |                      - |
| LLM-A*              | GPT-4o mini | Few-Shot        |                     - |                   - |                          - |                      - |
|                     |             | CoT             |                     - |                   - |                          - |                      - |
|                     |             | RePE            |                     - |                   - |                          - |                      - |
|                     | LLAMA3      | Few-Shot        |                     - |                   - |                          - |                      - |
|                     |             | CoT             |                     - |                   - |                          - |                      - |
|                     |             | RePE            |                     - |                   - |                          - |                      - |
| Ours                | GPT-4o mini | Few-Shot        |                 67.13 |               84.84 |                     101.51 |                    100 |
|                     |             | CoT             |                 64.45 |               83.15 |                     101.62 |                    100 |
|                     |             | RePE            |                 70.08 |               87.47 |                     101.28 |                    100 |
|                     | LLAMA3      | Few-Shot        |                     - |                   - |                          - |                      - |
|                     |             | CoT             |                     - |                   - |                          - |                      - |
|                     |             | RePE            |                     - |                   - |                          - |                      - |
