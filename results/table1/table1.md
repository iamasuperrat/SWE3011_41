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
| Ours                | GPT-4o mini | Few-Shot        |                 14.17 |               38.80 |                     100.44 |                    100 |
|                     |             | CoT             |                 14.31 |               38.95 |                     100.40 |                    100 |
|                     |             | RePE            |                 14.21 |               38.84 |                     100.44 |                    100 |
|                     | LLAMA3      | Few-Shot        |                     - |                   - |                          - |                      - |
|                     |             | CoT             |                     - |                   - |                          - |                      - |
|                     |             | RePE            |                     - |                   - |                          - |                      - |
