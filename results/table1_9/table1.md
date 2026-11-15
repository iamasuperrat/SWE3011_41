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
| Ours                | GPT-4o mini | Few-Shot        |                 43.79 |               62.40 |                     102.88 |                    100 |
|                     |             | CoT             |                 42.96 |               61.88 |                     102.77 |                    100 |
|                     |             | RePE            |                 41.68 |               60.97 |                     102.64 |                    100 |
|                     | LLAMA3      | Few-Shot        |                     - |                   - |                          - |                      - |
|                     |             | CoT             |                     - |                   - |                          - |                      - |
|                     |             | RePE            |                     - |                   - |                          - |                      - |
