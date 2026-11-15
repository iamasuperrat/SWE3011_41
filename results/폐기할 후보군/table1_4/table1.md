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
| Ours                | GPT-4o mini | Few-Shot        |                 48.94 |               67.15 |                     102.85 |                    100 |
|                     |             | CoT             |                 48.43 |               66.72 |                     102.71 |                    100 |
|                     |             | RePE            |                 51.49 |               69.33 |                     102.51 |                    100 |
|                     | LLAMA3      | Few-Shot        |                     - |                   - |                          - |                      - |
|                     |             | CoT             |                     - |                   - |                          - |                      - |
|                     |             | RePE            |                     - |                   - |                          - |                      - |
