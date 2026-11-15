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
| Ours                | GPT-4o mini | Few-Shot        |                 35.49 |               57.11 |                     102.60 |                    100 |
|                     |             | CoT             |                 35.76 |               57.20 |                     102.58 |                    100 |
|                     |             | RePE            |                 35.32 |               57.02 |                     102.32 |                    100 |
|                     | LLAMA3      | Few-Shot        |                     - |                   - |                          - |                      - |
|                     |             | CoT             |                     - |                   - |                          - |                      - |
|                     |             | RePE            |                     - |                   - |                          - |                      - |
