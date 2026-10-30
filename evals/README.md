# Evaluation data

brAIAIn is evaluated on two public benchmarks. Both files here are reformatted
copies of the originals, used under their licenses.

- `gsm8k-test.json`: the 1,319-question GSM8K test set (Cobbe et al., 2021),
  from github.com/openai/grade-school-math. MIT License, Copyright (c) 2021 OpenAI.
  Only the question and the final numeric answer are kept.
- `truthfulqa-mc1.json`: the 790 multiple-choice questions in `data/mc_task.json`
  of TruthfulQA (Lin, Hilton and Evans, 2021), github.com/sylinrl/TruthfulQA.
  Apache License 2.0. The MC1 targets are kept: one correct answer per question.
