# Babysitting a Small Language Model through One-Step Tree-of-Thoughts Knowledge Distillation

## Overview
Paper: https://www.zichenz.me/project/slm_tot/slm_tot.pdf

This repository contains the code and datasets used for the paper **"Babysitting a Small Language Model through One-Step Tree-of-Thoughts Knowledge Distillation"**. The project explores a novel approach to enhance the reasoning capabilities of Small Language Models (SLMs) using a simplified prompting method called One-Step Tree-of-Thoughts (ToT) and knowledge distillation from Large Language Models (LLMs).

## Methods and Results

The project addresses the limitations of SLMs in handling complex reasoning tasks by:
- Introducing the **One-Step Tree-of-Thoughts** prompting framework.
- Fine-tuning SLMs using a synthesized dataset derived from LLM-generated responses.
- Evaluating the performance on the **Game of 24** reasoning benchmark.

Key results:
- One-Step ToT significantly improves reasoning performance over Chain-of-Thought (CoT) prompting.
- The fine-tuned SLM achieves competitive accuracy with vastly more efficient resource utilization compared to LLMs.
