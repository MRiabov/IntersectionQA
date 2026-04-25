Task for this night, apr 25:
I want to fine-tune qwen-3.5-4b using reinforcement learning (GRPO) on both IntersectionQA and IntersectionEdit this night.

Warning: I'm sleeping, so you are on your own. I will send you automated reminders to continue, but you need to continue on your own anyway.

Important: the current sft pipeline with "only one answer" doesn't exactly fit for GRPO. GRPO needs reasoning tokens, so we need to define prompts an architecture for this to happen. I expect we need at least 2k max new tokens? In sft we had `batch_size = 32` I think, but I'm not sure how much will fit into H100/A100

We have a set of tasks today:
1. Implement the epic 15. Basically, make the dataset publishable. In fact, it would be great if you publish it on huggingface this night, as I have already done with IntersectionQA. Make splits for the IntersectionEdit.
    - We had a bug (bad logic) during intersectionQA implementation: we had imbalanced splits of datasets during training which led to poor SFT performance - because some class was hitting only with 10% chance, it fit very poorly to that class.
    - As such, ensure balanced datasets.
2. An important task: we don't want label contamination with SFT or RL. As such, split the train *internally* into RL and SFT. it's not split on dataset release level, rather we simply split it as consumers of the dataset. 
3. Do all the architecture for GRPO to happen. I think we already have it.
4. The most important part of the night: we want to test train the model on GRPO. So I want you to try and beat 
    - Note that "beat" may be fairly ill-defined here. Because in sft pipeline we had tall results with predicting with "max_output_tokens=16", which meant the pipeline was very cheap.
    - With that, that SFT pipeline would lose generality of the model. And to solve it, we either need to SFT on reasoning traces or we simply need to RL train the model. And I think the first step is RL unless you want to generate a few dozens of SFT reasoning traces, which probably will degrade it anyway. On the other hand, RL will be costly - we need much more compute to make RL work. So I'm split between it. But I will need to do RL in some other pipeline (and also, I am creating intersectionQA and intersectionEdit specifically to improve that pipeline; that pipeline is CAD-using-LLM pipeline, and we would need to RL that anyway), so I'm keen to get my hands dirty with custom RL (I never did it before.), although I acknowledge that SFT may solve the problem cheaper/faster.
5. Also, we didn't have a proper evaluation wired into the training loop(s). We trained blindly. Add evals every so often so that we can track it.
6. At the end of the night, whatever the train artifacts, upload them to hf bucket. https://huggingface.co/docs/huggingface_hub/guides/buckets.md. Also update the qwen3p5-4b-tuning experiment.
7. Make sure that the IntersectionEdit is well-covered in the document.

(I'm not quite sure if I should do any SFT going forward. It may prove results, and in fact it did in @docs/experiments/qwen-3p5-4b-tuning.md, however I don't know...)
Also, I suspect that we have dropped all unsloth training in SFT pipeline. Unsloth is 2x faster and results could be supreme!
Also, GRPO isn't the best, there is also Dr. GRPO, there is also GSPO... I'm not sure what these do, but I know they are improvements. Check out Unsloth docs, it implements both.

Also, commit your changes every once in a while.
