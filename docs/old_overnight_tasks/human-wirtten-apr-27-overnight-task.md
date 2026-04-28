# Overnight task Apr 27 to Apr 28
We have done these pieces of the pipeline:
- We have done 333 sample per task type (and we have 7 tasks, so its something liek 2.3K samples)
- Those samples are 8K reasoning length + answer

Now, we need to continue our rejection sampling pipeline:
We can do two things, and I suggest running a canary evaluation on both (quick stop and decide what's better):
1. Distill the reasoning traces into 256-token len traces, to capture the essense of the reasoning, which we will later SFT, and then RL on,
   - Hint: to construct a good prompt, the agent should think what is the closest object in the cadquery design. So out of all parts, which shape that is not removed is the closest, because obviously, this will be always the key to solution.(toy example: two parts: left and right. The right part consists of two spheres, the left and the right sphere.). 
   - To shorten the traces use openrouter deepseek v4 pro, the 0.78$/Mtok output or similar
2. Option number two: don't distill the traces, and sft directly on the 9k len traces.

The study aims to know the following: when training models, is reasoning length useful if have already SFT'd on reasoning traces.

Also, please manually monitor at least a number of the distilled "proofs" option in #1 because they can be valid or fallacious.

Also, please go onto huggingface qwen 3.5 docs and check in which order it expects the inputs to be served. I don't know. So you want to format it appropriately, not just as `<think></think><answer></answer>`. Use qwen-native models.

And yeah, I want you to make the whole experiment through the night. It's a small experiment but both will take a while. 

So yes, rejection sampling + GRPO for the win, that's what I can say.

Use the run-experiment agent skill or whatever.

Yeah, if you will have time by the morning yet, please run the further experiments on the GPU. I'll come back to you at 5:45 AM Dublin time.

Note: the #2 that I've mentioned is not documented. I just want to experiment, document it as an experiment

Oh yes, there is also a stopped, but downloaded Vast instance, use that.

<!--Also add execution log and a researcher summary of events.-->

## Morning corrections:
> To shorten the traces use openrouter deepseek v4 pro, the 0.78$/Mtok output or similar
This means using a *LLM* and not deterministic extraction

>Also, please manually monitor at least a number of the distilled "proofs" option in #1 because they can be valid or fallacious.
Same, it should've been done through LLM!

