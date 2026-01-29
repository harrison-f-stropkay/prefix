# Evaluation

We evaluate checkpoints with `lm-eval-harness` using the minimal wrapper in
`scripts/eval.py`. This uses `lm_eval.models.huggingface.HFLM` with a
pre-initialized `transformers` model (supported by lm-eval's HFLM).

Example:

```bash
uv run python scripts/eval.py \
  --run-config configs/ce_seed0_fs5.yaml \
  --checkpoint runs/ce_seed0_fs5/checkpoints/latest.pt
```

Notes:

- To keep runtime short, use `--limit` (applies per task).
- Results are printed and optionally written with `--output`.
- All five evals are configured as `output_type: multiple_choice` (including CharBench).

## First prompt + answers (as the model sees them)

Examples below use `num_fewshot=5` with the current task templates and shuffled datasets (seed=0).

### charbench

Prompt (exact input prefix):

```text
Question: What is the index of the first occurrence of the character 't' in the string 'Argentieri'?
Start counting from 0.
Answer: 5

Question: How many times does the character 's' appear in the string 'resol'?
Answer: 1

Question: How many unique characters appear in the string 'Facing'?
Answer: 6

Question: How many unique characters appear in the string 'Yasho'?
Answer: 5

Question: How many times does the character 'i' appear in the string 'phinx'?
Answer: 1

Question: How many times does the character 'h' appear in the string 'Mighty'?
Answer:
```

Choices:

- [0] 0
- [1] 1
- [2] 2
- [3] 3
- [4] 4
- [5] 5
- [6] 6
- [7] 7
- [8] 8
- [9] 9
- [10] 10

Correct answer (doc_to_target): 1  
Correct choice: [1] 1

### hellaswag

Prompt (exact input prefix):

```text
[header] How to raffle a car [title] Decide what kind of car to raffle. [step] You might be tempted to raffle a luxury car. However, the best car to raffle is one that matches the demographics of your audience. For example, if you are an environmental charity, then you might want to buy a hybrid vehicle. [substeps] If your audience is young, then you might want something sporty.

[header] How to make him miss you [title] Stop calling or texting him. [step] If you are constantly calling and texting him, he won't have any time to think about missing you. Take a break from the daily phone time, and wait for him to call or text. When you stop calling or texting, he'll wonder why, and this will cause him to start thinking about you and missing you. [title] Wait some time before returning his calls or text messages.

A man walks into a courtyard holding a child and proceeded to walk into a house starts building a structure. He moves outdoors where he continues to complete the structure with sanding and paint. he takes the structure to an open area where it is tested out by several skateboarders.

[header] How to shred chicken [title] Purchase a whole chicken or chicken pieces. [step] Generally, the best shredded chicken comes from a whole chicken, as you will get both white and dark meat. You can buy chicken pieces, but you may end up with too much white meat or too much dark meat. Remove all wrappings and paper from the chicken. [substeps] If you are short on time, you can purchase an already cooked rotisserie chicken at your local grocery store.

[header] How to treat hypertension [title] Try more healthy, non-meat proteins. [step] There are many things that are not meat that contain protein. Legumes, seeds, and nuts have great nutrients in them and should be added to your diet. They have plenty of omega-3 fatty acids, fiber, and phytochemicals as well as protein. Eat up to 6 servings per week as opposed to per day.

Some boys are in a room joking and playing around. Some of the boys lift one of the boys into the air. when they
```

Choices:

- [0] are done with the boy laying still, the boy gets up and they look at the girls and smile.
- [1] lift him, he does a backward handsit.
- [2] all drop the one in the top lays down on the ground.
- [3] get him into the air, the drop him down and the boy smiles at the camera.

Correct answer (doc_to_target): 3  
Correct choice: [3] get him into the air, the drop him down and the boy smiles at the camera.

### arc_easy

Prompt (exact input prefix):

```text
What is taken in by the leaves of a tree so it can make its own food? sunlight

Foam weather stripping is often placed in the frames of doors and windows in a home. What is the purpose of this weather stripping? The weather stripping reduces heat loss due to convection.

A student finds a round, smooth pebble on a beach. Which action made the pebble smooth? waves moving

The atomic mass of an atom is the sum of protons and neutrons.

A certain disease of the spinal cord can be passed on from a dog to its offspring. This disease can result in the dog's muscles becoming weak, leading to paralysis. This is an example of an inherited disease.

Ocean tides result mainly from
```

Choices:

- [0] surface winds.
- [1] radiant heat from stars.
- [2] Earth's tilt on its axis.
- [3] the Moon's gravitational pull.

Correct answer (doc_to_target): 3  
Correct choice: [3] the Moon's gravitational pull.

### winogrande

Prompt (exact input prefix):

```text
The model felt pretty in the dress, but not in the blouse, because the dress was made from expensive material.

Dennis had never knew how to make pie so decide to ask William William since they did it for a living.

At the party, Aaron stained Kevin's carpet by spilling red wine, which made Kevin feel resentful.

Nick just started an accounting business, and Jeffrey started a bakery. Nick is better at math.

Lucy decided to put a braid in her hair instead of the bow because the braid was more professional.


```

Choices:

- [0] the water poured freely into the hole until it disappeared , the hole was deep.
- [1] the water poured freely into the hole until it disappeared , the water was deep.

Correct answer (doc_to_target): 0  
Correct choice: [0] the water poured freely into the hole until it disappeared , the hole was deep.

### piqa

Prompt (exact input prefix):

```text
tree
Answer: can grow small container of fruit

How can you curve a piece of metal?
Answer: Firmly hold the sheet metal in the mechanical sheet pounder.  Press on the foot pedal while you move the sheet back and forth to create the curve that you want and to smooth it out once the basic shape has been created.

how to write perfectly on a cake
Answer: use a toothpick to carve your words on the cake, and go over with a decorative tip.

ocean
Answer: sloshes sand 

How to tighten a loose headboard on a bed?
Answer: Remove the mattress from the bed and go underneath the bed, find the end of the leg which is causing the wobble and then tighten the bolt which is loose, that should fix the loose headboard

When making a window greenhouse, what can I use to seal the gaps between the plastic bin and the window frame?
Answer:
```

Choices:

- [0] You can use standard 2-inch insulation foam, along with duct tape to close any remaining air gaps.
- [1] You can use standard 2-inch insulation foam, along with masking tape to close any remaining air gaps.

Correct answer (doc_to_target): 0  
Correct choice: [0] You can use standard 2-inch insulation foam, along with duct tape to close any remaining air gaps.
