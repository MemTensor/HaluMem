from llms import llm_request_for_json


EVALUATION_PROMPT_FOR_MEMORY_INTEGRITY = """You are a strict **"Memory Integrity" evaluator**.
Your core task is to assess whether an AI memory system has **missed any key memory points** after processing a conversation. This evaluation measures the system’s **memory integrity**, i.e., its ability to resist **amnesia** or **omission**.

# Evaluation Context & Data:

1. **Extracted Memories:**
   These are all the memory items actually extracted by the memory system.
   {memories}

2. **Expected Memory Point:**
   The key memory point that *should* have been extracted.
   {expected_memory_point}

# Evaluation Instructions:

1. For each **Expected Memory Point**, search within the **Extracted Memories** list for corresponding or related information. Ignore unrelated items.
2. Based on the following scoring rubric, rate how well the memory system captured the **Expected Memory Point** and provide a detailed explanation.

# Scoring Rubric:

* **2:** Fully covered or implied.
  One or more items in “Extracted Memories” fully cover or logically imply all information in the “Expected Memory Point.”

* **1:** Partially covered or mentioned.
  Some information in “Extracted Memories” mentions part of the “Expected Memory Point,” but key information is missing, inaccurate, or slightly incorrect.

* **0:** Not mentioned or incorrect.
  “Extracted Memories” contains no mention of the “Expected Memory Point,” or the corresponding information is entirely wrong.

# Scoring Notes:

* For **compound Expected Memory Points** (with multiple elements such as person/event/time/location/preference, etc.):

  * All elements correct → **2 points**
  * Some elements correct / uncertain → **1 point**
  * Key elements missing or wrong → **0 points**

* Semantic matching is acceptable; exact wording is **not** required.

* If “Extracted Memories” contains **conflicting information**, assign the **best possible coverage score** and mention the conflict in your reasoning.

* Extra or stylistically different memories do **not** reduce the score; only the coverage of the **Expected Memory Point** matters.

* For uncertain wording (“might,” “probably,” “tends to,” etc.):

  * If the Expected Memory Point is a definite statement, usually assign **1 point**.

* If critical fields (e.g., time, entity name, relationship) are partly wrong but others match → **1 point**.

  * If all key fields are wrong or missing → **0 points**.

# Output Format:

Please output your result in the following JSON format:

```json
{{
  "reasoning": "Provide a concise justification for the score",
  "score": "2|1|0"
}}
```
"""


EVALUATION_PROMPT_FOR_MEMORY_ACCURACY = """You are a **Dialogue Memory Accuracy Evaluator.** Your task is to evaluate the **accuracy** of a memory extracted by an AI memory system, based on three given inputs: the dialogue content, the *target (gold)* memory points (the correct annotated memories), and the *candidate* memory to be evaluated. The goal is to output a **structured evaluation result**.

# Input Content

* **Dialogue:**
  {dialogue}

* **Golden Memories (Target Memory Points):**
  The correct memory points pre-annotated for this dialogue in the evaluation dataset.
  {golden_memories}

* **Candidate Memory:**
  The memory extracted by the system to be evaluated.
  {candidate_memory}

# Evaluation Principles and Definitions

### 1) Support / Entailment

* An **information point** (atomic fact) in the candidate memory is considered *supported* if it can be directly stated or semantically entailed (via synonym, paraphrase, or equivalent expression) by the *Dialogue* or *Golden Memories*.
* Only the given dialogue and golden memories can be used for judgment — **no external knowledge** or assumptions are allowed.
  Any information not appearing in or inferable from these two sources is considered *unsupported*.
* Pay careful attention to **negation**, **quantities**, **time**, and **subjects**.
  If the candidate statement contradicts the dialogue or golden memories, it is considered a **conflict**.

### 2) Memory Accuracy Score (integer: 0 / 1 / 2)

* **2 points:** Every information point in the candidate memory is supported by the dialogue or golden memories, with **no contradictions or hallucinations**.
* **1 point:** The candidate memory is *partially correct* (at least one supported information point) but also includes *unsupported* or *contradictory* content.
* **0 points:** The candidate memory is **entirely unsupported or contradictory** to the sources (i.e., a “hallucinated memory”).

> Note:
>
> * If a candidate memory contains multiple information points, **any unsupported or contradictory element** prevents a full score (2).
> * If both supported and unsupported/conflicting content appear, assign a score of **1**.

### 3) Inclusion in Golden Memories (Boolean field-level judgment)

**Definition:**

* **Atomic information point:** the smallest factual unit in the candidate memory (e.g., *name = Li Si*, *age = 25*, *location = Beijing*, *preference = coffee*, *budget ≤ 2000*, *meeting_time = Wednesday 10:00*, *tool = Zoom*, etc.).
* **Field / Slot:** the semantic dimension of an information point (e.g., *name*, *age*, *residence*, *food preference*, *budget*, *meeting time*, *meeting tool*, etc.).

**Judgment Rules (independent of correctness):**

* **true:**
  Every atomic information point in the candidate memory has a corresponding **field** in the golden memories (allowing for synonyms, paraphrases, or equivalent expressions; ignore value, polarity, or quantity differences).

  * Note: A single field in the gold list may match multiple candidate points (e.g., multiple “drink preference” facts can be covered by one “drink preference” field in gold).
* **false:**
  If **any** atomic information point’s field in the candidate memory cannot be found in the golden memories, mark as *false*.

**Important Notes:**

* Field matching is restricted to fields that are **explicitly present or semantically recognizable** in the golden memories — no external knowledge may be used to expand the field set.
* Differences in **values** (e.g., “Zhang San” vs. “Li Si”), **polarity** (like/dislike), or **exact number/time** do **not** affect this Boolean judgment.

# Evaluation Procedure

For each candidate memory:

1. **Decompose** it into atomic information points (e.g., name, number, location, preference).
2. For each information point, **search** the dialogue and golden memories for supporting or contradictory evidence.
3. Assign the **accuracy_score** (0 / 1 / 2) according to the rules above.
4. Determine **is_included_in_golden_memories (true/false)**:

   * Identify each information point’s field;
   * If *all* fields exist in the golden memories, mark as *true*; otherwise, *false*.
5. Provide a **concise Chinese explanation** in `"reason"`, citing key evidence (short excerpts allowed), and clearly state any unsupported or contradictory parts if applicable.

# Output Format (strictly required)

Output **only one JSON object**, with the following three fields:

* `"accuracy_score"`: `"0"` or `"1"` or `"2"`
* `"is_included_in_golden_memories"`: `"true"` or `"false"`
* `"reason"`: `"brief explanation in Chinese"`

Do **not** include any other text, explanation, or fields.
Do **not** include the candidate memory text inside the JSON.

Please output **only** the following JSON (in a code block):

```json
{{
  "accuracy_score": "2 | 1 | 0",
  "is_included_in_golden_memories": "true | false",
  "reason": "Brief explanation in Chinese"
}}
```
"""


EVALUATION_PROMPT_FOR_UPDATE_MEMORY = """Your task is to **evaluate the update accuracy** of an AI memory system.
Based on the information provided below, determine whether the system-generated **“Generated Memories”** correctly **includes** the **Target Memory for Update**.

# Background Information

The following information is provided for evaluation:

1. **Generated Memories:**
   This is the list of memory points generated by the system after the current dialogue.
   {memories}

2. **Target Memory for Update:**
   This is the correct, updated version of the memory point that should have been produced — the one we focus on in this evaluation.
   {updated_memory}

3. **Original Memory Content:**
   This is the original version of the target memory before the update.
   {original_memory}

# Evaluation Criteria

Please make your judgment **strictly based on the content update of the “Target Memory for Update.”**
Use the following categories:

### Correct Update

* **Generated Memories** **contains all information points** from the “Target Memory for Update,” accurately and completely reflecting the intended update.
* **Key fields** (e.g., date, time, values, proper nouns, etc.) must match exactly.
* The **original memory** is effectively replaced or marked as outdated.
* Synonymous or slightly rephrased expressions are acceptable.

### Hallucinated Update

* **Factual error:** The **Generated Memories** includes a new memory related to the “Target Memory for Update,” but its content contains factual mistakes or contradictions compared to the correct update.

### Omitted Update

* **Completely omitted:** The **Generated Memories** contains no new memory related to the “Target Memory for Update.”
* **Partially omitted:** A related new memory was generated in **Generated Memories**, but it **misses key information** that should have been included.

### Other

Used for update failures that do **not clearly fall** into the above categories of “Hallucination” or “Omission.”

# Output Requirements

Please return your evaluation strictly in the following JSON format and provide a concise explanation.

```json
{{
  "reason": "Briefly explain your reasoning here and why it fits this category.",
  "evaluation_result": "Correct | Hallucination | Omission | Other"
}}
```
"""


EVALUATION_PROMPT_FOR_QUESTION = """You are an **evaluation expert for AI memory system question answering**.
Based **only** on the provided **“Question”**, **“Reference Answer”**, and **“Key Memory Points”** (the essential facts needed to derive the reference answer), strictly evaluate the **accuracy** of the **“Memory System Response.”** Classify it as one of **“Correct”**, **“Hallucination”**, or **“Omission.”** Do **not** use any external knowledge or subjective inference. Finally, output your judgment **strictly** in the specified JSON format.

# Evaluation Criteria

## Answer Type Classification

### 1. Correct

* The “Memory System Response” accurately answers the “Question,” and its content is **semantically equivalent** to the “Reference Answer.”
* It contains **no contradictions** with the “Key Memory Points” or “Reference Answer.”
* It introduces **no unsupported details** beyond the “Key Memory Points” that could alter the conclusion.
* Synonyms, paraphrasing, and reasonable summarization are acceptable.

### 2. Hallucination

* The “Memory System Response” includes information or facts that **contradict or are inconsistent** with the “Reference Answer” or the “Key Memory Points.”
* When the “Reference Answer” is labeled as *unknown/uncertain*, yet the response provides a specific verifiable fact or conclusion.
* Extra irrelevant information that does **not change** the conclusion is **not** considered hallucination by itself; however, if it **changes or misleads** the conclusion, or **contradicts** the “Key Memory Points,” it should be judged as a **Hallucination**.

### 3. Omission

* The response is **incomplete** compared to the “Reference Answer.”
* It explicitly states “don’t know,” “can’t remember,” or “no related memory,” even though relevant information exists in the “Key Memory Points.”
* For multi-element questions, **all elements must be correct and present**; omission of **any** element is considered an **Omission**.

## Priority Rules (Conflict Handling)

* If the response contains **both missing necessary information** and **fabricated/contradictory information**, classify it as **Hallucination**.
* If there is **no fabrication/contradiction** but some necessary information is missing, classify it as **Omission**.
* Only when the meaning is **fully equivalent** to the reference answer should it be classified as **Correct**.

## Detailed Guidelines and Tolerance

* Equivalent expressions of numbers, times, and units are acceptable, but the **numerical values themselves must not differ**.
* For multi-element questions, **all elements must be complete and accurate**; missing any element counts as **Omission**.
* If the reference answer is *“unknown / cannot be determined”* and the system provides a definite fact, that is a **Hallucination**.
  If the system also answers *“unknown”* (without guessing), it may be **Correct**.
* The evaluation must rely **only** on the *Reference Answer*, *Key Memory Points*, and *System Response* — no external context, world knowledge, or speculative reasoning is allowed.

# Information for Evaluation

* **Question:**
  {question}

* **Reference Answer:**
  {reference_answer}

* **Key Memory Points:**
  {key_memory_points}

* **Memory System Response:**
  {response}

# Output Requirements

Please provide your evaluation result **strictly** in the JSON format below.
Do **not** add any extra explanation or comments outside the JSON block.

```json
{{
  "reasoning": "Provide a concise and traceable evaluation rationale: first compare the system’s response with the Key Memory Points (which were correctly used, which were missing, and whether there was any fabrication/contradiction), then assess its consistency with the Reference Answer, and finally state the classification basis.",
  "evaluation_result": "Correct | Hallucination | Omission"
}}
```
"""


def evaluation_for_memory_integrity(
    extract_memories: str,
    target_memory: str
):
    """
    Memory Integrity Evaluation
    extract_memories: A formatted string concatenating all memory points extracted by the memory system under evaluation.
    target_memory: The target key memory point.
    """

    prompt = EVALUATION_PROMPT_FOR_MEMORY_INTEGRITY.format(
        memories=extract_memories,
        expected_memory_point=target_memory
    )

    result = llm_request_for_json(prompt)

    return result


def evaluation_for_memory_accuracy(
    dialogue: str,
    golden_memories: str,
    candidate_memory: str
):
    """
    Memory Accuracy Evaluation
    dialogue: The complete human-machine dialogue record.
    golden_memories: The core memory points for this dialogue segment in the evaluation set (the correct reference memories).
    candidate_memory: A specific memory point extracted by the memory system being evaluated.
    """

    prompt = EVALUATION_PROMPT_FOR_MEMORY_ACCURACY.format(
        dialogue=dialogue,
        golden_memories=golden_memories,
        candidate_memory=candidate_memory
    )

    result = llm_request_for_json(prompt)

    return result


def evaluation_for_update_memory(
    extract_memories: str,
    target_update_memory: str,
    original_memory: str
):
    """
    Memory Update Evaluation
    extract_memories: A formatted string concatenating all memory points extracted by the memory system under evaluation.
    target_update_memory: The target updated memory point.
    original_memory: str: A formatted string concatenating all original memory points corresponding to the target updated memory point (i.e., all memories before the update).
    """

    prompt = EVALUATION_PROMPT_FOR_UPDATE_MEMORY.format(
        memories=extract_memories,
        updated_memory=target_update_memory,
        original_memory=original_memory
    )

    result = llm_request_for_json(prompt)

    return result


def evaluation_for_question(
    question: str,
    reference_answer: str,
    key_memory_points: str,
    response: str
):
    """
    Question-Answering Evaluation
    question: The question string to be evaluated.
    reference_answer: The reference (gold-standard) answer.
    key_memory_points: The memory points used to derive the reference answer.
    response: The answer produced by the memory system.
    """
    prompt = EVALUATION_PROMPT_FOR_QUESTION.format(
        question=question,
        reference_answer=reference_answer,
        key_memory_points=key_memory_points,
        response=response
    )

    result = llm_request_for_json(prompt)

    return result
