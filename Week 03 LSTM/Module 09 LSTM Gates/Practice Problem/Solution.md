# LSTM Assignment

**Notation used throughout:** $x_t$ is the input at time step $t$, $h_{t-1}$ is the previous hidden state, $C_{t-1}$ is the previous cell state, and $*$ means element-wise multiplication (each number in one vector is multiplied by the number at the same position in the other vector).

**Cell state vs. hidden state (important for every answer below):**

- $C_t$ (cell state) is the long-term memory. It runs through the LSTM from one time step to the next and is changed only a little at each step.
- $h_t$ (hidden state) is the output of the cell at this time step. It is passed to the next time step and also to any layer above the LSTM.
- They are **not the same thing**. $h_t$ is made from $C_t$, but only the part that the Output Gate lets through.

---

## 1. Forget Gate

**Purpose:** The Forget Gate decides which information from the previous cell state should be kept and which should be removed. This is needed because an LSTM reads a long sequence, and not everything it saw earlier stays useful. If the cell never cleared old information, the memory would fill up with stuff that no longer matters.

**Formula:**

$$
f_t = \sigma(W_f[h_{t-1}, x_t] + b_f)
$$

| Symbol | Meaning |
|---|---|
| $f_t$ | Output of the Forget Gate at time $t$. It is a vector with one value per memory cell, each between 0 and 1. |
| $\sigma$ | Sigmoid function. Squashes any number into the range 0 to 1. |
| $W_f$ | Weight matrix of the Forget Gate. It is learned during training. |
| $h_{t-1}$ | Hidden state from the previous time step (what the cell output last time). |
| $x_t$ | Input at the current time step. |
| $[h_{t-1}, x_t]$ | The two vectors joined (concatenated) into one longer vector, so the gate can look at both together. |
| $b_f$ | Bias term of the Forget Gate. It is also learned. |

**What the sigmoid does:** The sigmoid is $\sigma(z) = \frac{1}{1 + e^{-z}}$. Its output is always between 0 and 1. That makes it a good fit for a gate, because the value can be read as "how much should pass through":

- close to 1 → keep almost all of this memory
- close to 0 → forget almost all of this memory
- in between → keep part of it

**How it works:** The gate takes the previous hidden state and the current input, joins them, multiplies by $W_f$, adds $b_f$, and applies the sigmoid. So $f_t$ depends on what the cell just did and on what it is seeing right now. Later, $f_t$ is multiplied with $C_{t-1}$ to decide how much of the old memory survives.

**Small example:** Suppose the cell is remembering that the subject of a sentence is "Ali". When a new sentence starts about "Sara", the Forget Gate can output a value near 0 for the memory cell storing "Ali" so that it gets cleared out.

---

## 2. Input Gate

**Role:** The Input Gate controls how much new information gets written into the cell state. It has two parts that work together: the input gate value $i_t$ and the candidate memory $\tilde{C}_t$.

**Formulas:**

$$
i_t = \sigma(W_i[h_{t-1}, x_t] + b_i)
$$

$$
\tilde{C}_t = \tanh(W_c[h_{t-1}, x_t] + b_c)
$$

| Symbol | Meaning |
|---|---|
| $i_t$ | Input gate value, between 0 and 1 for each memory cell. |
| $\tilde{C}_t$ | Candidate cell state (candidate memory): the new information that *could* be added. |
| $\sigma$ | Sigmoid, output between 0 and 1. |
| $\tanh$ | Hyperbolic tangent, output between -1 and 1. |
| $W_i$, $W_c$ | Weight matrices for the input gate and the candidate memory. |
| $b_i$, $b_c$ | Bias terms for the input gate and the candidate memory. |
| $h_{t-1}$, $x_t$ | Previous hidden state and current input. |
| $[h_{t-1}, x_t]$ | The two joined into one vector. |

**The key difference:**

- $\tilde{C}_t$ is **what** new information could be added. It is a proposal built from the current input and the previous hidden state.
- $i_t$ is **how much** of that proposal is actually allowed to be written into memory.

So the candidate memory is the content, and the input gate is the filter on that content.

**About sigmoid:** Same idea as in the Forget Gate. Since $i_t$ is between 0 and 1, it works as a gate. Near 0 means "ignore this new information" and near 1 means "write it in fully".

**About tanh:** $\tanh$ squashes values into the range -1 to 1. It is used for the candidate memory because the new information can be positive or negative. A value of -1 to 1 lets the cell push a memory value up or down, while still keeping the numbers from becoming too large. (A sigmoid here would only give 0 to 1 and could only push values up.)

---

## 3. Cell State Update

**Formula:**

$$
C_t = f_t * C_{t-1} + i_t * \tilde{C}_t
$$

The formula has two parts.

**Part 1: $f_t * C_{t-1}$ (retained old memory).** The old cell state is multiplied by the Forget Gate. Where $f_t$ is near 1, the old memory is kept. Where $f_t$ is near 0, it is erased. What is left is the old memory that survived.

**Part 2: $i_t * \tilde{C}_t$ (selected new memory).** The candidate memory is multiplied by the Input Gate. Where $i_t$ is near 1, the new information is written. Where $i_t$ is near 0, it is ignored. What is left is the new memory that was chosen.

**Combining them:** The two parts are added. The result $C_t$ is the new cell state: some of the old memory plus some of the new memory.

**Intuition:** Think of a notebook. First you erase some old notes (Forget Gate). Then you write some new notes (Input Gate). What is on the page afterwards is $C_t$.

One more point: because the update is an addition rather than a full replacement, the old memory can pass through many time steps almost unchanged when $f_t$ stays close to 1. This helps the LSTM remember things over long sequences.

---

## 4. Output Gate

**Purpose:** The cell state $C_t$ holds the long-term memory, but the cell does not necessarily want to show all of it at every step. The Output Gate decides how much of the cell state is exposed as the hidden state $h_t$.

**Formulas:**

$$
o_t = \sigma(W_o[h_{t-1}, x_t] + b_o)
$$

$$
h_t = o_t * \tanh(C_t)
$$

| Symbol | Meaning |
|---|---|
| $o_t$ | Output gate value, between 0 and 1 for each memory cell. |
| $\sigma$ | Sigmoid, output between 0 and 1. |
| $W_o$ | Weight matrix of the Output Gate. |
| $b_o$ | Bias term of the Output Gate. |
| $h_{t-1}$, $x_t$ | Previous hidden state and current input. |
| $[h_{t-1}, x_t]$ | The two joined into one vector. |
| $C_t$ | The updated cell state (long-term memory). |
| $\tanh(C_t)$ | The cell state squashed into the range -1 to 1. |
| $h_t$ | The new hidden state (the output of the cell at time $t$). |

**Why $C_t$ goes through tanh:** The cell state is built by adding values over time, so it can grow larger than 1 or smaller than -1. Passing it through $\tanh$ brings it back into the range -1 to 1, so the hidden state stays at a reasonable scale. The values can still be negative or positive.

**Why multiply by $o_t$:** After $\tanh(C_t)$ we have a processed version of the memory. Multiplying by $o_t$ (0 to 1) then picks how much of each part to let out. If $o_t$ is near 0 for some position, that part of the memory stays hidden. If it is near 1, that part is shown in $h_t$.

**How $h_t$ is produced:**

1. Compute $o_t$ from $h_{t-1}$ and $x_t$.
2. Compute $\tanh(C_t)$.
3. Multiply them element by element to get $h_t$.

---

## 5. Why Three Gates Are Used

An LSTM has to do three different jobs at every time step: get rid of old information that is no longer useful, take in new useful information, and decide what to show to the rest of the network. One gate cannot do all three well, because these decisions are independent. For example, the cell may want to keep something in memory (not forget it) but not show it right now. Having a separate gate for each job gives the LSTM that flexibility.

The three gates do **different** things; they are not copies of each other.

| Gate | Responsibility |
|---|---|
| Forget Gate | Controls old memory: how much of $C_{t-1}$ is kept |
| Input Gate | Controls new memory: how much of the candidate $\tilde{C}_t$ is written |
| Output Gate | Controls what is exposed: how much of $\tanh(C_t)$ becomes $h_t$ |

**Forget Gate (old memory):**

$$
f_t = \sigma(W_f[h_{t-1}, x_t] + b_f)
$$

It is applied as $f_t * C_{t-1}$.

**Input Gate (new memory):**

$$
i_t = \sigma(W_i[h_{t-1}, x_t] + b_i), \qquad \tilde{C}_t = \tanh(W_c[h_{t-1}, x_t] + b_c)
$$

It is applied as $i_t * \tilde{C}_t$.

**Output Gate (what is exposed as hidden state):**

$$
o_t = \sigma(W_o[h_{t-1}, x_t] + b_o), \qquad h_t = o_t * \tanh(C_t)
$$

The first two gates work on the cell state itself (they decide what the memory contains), while the Output Gate works after the cell state is built (it decides how much of that memory is shown). Together they let the LSTM keep useful information for a long time and use it only when needed.

---

## 6. Comparison of the Three Gates

| Gate | Main Function | Main Input | Activation | Formula |
|---|---|---|---|---|
| Forget Gate | Decides how much of the old cell state $C_{t-1}$ to keep | $h_{t-1}$ and $x_t$ | Sigmoid ($\sigma$) | $f_t = \sigma(W_f[h_{t-1}, x_t] + b_f)$ |
| Input Gate | Decides how much of the candidate memory $\tilde{C}_t$ to write into the cell state | $h_{t-1}$ and $x_t$ | Sigmoid ($\sigma$) for $i_t$; tanh for $\tilde{C}_t$ | $i_t = \sigma(W_i[h_{t-1}, x_t] + b_i)$ and $\tilde{C}_t = \tanh(W_c[h_{t-1}, x_t] + b_c)$ |
| Output Gate | Decides how much of the cell state is exposed as the hidden state $h_t$ | $h_{t-1}$ and $x_t$ (then applied to $\tanh(C_t)$) | Sigmoid ($\sigma$) for $o_t$; tanh applied to $C_t$ | $o_t = \sigma(W_o[h_{t-1}, x_t] + b_o)$ and $h_t = o_t * \tanh(C_t)$ |

All three gates take the same inputs ($h_{t-1}$ and $x_t$), but each has its own weights and bias, so they learn to do different jobs. The gate values themselves ($f_t$, $i_t$, $o_t$) all use sigmoid. The difference is in **what each gate is multiplied with**: $C_{t-1}$, $\tilde{C}_t$, and $\tanh(C_t)$ respectively.

---

## 7. Controlling Old Memory vs. New Memory

**Old memory: $f_t * C_{t-1}$**

$C_{t-1}$ is everything the cell remembered up to the last step. The Forget Gate looks at the current situation and decides how much of each remembered item is still worth keeping.

- $f_t \approx 1$: keep it
- $f_t \approx 0$: drop it

Example: while reading a story, the cell remembers the name of the main character. As long as the story is still about that character, the forget value for that memory stays high. When the story moves on to someone else, the value drops and the old name fades out.

**New memory: $i_t * \tilde{C}_t$**

$\tilde{C}_t$ is the new information the cell could store based on the current input. The Input Gate decides how much of that is worth writing down.

- $i_t \approx 1$: store it
- $i_t \approx 0$: ignore it

Example: if the current word is something important (like a new character's name), the input value is high and the name gets stored. If the current word is just filler, the input value is low and nothing much is written.

**The difference:** The Forget Gate works on what is **already in memory** and can only reduce it. The Input Gate works on what is **arriving** and decides how much of it gets added. They are controlled separately, so the cell can, for instance, keep all the old memory and add nothing new, or clear the old memory and write something new in its place.

---

## 8. How the Output Gate Determines the Hidden State

The hidden state is calculated like this:

$$
o_t = \sigma(W_o[h_{t-1}, x_t] + b_o)
$$

$$
h_t = o_t * \tanh(C_t)
$$

The Output Gate does **not** create or change the cell state. By the time it is used, $C_t$ has already been built by the Forget and Input Gates. The Output Gate only looks at $C_t$ and controls how much of it is shown.

**Step by step:**

1. The cell state $C_t$ is already calculated.
2. $\tanh(C_t)$ squashes it into the range -1 to 1.
3. The Output Gate computes $o_t$ (values between 0 and 1) from $h_{t-1}$ and $x_t$.
4. $h_t = o_t * \tanh(C_t)$: for each position, the gate decides how much of the processed memory goes out.

**What this means:**

- If $o_t$ is close to 1 at some position, that part of the memory appears almost fully in $h_t$.
- If $o_t$ is close to 0, that part stays inside $C_t$ and is mostly hidden from $h_t$.

Because of this, $C_t$ can hold information that is not used right away, and $h_t$ only contains the part that is useful for the current output. So $h_t \neq C_t$. The cell state is the internal memory, and the hidden state is a filtered view of it.

---

## 9. Explaining the Statement

> "Forget Gate removes unnecessary information, Input Gate adds important information, and Output Gate controls the final output."

Each part of this statement matches one formula.

**Forget Gate removes unnecessary information**

$$
f_t * C_{t-1}
$$

The old cell state is multiplied by $f_t$ (0 to 1). Information with a small $f_t$ is shrunk toward zero, which means it is removed. Information with a large $f_t$ is kept.

**Input Gate adds important information**

$$
i_t * \tilde{C}_t
$$

The candidate memory is multiplied by $i_t$ (0 to 1). Only the important parts, where $i_t$ is large, are added to the memory.

**Cell state combines both**

$$
C_t = f_t * C_{t-1} + i_t * \tilde{C}_t
$$

The kept old memory and the selected new memory are added together to give the updated memory.

**Output Gate controls the final output**

$$
h_t = o_t * \tanh(C_t)
$$

The updated memory is passed through $\tanh$ and then multiplied by $o_t$, so only the needed part is released as the hidden state $h_t$.

**Putting it together:** the first two gates decide what goes into memory (remove some, add some), and the third gate decides how much of that memory is sent out. Note that "final output" here means the hidden state $h_t$, which goes to the next time step and to the next layer. It is not the same as the cell state $C_t$.

---

## 10. Complete Workflow of an LSTM Cell

At each time step, the LSTM cell goes through these steps.

**Step 1: Receive $x_t$ and $h_{t-1}$.**
The cell gets the current input $x_t$ and the previous hidden state $h_{t-1}$. It also has the previous cell state $C_{t-1}$. The first two are joined as $[h_{t-1}, x_t]$ and used by all the gates.

**Step 2: Forget Gate calculates $f_t$.**
$$
f_t = \sigma(W_f[h_{t-1}, x_t] + b_f)
$$
This tells the cell how much of each part of $C_{t-1}$ to keep.

**Step 3: Input Gate calculates $i_t$.**
$$
i_t = \sigma(W_i[h_{t-1}, x_t] + b_i)
$$
This tells the cell how much new information to allow in.

**Step 4: Candidate memory $\tilde{C}_t$ is calculated.**
$$
\tilde{C}_t = \tanh(W_c[h_{t-1}, x_t] + b_c)
$$
This is the new information that could be added.

**Step 5: Cell state $C_t$ is updated.**
$$
C_t = f_t * C_{t-1} + i_t * \tilde{C}_t
$$
Old memory is filtered by $f_t$, new memory is filtered by $i_t$, and the two are added.

**Step 6: Output Gate calculates $o_t$.**
$$
o_t = \sigma(W_o[h_{t-1}, x_t] + b_o)
$$
This decides how much of the cell state will be exposed.

**Step 7: Hidden state $h_t$ is calculated.**
$$
h_t = o_t * \tanh(C_t)
$$

**Step 8: Pass $C_t$ and $h_t$ to the next time step.**
$C_t$ carries the long-term memory forward. $h_t$ is used as $h_{t-1}$ in the next step and can also be sent to the next layer or used for the prediction.

**Flow diagram:**

```
        x_t   +   h_(t-1)        (and C_(t-1) from the previous step)
                  |
                  v
        Forget Gate:  f_t = σ(W_f [h_(t-1), x_t] + b_f)
                  |
                  v
        Input Gate:   i_t = σ(W_i [h_(t-1), x_t] + b_i)
        Candidate:    C~_t = tanh(W_c [h_(t-1), x_t] + b_c)
                  |
                  v
        Cell State Update:  C_t = f_t * C_(t-1) + i_t * C~_t
                  |
                  v
        Output Gate:  o_t = σ(W_o [h_(t-1), x_t] + b_o)
                  |
                  v
        Hidden State: h_t = o_t * tanh(C_t)
                  |
                  v
        C_t and h_t go to the next time step
```

In short, the LSTM cell first updates its memory (forget some, add some), and then decides how much of that memory to output.
