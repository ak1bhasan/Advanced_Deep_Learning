# LSTM Practice Problem Solutions

**1. What problem do traditional Recurrent Neural Networks face when learning long sequences, and how do Long Short-Term Memories solve it?**

A traditional RNN reads a sequence one step at a time and keeps a single running summary of what it has seen. At every step, that summary is largely rewritten using the new input. So when a sequence is long, details from the early steps get overwritten or watered down by the time the network reaches the end. Training makes this worse. The learning signal has to travel backward through every step, and over many steps it tends to shrink until it is almost zero. This is called the vanishing gradient problem. Because of it, the network struggles to learn long-term dependencies, meaning cases where something early in the sequence affects something much later.

An LSTM fixes this by adding a separate memory line, the cell state, that passes through time with only small, controlled changes. Gates decide what to erase, what to add, and what to show. Since memory is not rewritten from scratch at every step, important information can survive for a long time, and the learning signal can flow back along this line without fading as quickly. It doesn't solve the problem perfectly, but it makes long sequences far more learnable.

**2. Explain the difference between cell state and hidden state in an LSTM. Why are both needed?**

The cell state is the LSTM's long-term memory. It carries information forward across many time steps and is only changed gradually by the gates. The hidden state is the LSTM's output at the current step. It is a filtered view of the memory, showing only what is useful right now. It gets passed to the next layer or used for predictions, and it also goes back into the LSTM at the next step to help the gates make their decisions.

An easy way to picture it is a notebook and a spoken answer. The cell state is the notebook where you store everything you might need later. The hidden state is what you actually say out loud at a given moment, based on what you read from the notebook.

Both are needed because remembering and reporting are two different jobs. Some information should be stored for a long time even though it isn't relevant at the current step. For example, the subject of a sentence may need to be remembered until the verb arrives many words later. Having two separate states lets the LSTM keep that information safe in memory without forcing it into the output before it's needed.

**3. What is the role of the Forget Gate in an LSTM? What problems could happen if it did not exist?**

The forget gate decides what should be erased from the cell state. It looks at the current input and the previous hidden state and, for each piece of stored memory, decides how much to keep and how much to throw away. For example, when a sentence ends and a new topic begins, details from the old topic may no longer matter, and the forget gate can clear them out.

This matters because memory is limited and needs to stay useful. Without a forget gate, the cell state could only keep collecting information. Old and irrelevant details would pile up and mix with new, important ones, which makes the memory noisy and hard to use. The network would also have no way to reset when the context changes, so stale information would keep influencing later predictions. The stored values could also keep growing over time, which makes learning unstable. The forget gate keeps the memory clean and focused.

**4. How does the Input Gate help an LSTM decide what new information should be stored in memory?**

Storing new information in an LSTM happens in two parts. First, the network proposes some candidate new content based on the current input and the previous hidden state. Then the input gate decides how much of that candidate content is actually worth writing into the cell state.

You can think of the input gate as a filter for incoming information. If the current input contains something important, like a key word or a meaningful event, the gate lets a large part of it through. If the input is mostly noise, the gate lets very little through.

This is needed because not everything the network sees deserves to be remembered. Without selective writing, the memory would fill up with unimportant details. The forget gate and the input gate work as a pair: the forget gate decides what to remove from the old memory, and the input gate decides what to add. Together they update the cell state at each step.

**5. What is the purpose of the Output Gate and how does it help create the hidden state?**

The output gate controls how much of the cell state is shown to the outside at the current step. The cell state may hold a lot of information, but most of it isn't needed at every moment. The output gate picks out the parts that are relevant right now.

To create the hidden state, the LSTM first takes the current cell state and adjusts its values so they stay in a manageable range. The output gate then filters this result, letting the useful parts through and holding back the rest. What comes out of this filtering is the hidden state.

This is important because it keeps the long-term memory separate from what the network reports. The LSTM can store something in the cell state and keep it hidden until it becomes relevant, instead of letting every stored detail affect the output all the time. The hidden state is then used for the prediction at this step and also passed on to help the gates at the next step.
