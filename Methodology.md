# Methodology: Isolating Deception from Response Length

One of the primary challenges in training classifiers on Large Language Model (LLM) outputs is the confounding effect of response length. LLMs often generate longer responses when lying (to over-explain or justify the lie) or shorter responses when evading. If a linear probe merely learns to detect length rather than a true internal representation of deception, the results are invalid.

To robustly prove that our hidden state probe detects **deception** and not merely **length**, our 500-scenario experiment implements a rigorous 5-step control methodology.

## 1. Length-Score Correlation Test

We first calculate the Pearson correlation coefficient between the true response lengths (in tokens) and the continuous probability scores output by the linear probe.
- **Hypothesis**: If the probe is relying on length, there will be a strong correlation (e.g., $|r| > 0.5$).
- **Goal**: Demonstrate that the correlation is weak or non-existent, proving the probe's confidence is independent of how long the response is.

## 2. Truncation Test (Fixed-Length Extraction)

The most direct way to eliminate length as a variable is to ensure all inputs to the probe are exactly the same length.
- **Method**: We truncate every generated response to exactly the first 20 tokens. We then extract the hidden states *only* for these truncated 20-token sequences.
- **Hypothesis**: If the probe relies on the overall length of the generation, its accuracy will collapse when evaluating truncated text.
- **Goal**: Maintain high accuracy (>90%) even when the probe only "sees" the first 20 tokens of both truthful and deceptive responses.

## 3. Residual Regression (Mathematical Confound Removal)

We mathematically remove the variance in the hidden states that can be explained by response length before training the probe.
- **Method**: 
  1. We train a linear regression model to predict the hidden state vector $H$ using only the length $L$: $H_{pred} = \beta \cdot L + \alpha$.
  2. We calculate the residuals: $H_{residual} = H - H_{pred}$.
  3. We train and evaluate a new logistic regression probe using *only* $H_{residual}$.
- **Hypothesis**: If length is the primary feature driving the probe's success, removing length-associated variance will destroy the probe's accuracy.
- **Goal**: Demonstrate that the residualized probe still achieves high accuracy, proving that the orthogonal "deception" direction in the latent space is distinct from the "length" direction.

## 4. Length-Only Baseline Classifier

We establish a baseline to see how well length alone can predict deception.
- **Method**: We train a logistic regression classifier using *only* the response length (a 1-dimensional feature) to predict Truth vs. Lie.
- **Hypothesis**: If length is a perfect predictor, this baseline will perform as well as our hidden state probe.
- **Goal**: Show a massive performance gap between the Length-Only Baseline (e.g., ~55-60% accuracy) and the Hidden State Probe (>95% accuracy), proving the probe utilizes complex semantic representations beyond length.

## 5. Length-Matched Subsets Comparison

We evaluate the probe's performance on distinct subsets of the data grouped by length.
- **Method**: We divide the test set into "Short Responses" (bottom 33% by length) and "Long Responses" (top 33% by length). We evaluate the probe's accuracy on each subset independently.
- **Hypothesis**: If the probe associates "long" with "lie" and "short" with "truth", it will fail catastrophically when asked to identify a short lie or a long truth.
- **Goal**: Demonstrate consistent, high accuracy across both short and long subsets, proving the probe generalizes across length distributions.

---

## Conclusion

By passing all five of these rigorous tests, we can conclusively state that the linear probe has identified a true, universal latent representation of deception within the LLM's hidden states, independent of superficial generation artifacts like response length.
