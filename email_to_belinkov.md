# Email Draft — To Prof. Yonatan Belinkov

**To:** belinkov@technion.ac.il  
**Subject:** Deception Detection via Linear Probing of LLM Hidden States — Research Discussion Request

---

Dear Prof. Belinkov,

I hope this message finds you well. I am writing to you because your foundational work on probing classifiers — particularly "Probing Classifiers: Promises, Shortcomings, and Advances" and the more recent "LLMs Know More Than They Show" (ICLR 2025) — has been deeply influential in shaping our current research direction.

We have been conducting experiments on detecting deception in Large Language Models through linear probing of hidden state activations. Specifically, we trained a logistic regression probe on layer 10 activations of Qwen2.5-3B-Instruct and achieved 93.7% accuracy in distinguishing between truthful and deceptive model-generated responses across 435 scenarios spanning 16 real-world categories (870 total samples, p=0.0000).

What we believe makes this work particularly noteworthy is the following:

1. **Robust length confound controls.** We implemented five independent controls to rule out response length as a confound. A length-only classifier achieves just 51.7% (chance level), while a truncation test using only 20 tokens still achieves 93.1%. The length-score correlation is r=0.012 (p=0.72).

2. **White lie detection.** The probe detects socially motivated white lies with the same confidence as serious lies (t=-1.331, p=0.184), suggesting the model encodes factual accuracy independently of social context or intent.

3. **Superiority over text baselines.** The hidden state probe (93.7%) significantly outperforms a TF-IDF text classifier (86.2%), confirming the probe captures information beyond surface-level text features.

We are aware of the methodological challenges you have outlined in your work on probing classifiers, including the risk of probes learning superficial features rather than genuine linguistic representations. We have tried to address these concerns rigorously, but we would greatly value your perspective on our methodology and findings.

Would you be open to a brief conversation — either in person or via video call — to discuss this work? We are based in Israel and would be happy to meet at a time and place convenient for you, including at the Technion.

I would be glad to share our full code, data, and results for your review. Everything is fully reproducible on Google Colab with a single GPU in approximately 23 minutes.

Thank you very much for your time and consideration.

Best regards,  
[Your Name]  
[Your Affiliation]  
[Your Email]  
[Your Phone]
