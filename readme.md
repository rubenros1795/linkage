# Code for Article "Epistemic Capture Through Specialization in Post-World War II Parliamentary Debate"

This repository contains the code belonging to the article "Epistemic Capture Through Specialization in Post-World War II Parliamentary Debate", published in _Computational Humanities Research_.

The data used for the analysis can be found on [Zenodo](10.5281/zenodo.16754330).

The article investigates the phenomenon of "epistemic capture": the formation of stable specialisms in parliamentary debate. We demonstrate that in the Dutch House of Representatives, the postwar period saw far-reaching specialization of debates. We inquire into this transformation, mapping its character and impact on micro-, macro-, and meso-levels.

---

In ```0-lda_eval_entropy.py``` we aggregate the Shannon Entropy for the topic distributions. This serves to check whether speeches see an overall rise in diversity.

In ```1-macro-modularity.py``` we calculate network modularity and density as indicators of macro-level specialization.

In ```1-macro-stability.py``` we report the stability of network clusterings over time as a second indicator of specialization.

In ```2-meso-tns.py``` we look at "Topic Neighbourhood Stability": the extent to which topics share the same neighbourhoods over time.

In ```3-micro-actors.ipynb``` we study specialization on the level of individual actors. We identify dynamic communities using the ```cdlib``` python module.
