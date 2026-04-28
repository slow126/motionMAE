# Chapter 3 Framework Summary: Coverage-Based Coreset Selection with Diversity Guarantees

## Context for the Latex Agent

This document summarizes the planned theoretical and empirical framework for Chapter 3 of Spencer's dissertation. Chapter 3 centers on training data selection and curation for visual correspondence models (specifically CATs++), investigating how to select optimal training subsets from a heterogeneous pool (PointOdyssey, SPair-71k, PF-PASCAL, and potentially additional sources) to achieve cross-benchmark generalization across evaluation benchmarks (KITTI-2012, KITTI-2015, PF-PASCAL, PF-WILLOW, TSS, and potentially up to 9 total). The dissertation proposal should incorporate this material as the methodological and theoretical core of Chapter 3.

Chapter 2 (submitted to ECCV 2026 as "Beyond Realism: Diagnosing Transfer Gaps with Directed Coverage Metrics") introduced directed coverage using Bag-of-Flow-Vectors (BFV) descriptors as a diagnostic framework for predicting dataset transfer in dense correspondence tasks. Chapter 3 extends this from a passive diagnostic tool (ranking which dataset transfers best) to an active selection algorithm (constructing the best training subset from a heterogeneous pool).

---

## 1. Central Finding: Target Information is Necessary

The core insight shaping Chapter 3 is that target-agnostic selection cannot meaningfully beat random sampling because optimal training subsets are fundamentally target-dependent — different benchmarks require different motion distributions. This reframes the contribution from "zero-shot selector" to "target information is necessary for effective pair selection, and here is a principled way to use minimal target information."

The key evidence for this claim comes from the structured ablation suite:

- **`clustercov`** (the primary method): Coverage-based coreset construction using BFV descriptors, greedy subset selection at 0.5–1% budget. Substantially outperforms random sampling (+5–13 PCK across benchmarks) and achieves strong geometric generalization. However, it systematically under-allocates semantic sources because BFV motion descriptors alone cannot determine appropriate cross-source budget allocation when supervision formats are heterogeneous (dense optical flow vs. sparse semantic keypoints).

- **`mixed_balanced_lr1e4`**: A hand-balanced run (2K PF-PASCAL + 9K SPair + 9K PointOdyssey) that achieved superior semantic benchmark performance. This is the key counterexample showing the geometric-semantic tradeoff is a **budget allocation problem**, not a model capacity or representational interference problem.

- **`pfpascal_only`**: Strong on PF-PASCAL/PF-WILLOW but degrades on KITTI and TSS. Demonstrates that specialization fails even within the semantic correspondence family once geometric benchmarks are in scope. Established as not a meaningful competitor to `clustercov` via Pareto dominance framing.

---

## 2. Surprising Empirical Finding: BFV as a General Selection Surrogate

A central and counterintuitive empirical finding is that BFV-only coverage-based selection is effective not only for geometric benchmarks (where motion-based selection is expected to work) but also for semantic benchmarks. This operates at two levels:

### 2.1 Cross-Source Routing

When targeting semantic benchmarks like PF-PASCAL, the BFV coverage selector preferentially allocates budget toward SPair and PF-PASCAL pairs over PointOdyssey pairs — even though the selector has no concept of "semantic" versus "geometric" and operates purely on motion structure overlap. This occurs because PF-PASCAL's BFV distribution (sparse keypoints with large displacements in specific spatial regions) is geometrically closer in flow space to SPair's motion statistics than to PointOdyssey's dense, smooth, video-like flow.

### 2.2 Within-Source Curation

More strikingly, BFV `clustercov` reliably outperforms random sampling even within a single homogeneous semantic source (SPair-only pool). This means the flow statistics within a semantic dataset carry enough variation to differentiate which semantic pairs are more useful than others for a given target — without any semantic or appearance information.

The mechanistic explanation: different SPair object categories (e.g., bicycle viewed from the side with pedaling legs vs. cat turning its head) produce different BFV signatures — different spatial locations, displacement magnitudes, and motion directions. The selector doesn't know these are different categories but sees that their motion fingerprints occupy different BFV regions. When targeting a benchmark requiring diverse motion pattern coverage, it selects pairs spanning those regions rather than oversampling the densest cluster.

**Implication**: Semantic category diversity and motion diversity are correlated in correspondence datasets, at least sufficiently that optimizing motion coverage recovers semantic diversity. This is not obvious a priori and constitutes an independent empirical contribution.

### 2.3 Degeneracy Boundary

The one setting where BFV selection does not outperform random is PF-PASCAL as the source pool (~2,940 training pairs). At 1% budget (~30 pairs from ~3K), random sampling already draws representatively from a small pool. This is the predicted behavior — the method adds value precisely when the pool is large enough that random sampling leaves coverage gaps, which is the regime that matters in practice. This result should be reported and explained rather than hidden, as it validates the framework's predicted operating regime.

**Narrative framing**: "We hypothesized that BFV coverage would be a weak surrogate for semantic benchmarks, requiring appearance-based selection. Instead, we find that motion-only coverage correctly routes budget toward semantically compatible sources and meaningfully curates within homogeneous semantic sources, suggesting that motion distribution overlap is a more general compatibility signal than previously recognized. This finding extends the Chapter 2 result — where bidirectional motion coverage was the strongest single-modality predictor — from dataset ranking to active pair selection."

---

## 3. The Selection Problem: Formal Statement

Given a pool $V$ of training pairs drawn from heterogeneous sources, select a subset $S \subseteq V$ of size at most $k$ that makes a correspondence model generalize well across multiple target benchmarks simultaneously.

The central difficulty is that "generalize well" is not a single objective — KITTI requires geometric motion coverage, PF-PASCAL requires semantic category diversity, TSS requires something in between. Any single training subset makes implicit tradeoffs among these competing demands.

$$S^* = \arg\max_{S \subseteq V, |S| \leq k} f(S)$$

where $f(S)$ captures both coverage of multiple target distributions and diversity within the selected set to avoid redundancy.

---

## 4. The Two Components of f(S)

### 4.1 Utility: Directed BFV Coverage (Submodular)

For a single target benchmark $E$ with BFV distribution represented as a point cloud $X_E = \{x_1, \ldots, x_M\}$ in 4D normalized flow space:

$$C(E \to S, \varepsilon) = \frac{1}{M} \sum_{x \in X_E} \mathbf{1}\left[\min_{v \in S} \|x - \varphi(v)\|_2 \leq \varepsilon\right]$$

where $\varphi(v)$ maps training pair $v$ to its BFV descriptor cloud. This is exactly the $\varepsilon$-coverage from Chapter 2 (Eq. 5), now applied at the subset level rather than the dataset level.

**Monotone submodularity**: This function is monotone submodular, which is the key theoretical property enabling tractable optimization with guarantees.

- *Monotonicity*: Adding any pair can only increase or maintain coverage.
- *Submodularity (diminishing returns)*: Each target point can only transition from uncovered to covered once — once covered, it contributes zero marginal gain regardless of additional pairs.

**Joint utility across multiple targets**:

$$g(S) = \sum_t w_t \cdot C(E_t \to S, \varepsilon_t)$$

A non-negative weighted sum of monotone submodular functions is itself monotone submodular (standard closure property). So $g(S)$ inherits the theoretical properties and approximation guarantees apply to the joint objective directly.

### 4.2 Diversity: Min-Pairwise BFV Distance (Non-Submodular)

$$\text{div}(S) = \min_{u,v \in S : u \neq v} \text{dist}_{\text{BFV}}(u, v)$$

This directly penalizes selecting near-duplicate pairs. If two pairs have nearly identical BFV mean descriptors (e.g., consecutive PointOdyssey frames), $\text{div}(S)$ collapses to near zero. This term is not submodular — it is highly non-monotone, which is what makes the joint objective hard to optimize with standard greedy.

**Why min-distance rather than sum-of-distances or DPP**: In this setting, the redundancy problem is dominated by PointOdyssey consecutive frames — extremely dense clusters that consume budget. Min-distance is the right tool for preventing a small number of dense clusters from dominating, which is the empirically observed failure mode (the `multitarget_nn` run showing ~139K unique pairs versus ~222K for other methods).

### 4.3 Full Objective

$$f(S) = g(S) + \lambda \cdot \text{div}(S)$$

$\lambda$ controls the coverage-diversity tradeoff. In practice the diversity term acts primarily as a deduplication filter — qualitative behavior is stable across a wide range of $\lambda$.

---

## 5. Why Standard Greedy Fails

Standard greedy for monotone submodular maximization (add the element with highest marginal gain at each step) achieves $(1 - 1/e) \approx 0.63$ approximation for $g(S)$ alone but completely ignores $\text{div}(S)$.

This failure is empirically visible in the current experiments. Pure coverage-maximizing greedy on the full pool overwhelmingly selects PointOdyssey pairs because PointOdyssey has 4.4M pairs with dense BFV coverage — there are always PointOdyssey pairs with high marginal coverage gain. Consecutive frames with nearly identical BFV descriptors get selected repeatedly because each covers slightly different target points. The result wastes budget on near-duplicates, which is precisely what the `multitarget_nn` run demonstrates.

---

## 6. GIST Framework: Theoretical Grounding

GIST (Fahrbach et al.) provides a $(1/2 - \varepsilon)$-approximation algorithm for the joint diversity-plus-submodular-utility objective. The key idea is converting the hard joint problem into a sequence of easier subproblems by fixing the diversity requirement at a threshold $d$, then optimizing coverage subject to that constraint.

### 6.1 The Intersection Graph

For threshold $d$, define graph $G_d(V)$: nodes are training pairs, edges connect pairs with BFV distance $< d$. Any independent set of $G_d(V)$ automatically satisfies $\text{div}(S) \geq d$.

### 6.2 GreedyIndependentSet

At each step, only consider candidates with BFV distance $\geq d$ from all already-selected pairs. Among those, pick the one with highest marginal coverage gain. This enforces diversity by construction while approximately maximizing the submodular utility.

### 6.3 Key Lemma and Approximation Guarantee

**Lemma**: Let $S^*_d$ be optimal with diversity $\geq d$ and $|S| \leq k$. Let $T$ be GreedyIndependentSet output at threshold $d' < d/2$. Then $g(T) \geq g(S^*_d)/2$.

**Theorem (GIST)**: Sweeping over a geometric grid of thresholds and returning the best solution gives:

$$f(S) \geq (1/2 - \varepsilon) \cdot \text{OPT}$$

**Important qualification**: This guarantee applies to the coverage-diversity surrogate objective $f(S)$, not to downstream PCK. The relationship between $f(S)$ and actual transfer performance is empirical. The theoretical contribution is that optimizing the surrogate is tractable with guarantees; the empirical contribution is demonstrating that the surrogate is predictive of transfer. This is the standard gap in all surrogate-based subset selection work — it should be stated explicitly rather than obscured.

---

## 7. Domain-Specific Adaptations

### 7.1 Directed BFV Coverage as Utility (vs. GIST's Default)

GIST's original ImageNet experiments use margin uncertainty (target-agnostic model difficulty). Our utility is directed coverage $C(E \to S, \varepsilon)$: target-aware, training-free, computed purely from BFV descriptors, and grounded in Chapter 2. The marginal gain at each greedy step is the number of additional target BFV vectors brought within $\varepsilon$-radius, computable in $O(N_{\text{kpts}} \cdot \log M)$ via KD-tree.

### 7.2 Source-Partitioned Diversity Thresholding

The pool is not uniformly distributed in BFV space. PointOdyssey has 4.4M densely clustered pairs; SPair has 53K sparse, heterogeneous pairs. A single global threshold either over-deduplicates SPair or under-deduplicates PointOdyssey.

**Solution**: Apply diversity constraints within each source independently:

$$\text{dist}_{\text{partition}}(u, v) = \begin{cases} \text{dist}_{\text{BFV}}(u, v) & \text{if } \text{source}(u) = \text{source}(v) \\ \infty & \text{if } \text{source}(u) \neq \text{source}(v) \end{cases}$$

This modifies $G_d(V)$ to only connect same-source pairs. The GIST approximation guarantee is preserved because the key lemma's injection argument still holds — optimal pairs from each source are captured by greedy pairs from the same source.

**Operational note**: This means no cross-source deduplication. If PointOdyssey and SPair share a BFV region, redundant cross-source pairs could be selected. This is unlikely in practice due to the distributional differences between dense video flow and sparse semantic keypoints, but should be acknowledged.

### 7.3 Supervision-Density-Derived Coverage Radius

The coverage radius $\varepsilon$ determines what counts as "covered." Dense targets (KITTI, ~128 keypoints/pair) have densely sampled BFV clouds where small $\varepsilon$ is appropriate. Sparse targets (PF-PASCAL, ~12 keypoints/pair) have sparsely sampled BFV clouds where larger $\varepsilon$ avoids requiring implausibly precise matches.

$$\varepsilon_t = \varepsilon_{\text{base}} \cdot \rho_t^{-\beta}$$

where $\rho_t$ is supervision density (mean keypoints per pair) and $\beta > 0$ controls density sensitivity. This derives $\varepsilon$ from an observable target property without training.

**Note on relationship to existing beta parameter**: The existing beta in the scoring function controls normalization of per-target-point contributions. This epsilon-beta controls coverage radius. They address related concerns (density-awareness) through different mechanisms. For the dissertation, these should be unified into a single density-awareness parameter or one should be derived from the other to avoid confusion.

---

## 8. Multi-Target Portfolio Optimization

### 8.1 Target Weights

The weights $w_t$ in the joint utility $g(S) = \sum_t w_t \cdot C(E_t \to S, \varepsilon_t)$ determine budget allocation across benchmarks. **Uniform weights are the correct starting point** and may be sufficient.

### 8.2 Connection to Chapter 2 Ridge Predictor (Deferred/Future Work)

The Chapter 2 ridge predictor can in principle provide adaptive weights: targets predicted to transfer poorly receive higher weight. However, the predictor was validated at dataset-level ranking (0.70 pairwise accuracy) and its reliability at subset-level scoring is not yet established. Sample-pair-level rescoring (computing directed coverage features on subsets rather than full datasets, then feeding to the ridge predictor) is the methodological bridge, and preliminary work on this approach exists.

**Recommendation for the proposal**: Present iterative predictor-derived reweighting as a clearly scoped extension. Run the primary experiments with uniform weights. If uniform weights show a clear target imbalance identifiable from coverage diagnostics, perform one manual reweighting round informed by the predictor as a proof-of-concept. This is more defensible than an automated loop with no convergence guarantee, and sufficient for a dissertation contribution.

---

## 9. Experimental Plan

### 9.1 Required Ablation Ladder

The ablation structure is what justifies the framework complexity. Each step must demonstrate incremental value:

1. **Random sampling** (baseline): Random subsets at each budget level, multiple seeds for confidence intervals.
2. **Plain greedy coverage** (current `clustercov`): Standard greedy maximizing $C(E \to S, \varepsilon)$ with no diversity constraint.
3. **Greedy + FPS deduplication** (preprocessing): FPS-based clustering within PointOdyssey before pool construction, then greedy on the deduplicated pool.
4. **Greedy + source-partitioned diversity threshold** (GIST-lite): Single well-chosen diversity threshold per source, enforced during selection.
5. **Full GIST sweep** (if time permits): Geometric grid of thresholds, return best across all thresholds.

If gains are monotonic up the ladder, the framework complexity is justified. If gains plateau after step 3 or 4, the full sweep becomes future work.

### 9.2 Source Ablation Suite

Clean three-source ablation isolating source quality from selection strategy:

- **PointOdyssey-only**: Expected strong KITTI, weak semantic. Validates BFV coverage within a dense geometric source.
- **SPair-only**: Key diagnostic for whether category diversity within semantic data matters. Validates BFV coverage within a homogeneous semantic source (the surprising positive result).
- **PF-PASCAL-only**: Degeneracy case — pool too small for selection to meaningfully beat random. Validates the predicted operating regime boundary.

### 9.3 Cross-Architecture Validation

Currently: RAFT and CATs++. Adding a third architecture (e.g., FlowFormer, GMFlow, or a UFM-family model) gives sufficient context diversity for meaningful pairwise ranking statistics across up to 9 benchmarks.

### 9.4 Pareto Frontier Analysis

The key evaluation framing is Pareto dominance across benchmarks simultaneously, not single-benchmark comparisons. `clustercov` should be evaluated against the Pareto frontier: a method is justified if it is not Pareto-dominated by any simpler alternative across all targets. This directly addresses the "is the complexity worth it" reviewer question.

---

## 10. Reviewer Concerns and Defenses

### 10.1 "Overbuilt for the evidence"

**Defense**: The ablation ladder demonstrates incremental value of each component. More models (up to 3 architectures) and benchmarks (up to 9) provide sufficient experimental scope. The framework is presented modularly — each adaptation is independently justified and independently ablated.

### 10.2 "The ½-approximation is for the surrogate, not PCK"

**Defense**: State explicitly that the guarantee applies to the coverage-diversity objective. The empirical contribution is demonstrating predictiveness of the surrogate. This gap exists in all surrogate-based subset selection work.

### 10.3 "Min-pairwise distance is a blunt diversity measure"

**Defense**: The redundancy problem is dominated by PointOdyssey consecutive frames (dense clusters consuming budget). Min-distance directly addresses this failure mode. More sophisticated diversity measures (sum-of-distances, DPP) would not preserve the GIST guarantee.

### 10.4 "BFV is the wrong objective for semantic benchmarks"

**Defense**: This is directly refuted by the empirical finding. BFV coverage correctly routes budget cross-source AND meaningfully curates within-source on semantic datasets (SPair-only outperforms random). Motion diversity correlates with semantic category diversity in correspondence datasets — an independent empirical contribution.

### 10.5 "How does this connect to the LLM data mixing literature?"

**Defense**: DoReMi and Data Mixing Laws optimize source-level mixing proportions (how much of each dataset to include). Our framework operates at pair-level granularity within and across sources simultaneously. The connection is: both recognize that target information is necessary for effective data curation; both use proxy objectives rather than training the full model. The key difference is pair-level selection versus source-level proportions — our approach subsumes source-level mixing as a special case (source proportions emerge from pair-level selection).

---

## 11. Narrative Arc: Chapter 2 → Chapter 3

**Chapter 2**: Directed coverage is introduced as a diagnostic tool for predicting dataset-level transfer. The ridge predictor ranks candidate training datasets using bidirectional BFV and appearance coverage features, achieving 0.70 pairwise ranking accuracy. The contribution is measurement and prediction.

**Chapter 3**: The same directed coverage metric transitions from passive diagnosis to active construction. The $\varepsilon$-coverage from Chapter 2 (Eq. 5 in the ECCV paper) becomes the submodular utility function driving pair-level selection. The contribution is optimization and curation.

**Key conceptual link**: Chapter 2 asks "given a dataset, how well will it transfer?" Chapter 3 asks "given a target, what training subset should we construct?" The coverage metric is the bridge — the same quantity that predicts transfer also drives selection, and the GIST framework provides the theoretical guarantee that this selection is approximately optimal.

**The BFV generality finding strengthens both chapters**: Chapter 2 showed motion coverage is the strongest single-modality predictor. Chapter 3 shows it's also the most effective single-modality selection signal, extending from dataset ranking to pair curation, and from geometric to semantic benchmarks.

---

## 12. Summary of Theoretical Claims and Their Basis

| Claim | Basis |
|---|---|
| $g(S)$ is monotone submodular | Coverage functions are submodular; weighted sum preserves submodularity |
| GIST achieves $(1/2-\varepsilon)$-OPT | Theorem 3.1, Fahrbach et al. |
| Guarantee holds for joint multi-target $g(S)$ | Closure of submodularity under non-negative linear combinations |
| Guarantee holds under source partitioning | Key lemma injection argument preserved with partitioned graph |
| BFV coverage is effective for semantic targets | Empirical: cross-source routing + within-source SPair curation |
| Degeneracy at small pool sizes | Predicted by coverage theory; validated by PF-PASCAL-only result |
| Target information is necessary | Empirical: target-agnostic selection cannot beat random; confirmed by target-dependent optimal subsets |
