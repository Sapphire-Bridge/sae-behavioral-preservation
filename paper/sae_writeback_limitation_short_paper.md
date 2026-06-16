# Geometric Fidelity Does Not Certify Behavioral Preservation

**Author:** Felix Borck

**ORCID:** <https://orcid.org/0009-0001-9137-0730>

**Code and artifacts:** <https://github.com/Sapphire-Bridge/sae-behavioral-preservation>

## Abstract

This paper asks whether high geometric fidelity also answers a behavioral question. Does a sparse autoencoder preserve the behaviorally relevant causal effect of an activation under matched activation patching, or only its geometric structure? We test this in a deliberately controlled, context-sensitive setting of disambiguation in Gemma 3 4B. At layer 4, a frozen width-16k JumpReLU SAE reconstructs activations with cosine similarity 0.997, relative MSE 0.007, and FVU 0.137, yet patching from that reconstruction recovers only 62.1% of the raw activation-patching effect (0.506 versus 0.815). At layer 8, by contrast, reconstruction fidelity is also high (cosine 0.998, relative MSE 0.007, FVU 0.060), but behavioral recovery is almost complete (0.907 versus 0.931, recovery ratio 0.974). Two intervention sites can therefore show strong standard reconstruction diagnostics while differing substantially in whether they preserve behavior under matched intervention. High reconstruction fidelity certifies geometric fit, not preservation of the counterfactual behavioral role of the original activation. If SAE-based explanations are meant to support claims about model behavior, their adequacy has to be tested with matched behavioral interventions rather than inferred from reconstruction fidelity alone.

## 1. Introduction

Much of current SAE evaluation lives in geometry. Reconstructions are judged by cosine similarity, reconstruction error, and related fidelity criteria -- useful measurements, but ones that answer a narrower question than interpretability needs when making claims about behavior.

This work studies a simple preservation question. When an internal activation is passed through a sparse autoencoder and then used in an activation-patching intervention, does the reconstructed activation preserve the behaviorally relevant causal effect of the original activation? The result is not a claim that SAEs generally fail. It is a criterion for testing whether sparse reconstruction preserves behavior at a given layer-token intervention site.

A state can be close geometrically and still lose the signal that determines its behavior. High-fidelity reconstruction can therefore be a misleading comfort. It can suggest preservation even where the causal signal that matters for downstream behavior has already been weakened. The question of this paper is therefore stricter than standard reconstruction evaluation. When an activation already exhibits strong fidelity metrics, does an SAE reconstruction-and-patching path preserve the donor-directed behavioral effect of that activation under matched intervention?

The analogy is to a high $R^2$ in regression. A strong goodness-of-fit statistic can coexist with failure on the causal or counterfactual question one actually cares about.

We test this in a deliberately controlled and highly context-sensitive setting of homonym disambiguation. Whether a prompt containing a word such as "bank" continues toward {loan, money, account} or toward {river, water, stream} depends critically on the contextual state the model has built. If a reconstruction genuinely preserves the behaviorally relevant effect of an activation, that preservation should appear under matched activation patching at the same layer-token intervention site, measured against the donor-directed effect of the raw activation itself.

The model computation is deterministic, but the behavioral role of an internal state is relational. It depends on the context, intervention, and readout through which that state affects behavior.

We use the `google/gemma-scope-2-4b-pt` JumpReLU SAEs (width-16k; McDougall et al., 2025), whose configs target `google/gemma-3-4b-pt` (Gemma Team et al., 2025).[^gemma-scope] In Gemma 3 4B, L4 and L8 both reach high fidelity (cosine >= 0.997, relative MSE 0.007, FVU 0.137 at L4 and 0.060 at L8), yet matched SAE reconstruction patching recovers only 62.1% of the raw effect at L4 versus 97.4% at L8. At layer 4, rank-controlled PCA patching does not show the same under-recovery as SAE reconstruction patching; at layer 8, it falls below the SAE.

[^gemma-scope]: Gemma Scope for Gemma 2 is the predecessor suite (Lieberum et al., 2024).

Feature steering shows that individual learned directions can influence model outputs, often through out-of-distribution interventions. Our test asks a different preservation question. It asks whether the reconstruction of a naturally occurring activation preserves the effect of the original activation under matched activation patching.

Influence does not imply preservation. Preservation does not imply explanation.

This paper makes three contributions:

- **A behavioral preservation assay for SAE reconstructions.** We report CRR, the causal recovery ratio, a matched activation-patching statistic that measures how much of a raw activation-patching effect is retained by a test patching channel.
- **A substantial L4/L8 contrast.** In Gemma 3 4B, SAE reconstruction patching at layer 4 recovers only 62.1% of the raw activation-patching effect despite strong reconstruction diagnostics, while layer 8 recovers 97.4%.
- **A layer-specific under-recovery pattern.** At layer 4, rank-controlled PCA does not show the same under-recovery as SAE reconstruction patching, while layer 8 reverses the ordering. This leaves open which part of the SAE reconstruction-patching path is responsible.

![Strong geometric diagnostics, different behavioral recovery. Layer 4 and layer 8 both show strong SAE reconstruction diagnostics, with lower FVU at layer 8 than layer 4, but only layer 8 shows high recovery of the raw activation-patching effect.](../figures/sae_writeback_limitation/main_effect_figure.svg){width=\textwidth}

## 2. Method

### 2.1 Target behavior and score

The target behavior is donor-directed context-sensitive disambiguation on DISAMB. Each evaluation case pairs a donor context with a recipient prompt and asks whether transplanting donor-side internal state into the recipient shifts the recipient in the donor-consistent direction. DISAMB contains 52 authored cases: 13 ambiguous lexical targets, each instantiated as four matched context-pair variants. Each DISAMB item is a minimal pair built around one ambiguous target word. One side supplies the donor context and the other the recipient prompt. Both contain the same ambiguous target string, with intervention spans matched by tokenization. Each sense label is represented by exactly three continuations, held fixed across the paired variants for that target to ensure consistent aggregation. Each pair is evaluated in both directions (`A->B` and `B->A`), and "donor-directed" means that the receiver is scored against the donor side's expected label. The outcome is measured as a shift in a preference margin rather than as a raw token logit.

Representative DISAMB items illustrate the task surface:

| target | side A label and prompt fragment | side B label and prompt fragment | label continuations |
|---|---|---|---|
| `bank` | `finance`: "headed to the bank to file forms related to the" | `river`: "rested near the bank at the embankment and watched the" | `finance`: {loan, money, account}; `river`: {river, water, stream} |
| `bat` | `animal`: "saw a bat fly through the" | `sports`: "carried the bat to the stadium and watched the" | `animal`: {cave, night, air}; `sports`: {ball, pitcher, game} |
| `spring` | `season`: "looks forward to spring because gardens begin to" | `water`: "followed the spring from a hillside source toward the" | `season`: {bloom, warm, grow}; `water`: {water, rocks, stream} |

For directed evaluation row `i` and label `y` with continuation set `C_y`, the scoring pipeline aggregates continuation log-probabilities using `logmeanexp` over the label's continuation set. Length normalization is enabled (`normalize_by_length = true`), with `|c|` denoting tokenized continuation length, so the label score is

$$
s_i(y) =
\operatorname{logmeanexp}_{c \in C_y}
\left[
\frac{1}{|c|}
\sum_{t=1}^{|c|}
\log p(c_t \mid \mathrm{prompt}_i, c_{<t})
\right].
$$

The donor-directed margin is then

$$
m_i = s_i(y_{\mathrm{donor}}) - \max_{y' \ne y_{\mathrm{donor}}} s_i(y').
$$

Equivalently, `margin = expected_label_score - best_other_score`, where the competing score is the highest non-donor label score. The intervention effect is

$$
\mathrm{effect}_i = m_i(\mathrm{patched}) - m_i(\mathrm{base}).
$$

Here, "donor-directed effect" means the directed-row patched-minus-base shift in this margin under the specified activation-patching intervention.

The behavioral score is intentionally narrow. The label score averages over three fixed continuations per sense, so the result is not driven by a single continuation token. The margin asks whether the donor sense beats the strongest non-donor sense label, rather than measuring the donor sense's absolute probability. The patched-minus-base effect then subtracts the recipient prompt's prior preference and measures the shift caused by activation patching. Thus, behavioral preservation here means preserving this donor-directed margin shift under matched intervention, not preserving semantics in general.

### 2.2 Matched Activation-Patching Comparison

The core comparison evaluates two intervention substrates at the same matched residual-stream intervention site, the target-token `resid_post` vector at a specified transformer layer.

1. **Raw activation patching.** The donor residual activation replaces the recipient residual activation at the matched layer-token intervention site.
2. **SAE reconstruction patching.** Let `h_d` and `h_r` be the donor and recipient residual activations, and let `E,D` denote the configured SAE encoder and decoder. The reported SAE channel patches the decoded latent delta, `h_r + [D(E(h_d)) - D(E(h_r))]`, after the configured input transform; the run also checks the equivalent two-decode, error-preserving arm as a numerical diagnostic.

The relevant criterion is behavior preservation under matched activation patching. If the SAE reconstruction preserves the behaviorally relevant structure of the donor state, the SAE reconstruction-patching effect should track the raw activation-patching effect at the same intervention site. If it does not, then the SAE reconstruction-and-patching path is not preserving whatever structure is needed to recover the effect at that site, even when the activation-level reconstruction looks excellent by standard fidelity metrics.

Our main comparison focuses on layers 4 and 8 of Gemma 3 4B using Gemma Scope width-16k JumpReLU SAEs. We chose these as an early/later contrast to test whether depth changes how well high-fidelity reconstructions preserve behavior. We evaluate the 52 cases in both donor directions across five residual-stream layers (4, 5, 8, 11, 16), giving 52 x 2 x 5 = 520 comparability rows. We focus on L4 and L8 because they provide the cleanest illustration of how behavioral recovery can differ substantially by depth. The broader five-layer profile reported in Appendix B is consistent with matched-patching under-recovery being heterogeneous across depth rather than monotonic. Supporting diagnostics, including near-zero identity-control deviations, support the interpretation that the L4/L8 contrast is not a trivial intervention artifact.

### 2.3 Summary statistics

The primary reported quantities are the raw activation-patching effect, the SAE reconstruction-patching effect, and the paired `sae_minus_raw` difference. We also report a raw-patching-normalized recovery statistic, the causal recovery ratio (CRR). For directed evaluation rows `i`, let `m` be oriented so that larger values indicate stronger donor-directed target behavior. The expectation below denotes the equal authored-pair average over directed rows; the bootstrap resamples authored pairs. For a test patching channel `T` and reference patching channel `R`, we define the paper statistic `CRR(T | R)` as follows.

$$
\mathrm{CRR}(T \mid R) =
\frac{
\mathbb{E}_i\left[m(\mathrm{patched}_i(T)) - m(\mathrm{base}_i)\right]
}{
\mathbb{E}_i\left[m(\mathrm{patched}_i(R)) - m(\mathrm{base}_i)\right]
}.
$$

In the main comparison, `R` is matched raw activation patching at the same intervention site, and `T` is SAE reconstruction patching. In PCA controls, `T` is rank-controlled PCA projection patching. The projection basis is an uncentered SVD/PCA basis fit on raw donor-minus-receiver residual deltas using leave-one-pair-out over the 52 pairs. At each row, we set `n_active = #{j: |z_recv,j| > 1e-6}` for receiver SAE latents and use projection rank `min(n_active, r_max)`, where `r_max` is the number of components available from the leave-one-pair-out fit. We add the projected delta back to the receiver residual and patch the target-token `resid_post` span. It is a weak projection control rather than a reconstruction-fidelity control for the SAE. CRR estimates how much of the matched raw-patching behavioral effect is retained by the test patching channel. We compute a ratio of mean effects rather than a mean of per-item ratios, and interpret CRR only when the aggregate reference effect is non-negligible and directionally aligned. Values between 0 and 1 indicate partial recovery, values above 1 indicate amplification relative to raw activation patching, and values below 0 indicate reversal. CRR is a positive-control normalization of behavioral effect recovery, not independent proof of mechanistic faithfulness.

Reconstruction fidelity is summarized by activation cosine, relative MSE, and FVU. FVU is reported as a pair-clustered ratio of reconstruction-error SSE to layer-centered activation SSE over analysis-included receiver activations. The denominator is centered using the layer mean over included receiver-token activations. Supporting controls reported later in the paper use the PCA and compact top-k summaries, but the core result does not depend on any source-only diagnostic.

Five row-level diagnostics check activation, margin, score, CLT C/D equivalence, and identity-patch effects. Tolerance is `1e-6` for the activation ratio and `1e-4` for the absolute-difference gates. Across the full five-layer comparability run, `5` rows exceed pre-specified invariance diagnostics. Under the governed policy these flags are diagnostic; flagged rows remain in the reported aggregates. All headline uncertainty estimates use `B = 1000` pair-cluster bootstrap resamples over authored context pairs, with 95% confidence intervals. Target-level heterogeneity is assessed separately over the 13 ambiguous lexical targets. For the PCA control, the intended reading is deliberately weak: PCA tests whether a rank-controlled low-rank projection preserves the effect, not whether sparsity alone is the sole pressure point.

Full reproducibility surface is given in Appendix C and in the release manifest.

## 3. Main Result

Figure 1 and Table 1 show the central contrast. L4 and L8 both have very high cosine similarity and low relative MSE; FVU is lower at L8 than at L4. Nevertheless, these reconstruction diagnostics alone do not certify behavioral preservation. SAE reconstruction patching recovers only 62.1% of the raw effect at L4 but 97.4% at L8.

| Quantity | L4 | L8 |
|---|---:|---:|
| cosine | 0.997 | 0.998 |
| relMSE | 0.007 | 0.007 |
| FVU | 0.137 | 0.060 |
| raw | 0.815 | 0.931 |
| SAE | 0.506 | 0.907 |
| SAE-raw [95% CI] | -0.309 [-0.636, -0.027] | -0.024 [-0.270, 0.224] |
| CRR | 0.621 | 0.974 |
| PCA effect | 0.829 | 0.688 |

**Table 1.** L4/L8 comparison from the main summary, rounded to three decimals. `CRR` is computed as mean SAE effect / mean raw effect; the PCA effect row reports an effect, not a CRR. The inferential contrast is the paired `SAE-raw` difference at the same layer-token intervention site, not a comparison of marginal raw and SAE confidence intervals. Full-precision values remain in the release artifacts.

In this comparison, standard fidelity diagnostics do not certify matched behavioral preservation. Two intervention sites can show strong geometric diagnostics while differing sharply in recovered behavioral effect. The result is not that sparse reconstructions generally fail, but that reconstruction fidelity alone does not identify where matched SAE reconstruction patching will preserve the effect.

## 4. Diagnostics

### 4.1 PCA control

The PCA control is included to test a weak alternative explanation of the layer-4 under-recovery. Perhaps any low-rank projection of the donor-recipient intervention delta would reduce the effect, regardless of sparsity. The comparison does not support that reading. At layer 4, rank-controlled PCA patching sits at `0.829`, essentially at the raw level of `0.815`, while SAE reconstruction patching drops to `0.506`. Rank-controlled PCA therefore does not show the same L4 under-recovery.

At L8, the ordering reverses. SAE remains near raw (`0.907` versus `0.931`), whereas PCA falls to `0.688`. This supports a depth-heterogeneous interpretation of matched-patching preservation, without showing that sparsity is the sole pressure point or that PCA is generally a better substrate.

### 4.2 Compact top-k subsets

On a separate deterministic top-k diagnostic surface, compact activation-mass subsets fail to recover the full-set SAE reconstruction-patching effect. At L4, full-set patching is `0.496`, while top-20/50/100 reach only `0.156`/`0.138`/`0.285`; at L8, full-set patching is `0.907`, while top-20/50/100 reach `0.259`/`0.342`/`0.427`.

Because this diagnostic uses a separate 50/50 split and aggregation surface, the L4 full-set value differs slightly from the centerpiece estimate (`0.506`) and should be read qualitatively rather than as a headline estimate.

### 4.3 Target-level robustness note

Because DISAMB spans 13 target words, we also computed a target-level difference-in-difference robustness analysis on the pair-level table. For each target, we averaged direction-averaged pair-level `sae_minus_raw` values across its four matched context-pair variants separately at layer 4 and layer 8, then formed

$$
\begin{aligned}
\Delta\Delta_t
&=
\operatorname{mean}(\mathrm{sae\_minus\_raw})_{t,L4} \\
&\quad -
\operatorname{mean}(\mathrm{sae\_minus\_raw})_{t,L8}.
\end{aligned}
$$

This is best understood as a target-level paired interaction contrast, analyzed with an exact sign-flip randomization test and leave-one-target-out sensitivity analysis.

The mean contrast over targets is negative (`DeltaDelta = -0.285`), consistent with stronger SAE under-recovery at layer 4 than at layer 8. An exact sign-flip randomization test over the `13` target-level contrasts gives a two-sided `p = 0.082`.

Target-level heterogeneity is substantial. The strongest negative contributors to the layer-4-versus-layer-8 contrast are `spring` (`-1.163`), `date` (`-1.125`), and `mole` (`-0.862`), whereas the strongest counterforce is `bank` (`+0.672`). Leave-one-target-out estimates remain negative after removing `spring` (`-0.212`), `date` (`-0.215`), `mole` (`-0.237`), `watch` (`-0.312`), or `bank` (`-0.365`), so the aggregate contrast is not explained by a single-target artifact.

We therefore treat the word-level pattern as a robustness and heterogeneity analysis rather than as an independent headline result.

Additional auxiliary diagnostics are reported in Appendix A. They support the same L4/L8 contrast. These are auxiliary checks on the same evaluation surface, not a full sensitivity analysis over alternative behavioral metrics.

## 5. Relation to the Field

Methodologically, the closest family is activation patching, including causal tracing and interchange-intervention variants. Standard denoising activation patching asks which internal component restores behavior between a corrupted/base condition and a clean/reference condition. Prior methodological work emphasizes that patching results depend on metric choice and normalization (Zhang and Nanda, 2024; Heimersheim and Nanda, 2024). CRR uses the same normalized-recovery logic but changes the primary question of how much of a matched raw activation-patching positive control is preserved by a test patching channel at the same intervention site.

Templeton et al. (2024) explicitly frame SAE training loss in Scaling Monosemanticity as a proxy rather than a gold-standard evaluation criterion. They describe their weighted reconstruction-MSE-plus-L1 loss as a "useful proxy" under their chosen L1 weight, but also as an "imperfect metric," noting that other L1 weights or other objectives altogether might be better proxies to optimize. The L4/L8 contrast here operationalizes that caveat in a concrete intervention setting. Two intervention sites can show strong reconstruction metrics while differing substantially in behavior preservation under matched activation patching.

Matryoshka SAE results point in the same direction. Modestly worse reconstruction can coexist with comparable downstream cross-entropy loss and better performance on some targeted feature-quality evaluations. This is another reason to treat reconstruction quality as a proxy rather than a behavioral certificate (Bussmann et al., 2025).

Anthropic's circuit-tracing methods paper provides a useful nearby comparison because it separates replacement-model fit from mechanistic faithfulness under perturbation (Ameisen et al., 2025). In their local replacement-model evaluation, they compare net perturbations in the replacement model and the underlying model using cosine similarity and MSE. For the 10M dictionary in the 18-layer model, they report cosine-faithfulness around 60-80% in the layer following the intervention or perturbation layer. This is not the same metric as reconstruction cosine. It nevertheless motivates the same methodological distinction, since source-side fit does not by itself certify faithfulness under perturbation or intervention. We ask an analogous preservation question for SAE reconstruction patching. Even with reconstruction cosine 0.997-0.998, layer 4 loses a substantial fraction of the matched raw activation-patching effect.

The same point matters for biology-style analyses built on SAE features, including the attribution-graph style explored by Lindsey et al. (2025). Such explanations become stronger when the basis they rely on also passes a matched activation-patching preservation test at the intervention sites they claim to explain, independent of source-side reconstruction fidelity.

## 6. Discussion

The main result is narrow but clear on the governed aggregate. For the tested context-sensitive behavior at an active early-layer intervention site, high SAE fidelity does not certify behavior preservation under matched decode-and-patch intervention.

Conceptually, CRR reframes representation evaluation as a behavioral preservation assay based on matched activation patching. Instead of asking only whether a representation has good intrinsic reconstruction metrics, CRR asks whether a candidate patching channel preserves the behavioral effect of a matched raw activation-patching positive control. This makes the causal role of a representation contrastive and intervention-relative: it is assessed only relative to a base condition, a reference intervention, a test intervention, and an oriented behavioral readout.

One possible objection is that the layer-4 gap reflects properties of the encode-decode-patch intervention path rather than of the SAE basis alone, for example decoder bias, off-manifold patching, reconstruction-geometry mismatch, or nonlinear downstream sensitivity to small reconstruction errors. Two features of the current evidence make a purely generic protocol-level reading less compelling. First, the same encode-decode-patch pipeline produces near-complete recovery at layer 8, so the intervention path is not generically destructive across the tested intervention sites. Second, rank-controlled PCA patching at layer 4 remains close to raw despite also relying on a project-and-patch intervention. Rank-controlled PCA therefore does not show the same L4 under-recovery. These controls do not isolate sparsity as the sole pressure point, but they do argue against the weakest protocol-only reading of the result and reinforce the paper's narrower main claim that high-fidelity reconstruction metrics alone do not certify behavior preservation under matched activation patching.

Another objection is that the L4/L8 behavioral gap might simply track FVU, since L4 has worse FVU than L8. The auxiliary five-layer profile in Appendix B is not consistent with a simple monotone FVU-only account. L5 has worse FVU than L4 (`0.180` versus `0.137`) but much higher CRR (`0.908` versus `0.621`), and L4 has the second-best FVU among the five layers while also showing the second-lowest CRR. The intended reading is narrower. FVU may matter, but FVU alone does not order behavioral preservation in this run.

The layer-8 contrast already rules out the broadest anti-SAE reading. Sparse bases can and do preserve behavioral effects under matched activation patching. The claim is that existing proxy metrics do not tell us in advance which intervention sites they will.

The present evidence is limited to one model family, one SAE width, and one behavior class. The layer-4 interval excludes zero only narrowly (sae_minus_raw 95% CI: [-0.636, -0.027]), warranting replication rather than generalization.

The obvious open question is rescue. A wider SAE, a different sparsity target, or a different architecture may close the gap at layer 4. But because the standard fidelity metrics are already high at that intervention site, any rescue attempt has to be evaluated causally rather than by recycling the same proxy metrics that failed to distinguish layer 4 from layer 8.

These results motivate adding matched behavior-preservation checks to SAE evaluation alongside reconstruction fidelity. Sparsity remains useful for training and screening, but behavior-facing claims need targeted checks under matched intervention. If future sparse decompositions pass these checks, feature-level explanations will be stronger, not weaker.

## 7. Conclusion

High SAE fidelity can answer a geometric question without answering a behavioral one. In DISAMB, layer 4 and layer 8 both show strong reconstruction diagnostics, with lower FVU at layer 8 than layer 4, yet only layer 8 shows high recovery of the raw activation-patching effect. The layer-4 result shows that a reconstruction can remain close in activation space while losing part of what the state does for the model. This is not a blanket anti-SAE claim. Layer 8 shows near-raw recovery, and the present evidence is limited to Gemma 3 4B, Gemma Scope width-16k JumpReLU SAEs, and DISAMB. The point is narrower and more basic. Geometry can certify proximity, but it cannot by itself certify behavioral preservation. When an SAE basis is offered as the explanatory substrate at an intervention site, matched activation patching should be used to test whether the behaviorally relevant effect survives.

## Appendix A. Auxiliary diagnostics

We computed four auxiliary diagnostics on the same evaluation surface to check whether the L4/L8 contrast is driven by direction imbalance, degenerate baselines, control failures, or the particular margin definition.

First, the L4 SAE-minus-raw gap is negative in both donor directions (`-0.435` and `-0.183`). Second, receiver baselines are rarely degenerate: `2/104` directions have near-zero base margins and `7/104` are already donor-consistent at baseline. Third, an auxiliary binary-log-odds margin gives the same qualitative contrast, with L4 at `-0.342` and L8 at `-0.014`. Fourth, for the L4/L8 comparison, identity controls show `0` failures.

These diagnostics support the main interpretation of the L4/L8 contrast. They should not be read as a full sensitivity analysis over behavioral metrics, since they are auxiliary checks computed on the same evaluation surface rather than a separate rerun of the full experiment.

## Appendix B. Five-layer profile

The auxiliary five-layer summary provides context for the L4/L8 contrast rather than a separate headline claim. L5 and L8 show near-raw recovery, while L4, L11, and L16 under-recover, consistent with heterogeneity across depth rather than a monotonic depth trend. The main text focuses on L4/L8 because that pair was the compact early/later contrast used for the primary exposition, not because L4 is the worst layer in the five-layer profile. The five-layer profile also serves as a check against a simple FVU-only account. L4 has the second-best FVU but the second-lowest CRR, and L5 has worse FVU than L4 but much higher recovery.

| Layer | FVU | CRR [95% CI] | SAE-raw [95% CI] |
|---|---:|---:|---:|
| 4 | 0.137 | 0.621 [0.263, 0.954] | -0.309 [-0.636, -0.027] |
| 5 | 0.180 | 0.908 [0.626, 1.275] | -0.069 [-0.271, 0.140] |
| 8 | 0.060 | 0.974 [0.749, 1.320] | -0.024 [-0.270, 0.224] |
| 11 | 0.336 | 0.576 [0.285, 0.824] | -0.300 [-0.502, -0.092] |
| 16 | 0.538 | 0.632 [0.325, 0.940] | -0.204 [-0.397, -0.023] |

## Appendix C. Reproducibility

The governed release manifest, `release_manifest.json`, is stored with the release tables. It records the exact model and tokenizer revisions, dataset bundle hash, SAE bundle hash, public layer entries, execution profile, and inclusion policy.

The public repository URL is listed on the first page with the code and artifact statement.

In brief, the main public comparison centers on layers 4 and 8, while the source comparability profile also includes layers 5, 11, and 16. The run uses Gemma 3 4B with Gemma Scope width-16k JumpReLU residual-stream SAEs, `float32`, `seed = 42`, `bootstrap_seed = 42`, and `B = 1000`; the exact device and build profile are recorded in `release_manifest.json`.

## References

- Ameisen, E., Lindsey, J., Pearce, A., Gurnee, W., Turner, N. L., Chen, B., Citro, C., et al. (2025). *Circuit Tracing: Revealing Computational Graphs in Language Models.*
- Bussmann, B., Nabeshima, N., Karvonen, A., and Nanda, N. (2025). *Learning Multi-Level Features with Matryoshka Sparse Autoencoders.*
- Lieberum, T., Rajamanoharan, S., Conmy, A., Smith, L., Sonnerat, N., Varma, V., Kramár, J., Dragan, A., Shah, R., and Nanda, N. (2024). *Gemma Scope: Open Sparse Autoencoders Everywhere All At Once on Gemma 2.*
- Gemma Team et al. (2025). *Gemma 3 Technical Report.*
- Lindsey, J., et al. (2025). *On the Biology of a Large Language Model.*
- McDougall, C., Conmy, A., Kramár, J., Lieberum, T., Rajamanoharan, S., and Nanda, N. (2025). *Gemma Scope 2 Technical Paper.*
- Rajamanoharan, S., Lieberum, T., Sonnerat, N., Conmy, A., Varma, V., Kramár, J., and Nanda, N. (2024). *Jumping Ahead: Improving Reconstruction Fidelity with JumpReLU Sparse Autoencoders.*
- Templeton, A., et al. (2024). *Scaling Monosemanticity: Extracting Interpretable Features from Claude 3 Sonnet.*
- Zhang, F., and Nanda, N. (2024). *Towards Best Practices of Activation Patching in Language Models: Metrics and Methods.*
- Heimersheim, S., and Nanda, N. (2024). *How to Use and Interpret Activation Patching.*
