# ⚠️ DRAFT — pre-Fase-4 — Pre-registro de ejecución: OQ1 + Verifier-guided (2026-06-22)

> **Estado**: PRE-REGISTRO formal de los experimentos que el usuario aprobó correr
> (OQ1 causal + funcional; verifier-guided de-risk). Fija métricas, kill-rules y
> márgenes ANTES de ver datos (regla anti-p-hacking, research-methods §4). Escrito en
> la sesión ClaraVT 2026-06-22 tras la auditoría de estado real del repo.
>
> **Decisión del usuario que enmarca esto**: avanzar con OQ1 + verifier-guided;
> **eval-bulletproof queda para el próximo pod**; process-LoFiT GATEADO por OQ1-funcional.
> Norte DATA-FIRST ([[project_romulo_central_vision]]). Master de contexto:
> `2026-06-22-PROJECT-STATE-UNIFICATION-technical-and-roadmap.md`.

---

## 0. Qué de-riskea cada experimento (por qué estos, en este orden)

| # | Experimento | Pregunta | De-riskea | Estado de código | Costo |
|---|---|---|---|---|---|
| E1 | **OQ1-causal** | ¿el spillover es separable particionando heads? | cierra causalmente "disentanglement por partición" + §5 del paper | **LISTO + commiteado** (`db84581`) | €8-12 (400) / ~€20-30 (1000) |
| E2 | **OQ1-funcional** | ¿`v_inf ⊥ v_ret` en activaciones base? | **gate de process-LoFiT** (separable por proyección ⇒ process-LoFiT tiene fundamento) | scripts A ESCRIBIR; infra existe | €5-12 |
| E3 | **VG de-risk** | ¿el scorer rankea trazas *within-prompt*? | mata/habilita el verifier-guided **antes** de los 4 brazos | scripts A ESCRIBIR; no existe nada | €4-10 |
| E4 | VG full + process-LoFiT | (gateado) | — | NO se diseña hasta E2/E3 | — |

Regla de asimetría ([[project_romulo_central_vision]]): E3 ataca el pilar débil
(generación). Regla de falsación barata (ClaraVT): E3 es el test más barato que puede
tumbar la pata más segura; correrlo antes que los 4 brazos completos.

---

## 1. E1 — OQ1 causal (ablación de heads de D.1)

**Diseño (ya implementado, `scripts/run_oq1_ablation_pod.sh`).** 4 arms, eval
MMLU(spillover) + ARC(inference transfer) + social_i_qa(in-domain):
- `NULL` = full blob + `--null-adapter` → **GATE** (debe reproducir base MMLU ≈ 0.8306 ±2pp; auto-aborta si no).
- `FULL` = 24 heads → referencia del daño (canónico MMLU 0.6347 = −19.59pp).
- `ZEROHI` = 12 heads **high**-retrieval-AUC en cero (quedan 12 low-retr).
- `ZEROLO` = 12 heads **low**-retrieval-AUC en cero (quedan 12 high-retr).

Comparación decisiva: **ZEROHI vs ZEROLO** (no vs FULL), cruzada MMLU↔ARC.

**El problema metodológico (predicción = null) y su fix.** El finding €0 predice
`zerohi ≈ zerolo` (co-localización). Un test de diferencia (`|z|>2`) que NO rechaza
NO prueba equivalencia — puede ser co-localización **o** falta de poder. A n=400,
`SE_diff(MMLU) = √(2·0.022²) ≈ 3.1pp` → 90% CI de la diferencia ≈ ±5.1pp, que **no
cabe** en un margen de equivalencia de ±3pp. **Conclusión: para concluir
co-localización hace falta un TOST con n≈1000 en MMLU.**

**Pre-registro (fijado ANTES de ver datos):**
- **Margen de equivalencia** `Δ_eq = 3pp` en MMLU (justificación: ≈ 1×SE a n=400; el
  propio script lo nombra como "within noise"). Pre-registrado, no post-hoc.
- **TOST (two one-sided tests)**: concluir EQUIVALENCIA (⇒ co-localizado, NO separable)
  si el **90% CI de (MMLU_zerohi − MMLU_zerolo) ⊂ [−3pp, +3pp]**.
- **Verdict map** (crossed MMLU↔ARC):
  - equiv. en MMLU **y** ARC → **CO-LOCALIZADO / no separable** (predicción; confirma finding → "disentanglement por partición" refutado causalmente).
  - `ΔMMLU > 3pp` a favor de zerohi **sin** costo ARC proporcional → **SEPARABLE** (contradice; reabre).
  - `ΔMMLU` grande **con** `ΔARC` proporcional → **ACOPLADO** (co-localizado por otra vía).
  - CI ni rechaza diferencia ni concluye equivalencia → **INCONCLUSO por poder** → escalar.
- **Plan de escalado (pre-registrado, no es p-hacking):** correr E1 a `LIMIT=400`
  primero (barato, da ARC/siqa donde sí esperamos señal). Si `|ΔMMLU| < 3pp` pero el
  90% CI **no** cae dentro de ±3pp → re-correr **MMLU-only a `LIMIT=1000`** (4 arms).
  Alternativa: correr directo a `LIMIT=1000` si se prefiere un solo paso concluyente.
- **Poder de equivalencia a n=1000:** `SE_diff(MMLU) ≈ 1.7pp` → 90% CI ≈ diff ±2.8pp
  → cabe en ±3pp si diff≈0. Suficiente.

**Métrica primaria**: equivalencia/diferencia de MMLU(zerohi vs zerolo). Secundaria:
ARC (¿el lift de inferencia se conserva al ablacionar?), recovery vs FULL.

**Kill / lectura**: el verdict NO depende de "reproducir −19.59pp exacto" (FULL es
referencia, no kill). El GATE (NULL ≈ base) sí es kill: si falla, wrapper roto, todo aborta.

**Análisis**: `scripts/analyze_oq1_causal.py` (parcheado esta sesión con el TOST). Local €0.

---

## 2. E2 — OQ1 funcional (ángulo de direcciones, gate de process-LoFiT)

**Pregunta**: ¿las direcciones funcionales `v_inf[h]` (discrimina correcto-vs-incorrecto
en inference-form) y `v_ret[h]` (idem retrieval-form) son ortogonales / separables, o
comparten dirección? Es el test directo del §5 ("same heads carry orthogonal directions")
y el **gate de process-LoFiT** (orthogonal-decomposition loss sólo tiene sentido si las
direcciones son separables por proyección).

**Operandos (pre-registrados):**
- **Inference-form**: ARC-Challenge + HellaSwag (builders existen en
  `probe_v7_lofit_head_selection.py`: `build_arc_pairs`, `build_hellaswag_pairs`).
  *Nota*: Winogrande NO tiene builder — queda opcional, no bloquea.
- **Retrieval-form**: KR factual MC — SciQ + MedMCQA (retrieval limpio: lookup de hecho
  almacenado). Fallback si JSONL ausentes: TruthfulQA mc1 (`build_tqa_pairs`, vía HF;
  caveat: TQA es retrieval-ish/inference-ish, marcar como sensitivity-check, NO primario).
- **Dirección funcional**: `v[h] = fit_massmean(H_correct[:,L,h,:], H_wrong[:,L,h,:])`
  (mass-mean, Marks&Tegmark/ITI; misma función que V7 usa). Geometría = **ph_last**
  (per-head, last token) para alinear con la semántica de captura de V7.
- **Timing**: INFERENCE-TIME (forward de scoring, captura per-head vía o_proj pre-hook),
  NO post-gen. El `correctness_probe_postgen` NO sirve (es post-gen).
- **Heads elegibles**: aquellos con señal en AMBAS familias (probe acc ≥ 0.60 en inf y ret).

**Mediciones:**
1. `cos(v_inf[h], v_ret[h])` **crudo Y whitened** (Mahalanobis bajo covarianza base del
   head, `park_whiten`). El whitened es el que importa (el crudo vive en base no-ortonormal).
2. **Null**: permutación de etiquetas (recomputar dirección con labels barajados) — más
   potente que random-unit en high-dim. + BCa bootstrap CI sobre la media de cos.
3. **Cross-check del puente §134**: `cos(v_inf[h], θ_mc[h])` — ¿la dirección funcional de
   inferencia coincide con el offset entrenado del adapter mc? Valida que el offset ≈
   inference-direction shift (hoy sin verificar).
4. **Reportar condicionamiento** del whitening (eigvals near-zero suprimidos) — high-dim
   (head_dim 256/512) low-n (~600 pairs) puede ser ill-conditioned.

**Kill-rules / verdict (pre-registrado, del design doc `2026-06-22-oq1-functional-axis-angle-design.md`, refinado):**
- `cos_whitened` mean **dentro del null** (≈ 0) → **ORTOGONALES**: §5 sobrevive;
  disentanglement por proyección **viable en principio** → **process-LoFiT GATE = VERDE**.
- mean **>> 0** (alineadas, BCa LB > 0) → §5 falso / puzzle: afilar una mejora la otra,
  contradice el spillover → **reescribir §5**; process-LoFiT por orthogonal-decomp dudoso.
- mean **<< 0** (anti-colineal, BCa UB < 0) → un solo eje funcional → **disentanglement
  imposible por construcción** → **process-LoFiT GATE = ROJO** (matar antes de gastar corpus).

**Gotchas (de la auditoría, OBLIGATORIO respetar):**
- **head_dim heterogéneo en Gemma 4 31B** (sliding 256 / full 512): usar
  `_get_attention_shape_runtime` + `layer_dims` para saltar capas de dim no-dominante.
  NO asumir head_dim uniforme.
- **chat-template** consistente con V7 (IT model): `apply_chat_template(...,
  add_generation_prompt=True)` + answer raw, captura en last token. Mismo patrón que
  `probe_v7_lofit_head_selection.py` líneas 804-824.
- **Alineación (layer,head)** entre v_inf, v_ret y θ_mc: clave = tupla (layer, head);
  todas las extracciones del MISMO run para garantizar mismo `captured_indices`.

**Análisis**: local €0 (CPU). **Extracción**: pod, ~15-30 min/familia.

---

## 3. E3 — Verifier-guided de-risk (within-prompt ranking)

**El punto ciego que esto resuelve (verificado en la auditoría):** el post-gen
correctness probe tiene AUC 0.94-0.96 **POOLED between-prompt** (`postgen_*.pt` es
`[N,2,d]`, 1-respuesta-por-prompt). El verifier-guided necesita algo DISTINTO: dado K
trazas del **mismo** prompt, ¿el scorer rankea la correcta arriba? Un AUC pooled de 0.95
es compatible con ranking within-prompt ≈ azar. **Ese número no está medido.** E3 lo mide.

**Diseño:**
- **Generación (pod)**: BASE genera **K=8 trazas CoT** por prompt (`do_sample=True`,
  `temperature≈0.8`, `num_return_sequences` o loop), `max_gen_toks=2048` (gsm8k mostró
  cap-runaway). Tareas: **gsm8k** (generativo, pilar débil) + **analytic_entailment**
  (inference-dominant, CoT → letra). Guardar (trace_text, sum_logprob, parsed_answer,
  correct∈{0,1}) por traza.
- **Scoreo (local €0)**: para cada traza, 3 scorers:
  - (s1) **post-gen correctness probe** (re-entrenar el LogReg sobre el corpus viejo
    `runs/correctness_probe_postgen/`, aplicar a las activaciones de cada traza nueva,
    extraídas con `correctness_probe_postgen_extract.py`).
  - (s2) **base logprob** de la traza (baseline — self-consistency/length-norm).
  - (s3) **adapter loglik** de la respuesta parseada (la fortaleza scoring-face del adapter).
- **Métrica primaria (within-prompt)**: para cada prompt con ≥1 traza correcta y ≥1
  incorrecta (prompts "discriminables"), ¿la traza top-scored por s_i es correcta?
  Reportar **selection accuracy** (top-1 correcta) y **within-prompt ranking AUC** (cada
  scorer vs el orden de correctness), por scorer.
- **Baselines de comparación**: random pick (= tasa base de correctas), majority-vote
  (self-consistency), oracle (best-possible si eligiéramos siempre una correcta cuando existe).

**Kill-rule (pre-registrado, no apelable):**
- VG **paga** si **algún** scorer supera al random-pick **y** al base-logprob (s2) en
  selection accuracy, con **CI_lo > 0** sobre la diferencia, **n ≥ 80 prompts
  discriminables** (los nulls previos murieron underpowered a n=62).
- Si ningún scorer supera el logprob baseline within-prompt → **VG NO paga** → negativo
  first-class al paper; NO se construyen los 4 brazos (E4).
- **Estratificar por upstream-correctness** (lección [[feedback_nested_b2_leg_buried_2026_06_22]]):
  el adapter es premise-faithful (propaga fiel la premisa errada). Medir selection
  accuracy **condicional** (prompts premise-correct vs premise-error), no marginal — el
  scorer-adapter puede preferir trazas internamente coherentes pero factualmente erradas.

**Por qué esto es el de-risk, no el experimento final:** si el scorer rankea
within-prompt, el VG full (4 brazos con route-out + base emite) está justificado (E4).
Si no, lo matamos por €4-10 en vez de €6-12 + ingeniería de los 4 brazos.

**Corpus / n≥80**: gsm8k tiene 400 en el corpus postgen; analytic_entailment 70 +
abstract_narrative 200. Los prompts "discriminables" (mezcla de trazas correctas e
incorrectas) son un subconjunto; con K=8 y acc≈0.85, habrá suficientes discriminables en
gsm8k (acc 0.905 deja ~10% wrong × varianza por traza). Si analytic_entailment solo no
alcanza n≥80 discriminables, poolear con abstract_narrative (lección corpus-n).

---

## 4. E4 — Gateado (NO se ejecuta ni se diseña en detalle ahora)

- **Verifier-guided FULL** (4 brazos: BASE_WHOLE, **BASE_COT_DECOMP** [control crítico,
  lección H-anidada], PHASE_SPLIT, VERIFIER_GUIDED): sólo si E3 PASA. Métrica
  `A(brazo) − A(BASE_COT_DECOMP)`, kill-rule CI_lo>0 n≥80, estratificado upstream-correctness.
- **Process-LoFiT** (entrenar dirección sobre contrastes de paso-de-razonamiento): sólo
  si E2 da ORTOGONAL/VERDE **y** E3 no saturó ya la cosecha de generación. Requiere
  construir corpus PRM-style (good-step vs bad-step) — **costo real = semanas, no €-pod**;
  EOS-collapse NO desaparece por construcción (medir EOS smoke). Conecta
  [[backlog_crest_generation_axis_rescue_post_paper]] (trigger formal = post-paper-v1).

---

## 5. Reglas transversales (research-methods + lecciones del repo)

- **Pre-registro antes de datos** (research-methods §4): márgenes, kill-rules y métricas
  de §1-§3 fijados acá, ANTES de correr. No se ajustan post-hoc.
- **Control de explicitación** (lección H-anidada): cualquier lift se mide contra el
  control que descompone/explicita SIN routing — ~92% del "lift por pensar más" es
  explicitación. Aplica a E4, no a E1-E3.
- **Deltas condicionales, no marginales** (lección nested-B.2): estratificar por
  upstream-correctness en E3/E4.
- **Progress logging OBLIGATORIO** (regla user): todo loop largo imprime `[i/N]` con
  `flush=True`; escritura incremental con `f.flush()`.
- **Pod**: cada bloque self-contained con `cd /workspace/MSAP` + `pwd`; `nohup ... &
  disown`, NUNCA `tee|pipe`; `mkdir -p` del dir de salida antes del redirect; BASE real
  (no placeholder); `hf auth login` para backup.
- **Apples-to-apples**: config byte-idéntica entre base y adapter; chat-template
  consistente; `--log-samples` donde haya generación.

## 6. Artefactos

- **E1 (listo)**: `scripts/{make_oq1_ablation_variants.py, run_oq1_ablation_pod.sh,
  analyze_oq1_causal.py}` (+ TOST patch esta sesión).
- **E2 (a escribir)**: `scripts/extract_functional_axis_perhead.py` (pod),
  `scripts/probe_oq1_functional_angle.py` (local).
- **E3 (a escribir)**: `scripts/gen_ktrace_pod.py` (pod, gen+extract),
  `scripts/analyze_verifier_within_prompt.py` (local).
- **Reuso**: `probe_v7_lofit_head_selection.py` (fit_massmean, park_whiten,
  capture_per_head_activations, DATASET_LOADERS), `correctness_probe_postgen_extract.py`
  (residual extract), `correctness_probe_postgen_analyze.py` (LogReg + CV + bootstrap).
- **Inputs canónicos**: `runs/phase3_adapters/socialiqa/offsets_correctness.pt`,
  `runs/v7_lofit_gemma4_31b_chat/head_probe_completion.json`, `runs/kr_h5/acts_*.pt`,
  `runs/correctness_probe_postgen/`.
