# ⚠️ DRAFT — pre-Fase-4 — Pre-registro: causal-del-offset (¿el +35 es correctness o un hack OOD?) (2026-06-23)

> **Estado**: PRE-REGISTRO (métricas, kill-rules y construcción fijadas ANTES de datos).
> Origen: E2 (OQ1-funcional) + la caracterización de `theta_mc` (2026-06-22/23), que
> mostraron que el offset entrenado es **ortogonal a la dirección funcional de
> correctness** (`cos_whitened(theta_mc, v_inf) = −0.003`, dentro del null) y un empujón
> **off-axis grande** (norma 4.3× la separación correct/wrong, baja-varianza, no
> mean-shift). Artefactos: [[feedback_e2_functional_axis_aligned_2026_06_22]],
> `runs/oq1_functional_axis/{functional_directions.pt, theta_mc_characterization.json}`.

## 0. La pregunta (más grande que process-LoFiT-PRM)

El adapter mc lifts ARC +35 (scoring-face) y daña MMLU −19.59pp. ¿Ese lift es **afilar
la dirección funcional de correctness** (recuperable por un offset *on-axis* ∝ v_inf), o
un **hack OOD** que sesga el loglik por una dirección off-axis específica que el
training contrastivo encontró? La caracterización dice que `theta_mc ⊥ v_inf` — así que
si el lift fuera correctness, un offset on-axis debería darlo. Este experimento lo decide
causalmente, y de paso gatea process-LoFiT-PRM.

## 1. Diseño

**Construcción del offset on-axis (€0, `scripts/make_vinf_offset.py`):** para cada head
(L,h) del adapter mc (`offsets_mc.pt`, 48 pares), tomar la dirección funcional de
inferencia `v_inf[L,h]` (de `functional_directions.pt`, E2), normalizarla a unidad, y
escalarla a la **misma norma** que el offset mc aplicado en ese head:
`offset_vinf[L,h] = ‖α_mc·θ_mc[L,h]‖ · v_inf_unit[L,h]`, con signo **+v_inf** (hacia
"correcto"). Esto aísla la DIRECCIÓN: mismo tamaño de empujón que mc, on-axis (v_inf) vs
el off-axis actual. Se guarda como blob byte-compatible con `install_lofit_hooks`
(`alpha = ‖offset_mc‖`, `theta = v_inf_unit`, mismos `layer_head_pairs`).

**Brazos (eval ARC + MMLU + lambada + TQA-mc1, `--limit 400`, chat-template):**
- `NULL` — `offsets_mc + --null-adapter` → GATE (reproduce base MMLU ~0.8306; auto-abort si no).
- `MC` — `offsets_mc` → referencia (debe reproducir +35 ARC / −19.59 MMLU).
- `VINF` — `mc_vinf_offset` → el test (mismo norm, dirección v_inf).

**Métricas primarias:** `ΔARC(VINF) − base` (¿hay lift on-axis?) y `ΔMMLU(VINF) − base`
(¿hay daño?), comparados contra `MC`. Secundarias: lambada (retrieval/gen daño), TQA.

## 2. Kill-rules / verdict (pre-registrado)

Sea `Lift = ΔARC vs base`, `Harm = base − MMLU`. Comparaciones con stderr de lm-eval
(diferencias > ~3pp a n=400 son señal; <3pp ruido — escalar a 1000 si borderline).

- **VINF Lift ≈ MC Lift (CI overlap) Y VINF Harm < MC Harm** → **process-LoFiT-PRM VIVO**:
  la dirección on-axis afila igual con MENOS spillover. El lift ES correctness-direction-
  sensitive; alinear el offset con v_inf es el camino. (Gate de process-LoFiT = VERDE causal.)
- **VINF Lift << MC Lift** (p.ej. < 50% del lift de MC) → **el +35 es un HACK OOD**: la
  dirección de correctness NO da el lift; el offset off-axis específico es necesario.
  process-LoFiT-PRM **muerto**, PERO se gana el **mecanismo unificado** (las tres
  patologías — D5 no-transfiere, spillover, EOS-collapse — = un empujón OOD). Paper-fuerte.
- **VINF Lift ≈ MC Lift Y VINF Harm ≈ MC Harm** → **ACOPLADO**: lift y daño van juntos
  sin importar la dirección → el spillover es intrínseco al empujón de esa magnitud, no a
  su dirección. Disentanglement por dirección refutado.
- **VINF Lift negativo / ruido** → la dirección v_inf no modula el scoring (posible: v_inf
  de baja calidad en los heads del adapter; ver caveat) → inconcluso, revisar acc_inf.

## 3. Caveats (pre-registrados, Metodólogo)

- **Calidad de v_inf en los heads del adapter**: v_inf se midió sobre inference-form
  (ARC+HellaSwag), dominio **compatible** con el training de mc (TQA+ARC+HSwag+Wino). Pero
  no todos los 48 heads del adapter tienen acc_inf ≥ 0.6 — el offset on-axis en heads con
  v_inf ruidoso es ruido. `make_vinf_offset.py` reporta cuántos heads tienen acc_inf < 0.6;
  el análisis estratifica/reporta el verdict con y sin esos heads.
- **Norm-match aísla dirección, no magnitud**: si el lift dependiera de una magnitud mayor
  que ‖offset_mc‖, un VINF norm-matched no lo vería. Por eso, si VINF Lift es chico,
  un sweep de escala (`--scale 1,2,4`) es el follow-up antes de declarar "OOD-hack".
- **Signo**: +v_inf (hacia correcto). Si VINF da daño-sin-lift, probar −v_inf como sanity.
- **n=400 poder**: stderr ARC/MMLU ~2.2-2.5pp. Diferencias de lift grandes (esperadas si
  OOD-hack) son detectables; un lift parcial requiere escalar a 1000.
- **Las 10 capas full-attention** quedan fuera de v_inf (head_dim heterogéneo); los 48
  heads del adapter están todos en la grid capturada (E2 lo verificó: 48/48 aligned).

## 4. Pasos

**Pod (GPU, comparte bring-up con E3/E1):**
```bash
cd /workspace/MSAP && git pull
nohup bash scripts/run_vinf_pod.sh > runs/vinf_causal/vinf.log 2>&1 & disown
tail -f runs/vinf_causal/vinf.log     # esperá "NULL GATE PASSED"
```
**Local (€0):** `python -m scripts.analyze_vinf_causal --dir runs/vinf_causal`.

**Costo:** ~3 brazos × 4 tasks @ 400 ≈ €10-15. Comparte el bring-up del causal/E3.

## 5. Artefactos
- `scripts/make_vinf_offset.py` (construye el offset on-axis, €0 CPU).
- `scripts/run_vinf_pod.sh` (3 brazos, NULL-gate, eval, backup HF).
- `scripts/analyze_vinf_causal.py` (verdict local €0).
- Inputs: `runs/v7_lofit_gemma4_31b_chat/offsets_mc.pt` (committed),
  `runs/oq1_functional_axis/functional_directions.pt` (E2, en pod / HF).
- Salidas: `runs/vinf_causal/{mc_vinf_offset.pt, d_{null,mc,vinf}.json}`.
