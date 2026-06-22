# Pre-registro — Batería de robustez de evaluación "bullet-proof" (2026-06-22)

**Propósito**: pre-registrar, ANTES de los datos, las mediciones que hacen
bullet-proof los 6 números estrella del paper del eje + el control de mecanismo,
con la condición `bulletproof-iff / retract-if` por número. Diseño DATA-FIRST por
directiva del user (2026-06-22): *"cualquier otra cosa necesaria para que nuestros
números estén bullet proof"*. Origen: workflow adversarial de 8 lentes (71 ataques)
→ síntesis. Disciplina D4 (kill-rules pre-data). **Nada se mide antes de fijar este
doc.**

## Decisiones de scope (locked por el user, 2026-06-22)

1. **Align-LoRA control = IN** (train + eval). SFT no-contrastivo sobre los MISMOS
   datos que V7-mc. Es el ataque P0 al mecanismo (arXiv:2508.05078): sin él, el
   headline es "el fine-tuning ayuda"; con él, "el EJE CONTRASTIVO ayuda".
2. **MMLU = stratified ~2k** (57 subjects), no full 14042. CI ±1.5pp basta para un
   claim de no-regresión (A8 "no por debajo de base", no es win headline).
3. **EOS 60→4 = re-medido en bundle** (misma sesión de pod; amortiza bring-up;
   cierra el único número ASSERTED del paper).

## Gates locales pre-pod — RESULTS (2026-06-22, €0, antes de tocar el pod)

Corridos ANTES del pod (son inputs, no outcomes bajo prueba). Ambos PASS:

- **L6 (provenance de offsets_mc.pt) — PASS.** El blob tiene `config.datasets =
  ["tqa", "arc"]` (+ top_k=48, steps=800, lr=0.01, loss=direct). **HellaSwag y
  Winogrande NO están en el entrenamiento** → el claim de "+17 HSwag / +19 Wino
  clean transfer" queda confirmado A NIVEL DE ARTEFACTO, no asumido. La condición
  `bulletproof-iff` de contaminación de HSwag/Wino: SATISFECHA. Bonus: el R11 smoke
  tiene un valor esperado para asertar — 48 (layer,head) pairs en 12 capas
  [30-34,36-40,49,55]. Script: `scripts/analyze_offsets_provenance.py`.
- **L5 (dedup ARC-Challenge train/test) — PASS (CLEAN).** test n=1172 vs train
  n=1119: 7 exact-stem (0.60%), 0 near-dups (Jaccard≥0.9). 0.60% < umbral 1% → el
  lead +35.0 ARC es seguro; se nota el 0.60% en limitations. Script:
  `scripts/analyze_arc_contamination.py`.

Consecuencia: dos de los seis números headline (HSwag +17, Wino +19) tienen su gate
de contaminación CERRADO pre-pod; el +35 ARC sobrevive su gate de leakage. El pod ya
no carga el riesgo de contaminación sobre esos tres.

## Convención de split canónica (de `train_v7_lofit.py:113`, TRAIN_FRAC=0.8)

- **TQA**: `truthful_qa/multiple_choice/validation` (817 items). `perm =
  torch.randperm(len(ds), generator=Generator().manual_seed(0))`. **split='train'
  = primeros `int(n_total*0.8)` del orden permutado** (lo que vio V7-mc);
  split='test' = resto. `n_total` es post-filtro del builder.
- **ARC**: `ai2_arc/ARC-Challenge`, split nativo `train` (entrenamiento) vs `test`
  (eval) → **DISJOINT por construcción**.
- **Replay obligatorio**: el corpus del control, la partición L4 y el held-out R5
  **importan `build_tqa_pairs`/`build_arc_pairs` de `train_v7_lofit.py`** (no
  reimplementan) → identidad byte-a-byte garantizada.

## El control de mecanismo (Align-LoRA matched)

- **Datos**: corpus SFT = `build_tqa_pairs(split='train', seed=0)` (prompt→answer)
  + `build_arc_pairs(split='train', seed=0)`. **MISMOS datos que V7-mc**, NO la
  unión de 4 tasks del default de `train_align_lora.py` (eso confundiría con
  MMLU/GSM8K/SocialIQA).
- **Objetivo**: SFT next-token CE (prefix-masked), r=256, q/k/v/o. Difiere de V7-mc
  SOLO en el objetivo (SFT vs offsets contrastivos de cabeza), no en los datos.
- **Budget**: comparable (r256, 2 epochs, fit convergido). "Matched" = mismos datos
  + budget razonable, NO step-count byte-idéntico (documentar config; el train-loss
  convergido y el eval hablan).
- **Eval del control**: SOLO donde V7-mc se compara — R1 (MC trío 0-shot n=400),
  R6 (gsm8k), R10 (MMLU strat + GPQA), y TQA held-out (vía R5). NO toda la batería
  (acota costo). TQA del control se evalúa en el MISMO held-out 20% (el control
  también vio TQA-train-80% → su eval limpio es el held-out, igual que V7-mc).
- **Lectura**: si Align-LoRA(TQA+ARC, SFT) NO replica el +35/+37 ni la firma de
  spillover → el lift es del objetivo contrastivo, no de "ver los datos". Si SÍ los
  replica → el paper se reencuadra a "fine-tuning ayuda / Pareto" (data-first).

## Matriz de runs de pod

Runner: `scripts/run_lm_eval_v7.py` (V7HFLM, path de scoring estándar de HFLM).
Config base compartida salvo lo indicado: bf16, `max_length=4096`,
`apply_chat_template=True`, ambos/los brazos config byte-idéntica, `--log-samples`.
Arms: `base` (hf), `v7mc` (offsets_mc.pt), `null_adapter` (v7hf con alpha=theta=0),
`align_lora` (LoRA SFT matched).

| run | tasks | arms | n-shot | n | template | objetivo / defusa |
|---|---|---|---|---|---|---|
| **R1** | arc_challenge, truthfulqa_mc1, hellaswag, winogrande | base, v7mc, **null_adapter**, **align_lora** | 0 | 400 (mismos doc_ids) | on | reproduce canónico; null_adapter prueba que el lift es offsets no wrapper; align_lora = control mecanismo |
| **R2** | arc_challenge, truthfulqa_mc1, hellaswag, winogrande | base, v7mc | 5 | 400 | on | 2×2 shots; base@5-multiturn = base honesto más fuerte (`fewshot_as_multiturn=True`) |
| **R3** | arc_challenge, truthfulqa_mc1 | base | 5 | 400 | on | `fewshot_as_multiturn=False` (concat) → mostrar base@5-multiturn ≥ base@5-concat (control de DoF) |
| **R4** | arc_challenge, hellaswag, winogrande, truthfulqa_mc1 | base, v7mc | 0 | **full** (1172/10042/1267/817) | on | de-subsampleo; el delta full debe caer dentro del CI bootstrap de n=400. TQA-817 alimenta L4/R5 |
| **R5** | truthfulqa_mc1, truthfulqa_mc2 | base, v7mc, **align_lora** | 0 y 5 | full 817 (partición local a held-out) | on | contaminación TQA: el headline = delta sobre el held-out 20% (seen via L4); mc2 contra metric-gaming |
| **R6** | gsm8k | base, v7mc, **align_lora** | 5 | full 1319 | on | barrido `max_gen_toks {512,1024,2048}`; strict+flexible; ¿el −25 es truncación de CoT? |
| **R7** | gsm8k | **null_adapter** | 5 | 400 | on | el wrapper v7hf es identidad en generación (debe == base 0.9775) |
| **R8** | gsm8k | base, v7mc | 5 | 400 | **off** | el −25 persiste sin chat-template (no es artefacto de scaffold) |
| **R9** | lambada_openai | base, v7mc | 0 | full 5153 + 400 | **off** | reemplaza el scorer custom por lm-eval estándar; cross-valida base ~0.64; scorer custom → apéndice |
| **R10** | mmlu (stratified ~2k, 57 subj) , leaderboard_gpqa_diamond (198) | base, v7mc, **align_lora** | 0 | strat ~2k / full 198 | on | re-baseline de recall/spillover bajo config EXACTA del trío (hoy base ~0.83 sin provenance) |
| **R11** | arc_challenge (8) + gsm8k (8) | v7mc (instrumentado) | 0 | 8 | on | dump diagnóstico: pares post-filtro, capas sobrevivientes, `len(_v7_handles)`, fire-counts |
| **R12** | arc_challenge, truthfulqa_mc1 | v7mc | 0 | 400 | on | (a) seed distinto → acc_norm byte-idéntico (determinismo MC); (b) `--dtype float32` → \|Δ\| vs bf16 <1pp |
| **EOS** | `eval_adaptive_verbosity.py --label v7mc --include-social` | base, v7mc | — | smoke | — | re-medir el 60→4 pod-lost (bundle) |

## Análisis locales (€0, sin GPU — corren en WSL post-pod)

- **L1** Significancia pareada: alinear base/adapter por `doc_hash`, McNemar exacto +
  bootstrap pareado (10k reps) → Δ ± CI95 y p por task. **Precondición**: assert
  `doc_hash` set base == adapter por task (si falla, el framework pareado es nulo).
- **L2** Tabla acc + acc_norm (ambos en el artefacto): mostrar que acc_norm es la
  elección conservadora y ambos coinciden en signo.
- **L3** Clasificación de fallos gsm8k: separar terminated-sin-marcador vs hit-cap.
  (Evidencia parcial ya: 27/400 no-#### son largas, 0 vacías → verbosidad, no
  EOS-a-vacío.)
- **L4** Partición contaminación TQA: replay `build_tqa_pairs(split='train',seed=0)`,
  etiquetar cada eval doc SEEN/UNSEEN, recomputar Δ por estrato.
- **L5** Dedup ARC train/test (Jaccard/MinHash sobre stems normalizados) → % de test
  con near-twin en train. Gate del lead headline.
- **L6** Provenance de `offsets_mc.pt`: dump de manifest/training-config; confirmar
  pares de entrenamiento = SOLO TQA+ARC (excluye hellaswag/winogrande/gsm8k).
- **L7** Lockfile reproducibilidad: SHA del checkpoint base + sha256 de
  safetensors/tokenizer/offsets; pin de revisiones HF; freeze pip (lm_eval, transformers);
  hash de `tokenizer.chat_template`. Cierra el `model_sha=''` real del artefacto.
- **L8** Audit de prompt renderizado: dump byte-idéntico base vs adapter (solo hooks
  difieren); fracción con len>2048 (sin truncación asimétrica a max_length=4096).
- **L9** Unit tests de hooks (local, sin pod): offsets distintos por hook (no
  aliasing); 1 hook por o_proj sobreviviente / 0 en no-sobrevivientes; identity con
  alpha=theta=0 bit-a-bit.
- **L10** Corrección de multiplicidad: Holm-Bonferroni sobre TODOS los McNemar p del
  set declarado (6 headline + mmlu + gpqa). Reportar p corregido; NINGÚN task dropeado.
- **L11** Determinismo del filtro: set (layer,head) sobreviviente byte-idéntico entre
  n=400 y full y entre probes repetidos → el de-subsampleo mantiene el adapter fijo.

## Pre-registro por número (bulletproof-iff / retract-or-hedge-if)

- **ARC +35.0 (acc_norm 0.505→0.855)** — BULLETPROOF-IFF: L5 near-dup <1%; R1
  null_adapter reproduce base 0.505 ±2pp; R4 full-1172 Δ dentro del CI de n=400; R2
  v7mc@0 − base@5-multiturn sigue CI-separado >+20pp; L2 acc y acc_norm ambos
  positivos; L1 McNemar p<<0.001. RETRACT/HEDGE-IF: near-dup >3-5% → re-eval dedup;
  o base@5 a ~5pp de v7mc → "+X sobre 0-shot, matcheado a 5-shot"; o null_adapter ≠
  base → Δ vs null_adapter.
- **TQA +37.25 (mc1)** — BULLETPROOF-IFF (held-out es el headline): R5 Δ held-out
  20% ≥+15pp con CI sin 0; R5 mc2 mismo signo y >+10pp; base@5 en tail no cierra el
  gap; L4 gap SEEN-vs-UNSEEN modesto. Reportar el número held-out (menor) como
  headline + apéndice de memorization-gap. RETRACT-IF: Δ held-out <~+5pp o CI
  incluye 0 → **retirar la generalización de TQA, liderar con ARC/HSwag/Wino**; o
  mc2 ~0/neg → reframe a "selección mc1-específica, no truthfulness".
- **HellaSwag +17.0 (acc_norm)** — BULLETPROOF-IFF: L6 provenance excluye hellaswag;
  R4 full-10042 Δ dentro del CI de n=400; R1 null_adapter reproduce 0.54; L2 acc
  (+22.25) ≥ acc_norm (+17.0) (no artefacto de longitud); R2 sobrevive base@5.
  RETRACT-IF: hellaswag en el mix de entrenamiento (L6) → no es transfer; o full-set
  Δ deriva >CI; o base@5 cierra el gap.
- **Winogrande +19.0 (acc)** — BULLETPROOF-IFF: L6 excluye winogrande; R4 full-1267
  dentro del CI; R1 null_adapter reproduce 0.6575; R2 sobrevive base@5; L1 McNemar
  p<<0.001. RETRACT-IF: winogrande en el mix; o Δ full fuera del CI; o base@5 cierra.
- **gsm8k −25.0 (strict)** — BULLETPROOF-IFF: R6 acc de v7mc PLANA en cap
  {512,1024,2048} (<2pp) Y cap-hit-rate ≈ base Y subset both-terminated Δ ≥15pp; R6
  flexible ~ strict (−24 vs −25); R7 null_adapter == base 0.9775; R8 persiste con
  template off; L3 las no-#### son CoT largo genuino. RETRACT/REFRAME-IF: acc de
  v7mc SUBE con cap (cierra el −25) o cap-hit-rate >> base o both-terminated Δ <5pp
  → **reframe a "verbosidad bajo presupuesto fijo", NO pérdida de razonamiento**.
- **lambada −32 (0.64→0.32)** — BULLETPROOF-IFF: R9 lm-eval lambada_openai
  (template off) muestra drop del mismo signo, base ~0.64 reproducido ±2pp,
  custom-vs-lm-eval coinciden ±2pp en 400 docs compartidos → scorer custom al
  apéndice, headline usa el número lm-eval. RETRACT-IF: lm-eval no muestra drop o
  <5pp o signo discrepa → **DROP lambada del trío headline** (el trade-off
  inf/retr queda solo en gsm8k; custom number "ilustrativo").

## Mecanismo (Align-LoRA control) — pre-registro

- BULLETPROOF-IFF (eje contrastivo): align_lora(TQA+ARC, SFT) en R1 da un lift MC
  **materialmente menor** que v7mc (ej. <+10pp donde v7mc da +35) Y/O spillover de
  recall (R10) cualitativamente distinto. → "el lift es del objetivo contrastivo".
- REFRAME-IF: align_lora replica el lift MC y la firma de spillover dentro de ruido
  → el claim de mecanismo cae; el paper se reencuadra honesto a Pareto/fine-tuning
  (data-first; ya contemplado como rama válida).

## Costo + ejecución

- ~16-19 GPU-h (eval + train Align-LoRA + EOS smoke + MMLU strat) ≈ **€45-60** en
  RunPod community Blackwell. Pod destruido → ~15 min bring-up (runbook canónico).
- **Un solo batch nohup'd, front-loaded** (regla pod: bloque self-contained, GPU-only,
  progreso por iteración). Build del corpus del control = local/prelude determinista.
- Gotchas: `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` + per-item
  `empty_cache()` para gen-heavy (vocab 256k OOM en 4070, pero en 96GB holgado);
  `--max-length 4096` (Gemma 4 nested cae a 2048); `--gen-kwargs max_gen_toks=N`
  para el barrido R6; nohup+disown (NUNCA `tee|pipe`, SIGPIPE).

## Artefactos esperados

`runs/eval_bulletproof/` (canónico): por-run JSON con `--log-samples`; corpus del
control `runs/align_lora_control/corpus_matched.jsonl`; adapter
`runs/align_lora_control/r256/`; análisis locales en
`runs/eval_bulletproof/analysis/`. Backup HF al cierre. Cada número del paper que
sobreviva → entra al CLAIMS LEDGER con su provenance bulletproof; cada uno que caiga
→ se retira/reframe per este pre-registro (sin apelación).

## Scripts a producir (siguiente paso)

1. `scripts/build_align_control_corpus.py` (local; importa build_tqa/arc_pairs).
2. Edits a `scripts/run_lm_eval_v7.py`: `--num-fewshot`, `--fewshot-as-multiturn`,
   `--null-adapter` (zero offsets), `--instrument PATH` (dump diagnóstico R11).
3. `scripts/run_eval_bulletproof_pod.sh` (batch único, front-loaded).
4. `scripts/analyze_eval_bulletproof.py` (L1-L11, local).
