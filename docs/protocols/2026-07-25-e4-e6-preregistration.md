# Pod runbook — E4 (factorial objetivo×dominio) + E6 (Qwen causal spine)

**Pre-registro. Escrito ANTES de correr.** Decisión user 2026-07-25: correr E4+E6 e
**integrar al release v1** (no v2) — "hagamos un paper fuerte". Disciplina vigente:
bulletproof-before-prose · numbers-first · las rebajas se **reemplazan con dato**.

Contexto: L1 ya cerró local (B49) — el frac-CI pareado ARC es **[0.510, 0.732]**,
P(f>0.5)=0.982.

---

## Por qué estos dos

| Exp | Qué cierra | Estado en el paper hoy |
|---|---|---|
| **E4** | El confound **"generic" vs "domain exposure"** | `limitations.tex:114-124` lo declara como limitación ABIERTA y justifica no correrlo apelando a *"una atribución cuyo intervalo ya es [45,85]"* — justificación que **L1 invalidó** (ahora [51,73]) |
| **E6** | **Universalidad cross-familia** del off-axis | Qwen tiene el efecto replicado (ARC .4825→.6625) y la **geometría** hecha, pero **cero causal**: el propio artefacto avisa *"This is GEOMETRY (norm), not the readout effect"* |

---

## E4 — factorial 2×2 objetivo × dominio

**Pregunta:** el recovery del control genérico, ¿es *"cualquier fine-tuning recupera
el lift"* o *"fine-tuning en ESTE dominio lo recupera"*? Hoy el paper no puede
separarlas y lo dice.

### Celdas (2 de 4 YA EXISTEN — no se recomputan)

| | in-domain (TQA+ARC) | OOD-matched (MedMCQA) |
|---|---|---|
| **contrastive** (`--loss direct`) | **A** = V7-mc canónico ✅ (+35.0 ARC) | **C** = NUEVO |
| **plain-CE** (`--loss plain_ce`) | **B** = E1 ✅ (36.2% ± 5.0, 3 seeds) | **D** = NUEVO |

**Corpus OOD = MedMCQA** (`phase1c_kr_medmcqa`, 2800 pares en
`runs/phase3_adapters/kr_opcion_d/train/medmcqa_pairs.jsonl`). Elegido porque:
formato idéntico (`prompt`/`correct`/`wrong`, MC con distractor plausible) → matched
en forma, no en contenido; dominio médico sin overlap con ARC/TQA/HSwag/Wino; **loader
ya existe** (cero código nuevo); y **no es EWoK** → limpio bajo el release-gate.

Todo lo demás **idéntico** a E1/canónico: mismos 48 head-pairs, mismos 12,336 params,
`--n-train 1500 --n-val 200 --steps 800 --batch-size 1 --grad-accum 4 --lr 0.01
--beta 0.1 --gamma 0.5 --seq-len-max 384 --chat-template`, 3 seeds, `--data-seed 0`.

### Kill-rules (pre-especificadas, binning 1/3–2/3 = el que el paper ya usa)

Sea `rec(X)` = fracción del lift ARC canónico que recupera el brazo X.

1. **Gate de convergencia (mismo del paper):** un brazo con `arm − base < 0.01` en ARC
   es **no-op** → se reporta como tal y NO se le computa fracción (no se scorea un
   control roto).
2. **El test del confound** — comparar `rec(D)` (plain-CE OOD) contra `rec(B)` = 36.2%:
   - `rec(D) ≥ ⅔·rec(B)` → **"cualquier FT"**: el dominio no es lo que importa. El
     párrafo de Limitations se **reemplaza por el resultado** y "generic" queda como
     está.
   - `rec(D) ≤ ⅓·rec(B)` → **"domain exposure"**: hay que **re-scopear** "generic
     fine-tuning" a "fine-tuning con exposición al dominio" en todos sus sitios.
     Ésta es la lectura que hoy el paper NO puede descartar y que, si sale, **debilita**
     el claim de inducibilidad — se declara igual (numbers-first).
   - En el medio → ambigüedad parcial, se reporta el número sin elegir lectura.
3. **El objetivo cross-domain** — `rec(C)` vs `rec(A)`=100%: si el contrastivo OOD
   mantiene el grueso del lift, el efecto contrastivo es **independiente del dominio**;
   si colapsa, el lift necesita el dominio y eso acota la generalidad del fenómeno.
4. **Net-effect de C y D** (cos con v_inf, ratio off-axis) → **2 puntos nuevos del
   espectro de Fig 1**, con el **dominio** como eje adicional al objetivo×capacidad.
   Predicción del marco: el ratio sigue al **objetivo**, no al dominio (E1 mostró que
   sigue al objetivo, no al aparato). Si el dominio mueve el ratio más que el objetivo,
   el marco necesita un tercer eje y hay que decirlo.

### Costo
6 trainings (2 celdas × 3 seeds, ~20 min c/u) + 6 evals de 4 tareas + 1 base
apples-to-apples (ya existe: `runs/e1_plain_ce/eval_base.json`) + net-effect ×6.
**~4 h GPU.**

---

## E6 — Qwen causal spine (universalidad cross-familia)

**Pregunta:** el patrón *off-axis funciona / on-axis falla* — el corazón del paper —
¿es una propiedad del PEFT contrastivo, o de Gemma?

**Ya existe (no recomputar):** adapter `runs/v7_lofit_qwen14b/offsets_mc.pt`;
evals base/null/v7mc (`runs/mechanism_v8/D_qwen_{base,null,v7mc}.json`, ARC
.4825→.6625 = **+18.0 pp**); direcciones funcionales
`runs/oq1_functional_axis/functional_directions_qwen.pt` (v_inf, v_ret, W_know + cov);
geometría `offset_parperp_qwen.json` (null aleatorio K=2000, 1/d = 0.0078).

**Falta = todo el lado causal.** Los 4 offsets se construyen **LOCAL** (CPU, €0) y el
pod sólo los evalúa.

### Brazos (cada uno = un blob de offsets byte-compatible con `run_lm_eval_v7 --offsets`)

| Brazo | Construcción | Predicción (Gemma como referencia) |
|---|---|---|
| **W_know** norm-matched | Fisher-LDA on-axis, misma norma aplicada | ~6.4% en Gemma → **el foil** |
| **off_par** | componente ∥ a v_inf, norma natural | ~0 |
| **off_perp** | componente ⊥, norma natural | ~100% |
| **off_perp_shuffle** | dirección random DENTRO del complemento ⊥, norma per-head exacta | ~0 (si no, la dirección aprendida no es privilegiada) |
| additividad | el blob `mc` original | = lift completo (control) |

Más: **no-transfer gen** (gsm8k Qwen base vs adapter; Gemma −24.2 ± 7.3) y **matched
control** (same-param `plain_ce` en Qwen, 3 seeds → inducibilidad cross-familia).

### Kill-rules (pre-especificadas)

1. **El test de universalidad (el load-bearing):** si **W_know Qwen recupera > ⅓** del
   lift Qwen, entonces *"on-axis falla"* **NO es universal** — es específico de Gemma.
   Eso **no invalida** el resultado de Gemma, pero obliga a re-scopear el claim de
   mecanismo a "en este modelo" en abstract/intro/discussion. Se declara sin adornos.
2. **Si `off_perp` NO recupera ≳⅔** del lift Qwen → la descomposición ∥/⊥ no transfiere
   y el spine cross-familia se reporta como **negativo**, no se esconde.
3. **Si `off_perp_shuffle` recupera ~tanto como `off_perp`** → en Qwen la especificidad
   de la dirección aprendida cae (no-identificabilidad del lado write) → §5.3 concede
   explícitamente para Qwen.
4. **Additividad**: el blob `mc` re-evaluado debe reproducir el lift Qwen publicado
   (tol 1 pp). Si no, hay un bug de construcción → **abort antes de interpretar nada**.
5. **gsm8k**: se reporta el signo y la magnitud con su spread; el claim de
   no-transferencia cross-familia sólo se afirma si el signo coincide con Gemma.

### Costo
4 evals de offsets (4 tareas × 400) + additividad + gsm8k gen + 3 trainings del matched
control + sus evals. Qwen 14B es ~2× más rápido que Gemma 31B. **~6-8 h GPU.**

---

## Orden de ejecución

**LOCAL primero (€0, antes de prender el pod):**
1. Parametrizar `make_wknow_offset.py`, `make_offpar_offperp_offset.py`,
   `make_shuffle_offperp.py` (hoy son paths hardcodeados a Gemma, sin argparse) →
   flags `--functional-dirs / --offsets / --out`, **defaults = los de Gemma** para que
   el path canónico no cambie de comportamiento.
2. Construir los 4 blobs de Qwen y verificar que cargan y que las normas cuadran.
3. Commit + push (el pod hace `git pull`).

**POD después:** E4 (trainings+evals) y E6 (evals). Regla del repo: sólo GPU en el pod.

---

## Bloques del pod (cada uno self-contained — pegar de a uno)

### 0. Bring-up + pull + verificación de inputs
```bash
cd /workspace/MSAP && pwd
git pull && pip install -e . --no-deps      # protege el pin huggingface_hub<1.14
export BASE=$(dirname "$(find /workspace -maxdepth 7 -name config.json -path '*gemma*31*' 2>/dev/null | grep -v /.locks/ | head -1)")
[ -f "$BASE/config.json" ] || { hf download google/gemma-4-31B-it --local-dir /workspace/models/gemma-4-31b-it; export BASE=/workspace/models/gemma-4-31b-it; }
mkdir -p /workspace/models && ln -sfn "$BASE" /workspace/models/gemma-4-31b-it
export BASE=/workspace/models/gemma-4-31b-it
echo "MODEL_REVISION=$(basename "$(readlink -f /workspace/models/gemma-4-31b-it)")"
# inputs que DEBEN venir del pull (construidos LOCAL, €0):
ls -la runs/vinf_causal_qwen/*.pt          # 7 blobs Qwen (vinf, wknow, offpar, offperp, shuffle×3)
ls -la runs/phase3_adapters/kr_opcion_d/train/medmcqa_pairs.jsonl   # corpus OOD de E4
nvidia-smi                                  # DEBE estar vacío antes de cada bloque
```

### 1. E4 — 6 trainings OOD (2 objetivos × 3 seeds)
```bash
cd /workspace/MSAP && pwd
export BASE=/workspace/models/gemma-4-31b-it
mkdir -p runs/e4_factorial
# smoke de 100 steps (NO 20 — ver el gate abajo), los dos objetivos
for L in direct plain_ce; do
python -m scripts.train_v7_lofit --base "$BASE" \
  --heads-file runs/v7_lofit_gemma4_31b_chat/head_probe_mc.json --top-k 48 \
  --datasets phase1c_kr_medmcqa --steps 100 --eval-every 25 \
  --lr 0.01 --loss $L --seed 0 --data-seed 0 --chat-template \
  --out runs/e4_factorial/smoke100_${L}.pt
done
```

**⚠ GATE DEL SMOKE — calibrado contra E1, no contra una expectativa a priori.**
La primera versión de este runbook pedía "val_acc sube en 20 steps, si no parar". Es un
criterio FALSO y hay que no repetirlo: el smoke100 de E1 —el control que sí funcionó y
está publicado (36.2% de recovery)— hace esto:

| step | 1 | 25 | 50 | 75 | 100 |
|---|---|---|---|---|---|
| val_loss | 30.27 | 23.44 | 17.08 | 14.93 | 13.20 |
| val_acc | 0.48 | 0.48 | **0.56** | 0.44 | 0.44 |

val_acc **oscila y termina por debajo de donde empezó**, y a step 25 todavía no se movió.
Con `n_val=200` y `batch=1` la columna `loss` es de un solo ejemplo (ruido puro) y
val_acc tiene ~±0.05 de ruido. El canónico recién llega a val_acc 0.94 a **800 steps**.

**Gate correcto:** `val_loss` baja monótono **y** `val_acc` toca **≥0.56** en algún eval.
Eso es "el trainer está sano", no "el adapter sirve" — lo segundo se juzga con el eval
de las 4 tareas, no con el smoke.
```bash
cd /workspace/MSAP && pwd
export BASE=/workspace/models/gemma-4-31b-it
nohup bash -c 'for LOSS in direct plain_ce; do for S in 0 1 2; do
  python -m scripts.train_v7_lofit --base /workspace/models/gemma-4-31b-it \
    --heads-file runs/v7_lofit_gemma4_31b_chat/head_probe_mc.json --top-k 48 \
    --datasets phase1c_kr_medmcqa --n-train 1500 --n-val 200 \
    --steps 800 --batch-size 1 --grad-accum 4 \
    --lr 0.01 --beta 0.1 --gamma 0.5 --loss $LOSS --seq-len-max 384 \
    --chat-template --seed $S --data-seed 0 \
    --out runs/e4_factorial/ood_${LOSS}_seed${S}.pt
done; done' > /workspace/MSAP/runs/e4_factorial/train.log 2>&1 & disown
tail -f /workspace/MSAP/runs/e4_factorial/train.log
```

### 1b. Convergencia REAL de las 6 corridas (2026-07-25) — el gate pasa holgado

| brazo | seed | best val_acc | @step | final | val_loss 1→800 |
|---|---|---|---|---|---|
| direct | 0 | 0.800 | 800 | 0.800 | 21.57 → **1.30** |
| direct | 1 | 0.800 | 200 | 0.680 | 21.57 → 4.03 |
| direct | 2 | 0.840 | 600 | 0.800 | 21.57 → 2.11 |
| plain_ce | 0 | 0.800 | 400 | 0.760 | 21.57 → 4.01 |
| plain_ce | 1 | 0.800 | 600 | 0.800 | 21.57 → 3.94 |
| plain_ce | 2 | 0.840 | 800 | 0.840 | 21.57 → 3.68 |

Los 6 van de 0.600 a 0.80–0.84. El trainer está sano en MedMCQA; el smoke de 20 steps
no lo mostraba (ver el gate corregido arriba). ~5-6 min de GPU por corrida.

### 2. E4 — evals de los DOS checkpoints (best + final, como E1)
E1 reportó best-ckpt **y** final (36.2% ± 5.0 vs 39.7% ± 3.1 → "rango honesto 36–40%");
E4 evalúa ambos para que las celdas se comparen sin asteriscos. El base apples-to-apples
YA existe: `runs/e1_plain_ce/eval_base.json` (en HF `msap-review-remediation-20260724`;
el análisis corre LOCAL, así que no hace falta bajarlo al pod).
```bash
cd /workspace/MSAP && pwd
nohup bash -c 'for CKPT in "" ".final"; do for LOSS in direct plain_ce; do for S in 0 1 2; do
  python -m scripts.run_lm_eval_v7 --base /workspace/models/gemma-4-31b-it \
    --offsets runs/e4_factorial/ood_${LOSS}_seed${S}${CKPT}.pt \
    --tasks arc_challenge,truthfulqa_mc1,hellaswag,winogrande --limit 400 \
    --apply-chat-template --max-length 4096 \
    --out runs/e4_factorial/eval_ood_${LOSS}_seed${S}${CKPT}.json
done; done; done' > /workspace/MSAP/runs/e4_factorial/eval.log 2>&1 & disown
tail -f /workspace/MSAP/runs/e4_factorial/eval.log
```

### 3. E4 — net-effect (los 2 puntos nuevos del espectro de Fig 1)
```bash
cd /workspace/MSAP && pwd
export BASE=/workspace/models/gemma-4-31b-it
for LOSS in direct plain_ce; do for S in 0 1 2; do
  python scripts/extract_align_direction.py --base "$BASE" \
    --offsets runs/e4_factorial/ood_${LOSS}_seed${S}.pt \
    --vinf runs/vinf_causal/mc_vinf_offset.pt --n 200 \
    --out runs/e4_factorial/neteffect_ood_${LOSS}_seed${S}.json
done; done
```

### 4. E6 — Qwen: base + GUARD de additividad (⚠ ABORTAR si falla)
```bash
cd /workspace/MSAP && pwd
hf download Qwen/Qwen2.5-14B-Instruct --local-dir /workspace/models/qwen2.5-14b-instruct
export QBASE=/workspace/models/qwen2.5-14b-instruct
mkdir -p runs/e6_qwen_spine
# base (null-adapter) -> DEBE dar ARC acc_norm 0.4825
python -m scripts.run_lm_eval_v7 --base "$QBASE" --null-adapter \
  --offsets runs/v7_lofit_qwen14b/offsets_mc.pt \
  --tasks arc_challenge,truthfulqa_mc1,hellaswag,winogrande --limit 400 \
  --apply-chat-template --max-length 4096 --out runs/e6_qwen_spine/eval_base.json
# additividad: el blob mc original -> DEBE dar ARC 0.6625 (tol 1pp)
python -m scripts.run_lm_eval_v7 --base "$QBASE" \
  --offsets runs/v7_lofit_qwen14b/offsets_mc.pt \
  --tasks arc_challenge,truthfulqa_mc1,hellaswag,winogrande --limit 400 \
  --apply-chat-template --max-length 4096 --out runs/e6_qwen_spine/eval_mc.json
```
**Si el base ≠ 0.4825 o el mc ≠ 0.6625 (±1 pp): PARAR.** Es bug de harness o de
chat-template — no interpretar ningún brazo hasta reconciliar (precedente C-3).

### 5. E6 — los 4 brazos causales
```bash
cd /workspace/MSAP && pwd
export QBASE=/workspace/models/qwen2.5-14b-instruct
nohup bash -c 'for ARM in mc_wknow_offset mc_offpar_offset mc_offperp_offset mc_offperp_shuffle_seed0 mc_offperp_shuffle_seed1 mc_offperp_shuffle_seed2; do
  python -m scripts.run_lm_eval_v7 --base /workspace/models/qwen2.5-14b-instruct \
    --offsets runs/vinf_causal_qwen/${ARM}.pt \
    --tasks arc_challenge,truthfulqa_mc1,hellaswag,winogrande --limit 400 \
    --apply-chat-template --max-length 4096 \
    --out runs/e6_qwen_spine/eval_${ARM}.json
done' > /workspace/MSAP/runs/e6_qwen_spine/arms.log 2>&1 & disown
tail -f /workspace/MSAP/runs/e6_qwen_spine/arms.log
```

### 6. E6 — no-transfer gen (gsm8k) + matched control Qwen
```bash
cd /workspace/MSAP && pwd
export QBASE=/workspace/models/qwen2.5-14b-instruct
python scripts/gen_gsm8k_scale_pod.py --base "$QBASE" \
  --offsets runs/v7_lofit_qwen14b/offsets_mc.pt \
  --n 200 --batch-size 16 --max-new-tokens 512 --scale-grid 0,1.0 \
  --out runs/e6_qwen_spine/gsm8k_scale.json --keep-per-item
```
```bash
cd /workspace/MSAP && pwd
export QBASE=/workspace/models/qwen2.5-14b-instruct
# matched control: MISMA config del canónico Qwen (tqa,arc / steps 800 / lr .01 /
# beta .1 / gamma .5 / top-k 48), sólo cambia el objetivo a CE-plain.
nohup bash -c 'for S in 0 1 2; do
  python -m scripts.train_v7_lofit --base /workspace/models/qwen2.5-14b-instruct \
    --heads-file runs/v7_lofit_qwen14b/head_probe_mc.json --top-k 48 \
    --datasets tqa,arc --n-train 1500 --n-val 200 \
    --steps 800 --batch-size 1 --grad-accum 4 \
    --lr 0.01 --beta 0.1 --gamma 0.5 --loss plain_ce --seq-len-max 384 \
    --chat-template --seed $S --data-seed 0 \
    --out runs/e6_qwen_spine/plain_ce_seed${S}.pt
  python -m scripts.run_lm_eval_v7 --base /workspace/models/qwen2.5-14b-instruct \
    --offsets runs/e6_qwen_spine/plain_ce_seed${S}.pt \
    --tasks arc_challenge,truthfulqa_mc1,hellaswag,winogrande --limit 400 \
    --apply-chat-template --max-length 4096 \
    --out runs/e6_qwen_spine/eval_plain_ce_seed${S}.json
done' > /workspace/MSAP/runs/e6_qwen_spine/matched.log 2>&1 & disown
tail -f /workspace/MSAP/runs/e6_qwen_spine/matched.log
```

⚠ **`--chat-template` en el training del matched control Qwen:** el `config` del blob
canónico (`offsets_mc.pt`) NO registra la flag, así que no consta si se usó. El bloque
la incluye por consistencia con Gemma; **verificar el efecto sobre el base eval antes
de comparar** — si el matched no reproduce el mismo base, re-correr sin la flag y
declarar cuál se usó.

### 7. Backup HF al cerrar el pod
```bash
cd /workspace/MSAP && pwd
hf upload sebacascardo87/msap-e4-e6-20260725 runs/e4_factorial runs/e4_factorial \
  --repo-type dataset --private
hf upload sebacascardo87/msap-e4-e6-20260725 runs/e6_qwen_spine runs/e6_qwen_spine \
  --repo-type dataset --private
```
Sin EWoK (E4 usa MedMCQA; E6 usa TQA+ARC) → limpio bajo el release-gate.

## ⚠ Gotcha nuevo (2026-07-25) — pod fresco + Git LFS

Los `.pt` del repo están en **Git LFS**. Un pod nuevo que sólo hace `git pull` se trae
los **punteros de texto**, no los binarios, y `torch.load` muere con
`_pickle.UnpicklingError: invalid load key, 'v'` — la `'v'` es el `version
https://git-lfs.github.com/spec/v1` del puntero. Pasó con `mc_vinf_offset.pt` en el
net-effect de E4.

**Va en el bloque 0, siempre:**
```bash
cd /workspace/MSAP && pwd
git lfs install && git lfs pull     # apt-get install -y git-lfs si no está
for f in runs/vinf_causal/*.pt runs/vinf_causal_qwen/*.pt runs/v7_lofit_qwen14b/offsets_mc.pt; do
  printf "%-52s %s\n" "$f" "$(head -c 9 "$f" | grep -q 'version h' && echo 'LFS-POINTER (roto)' || echo ok)"
done
```
Afecta a **todo** lo que viaje por git: los blobs canónicos de Gemma y los 7 de Qwen de
E6. NO afecta lo que el pod genera localmente (los `ood_*.pt` de E4).

## Gotchas heredados (B48, aplican igual)

- **Nunca dos procesos GPU a la vez** — Gemma 2×62 GB > 96 GB. `nvidia-smi` vacío
  antes de cada bloque.
- `mkdir -p` del dir de salida **antes** de redirigir el log; log con **path absoluto**
  (`cd X && nohup … & disown` corre el `cd` en el subshell).
- `nohup … > log 2>&1 & disown`, **nunca** `tee|pipe` (SIGPIPE silencioso).
- **`--apply-chat-template` SIEMPRE** en los cold-MC (el guard de C-3 abortó un run por
  esto y salvó ~2 h).
- `--null-adapter` requiere `--offsets`.
- Cada bloque self-contained con `cd /workspace/MSAP` + `pwd` (cwd no persiste).

## Integración (post-pod, una sola pasada)

L1 + E4 + E6 entran juntos al LaTeX — regla user B47 ("no actualizar a cada rato") +
regla repo ("un fix re-deriva todas sus cantidades"). Sitios ya mapeados para L1: los
9 que citan `[45\%, 85\%]` (abstract:16, discussion:217, introduction:49+232,
limitations:122, method:200, results:444+476+516). Después: verificador de invariancia
→ pdflatex ×3 → citecheck → regenerar `arxiv-v1.tar.gz` + README de submission.
