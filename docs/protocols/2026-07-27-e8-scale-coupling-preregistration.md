# E8 — ¿el lift de scoring y la compresión de longitud son una ganancia o dos?

**Fecha:** 2026-07-27 · **Estado:** KR-A/B/D **cerradas con artefactos preexistentes**;
KR-C abierta (pide pod, es secundaria)
**Análisis:** `scripts/analyze_scale_coupling.py` (local, CPU, read-only sobre sus entradas)
**No interfiere con R12** (`runs/paired_gen/`): artefactos en `runs/e8_scale_coupling/`,
sin commits a `feature/v8` mientras R12 corra.

> **Orden cronológico (importa para la validez del pre-registro).** Las kill-rules de
> este documento se escribieron **antes** de encontrar los datos que las responden. El
> barrido denso positivo se pre-registró para pod; recién después apareció en HF
> (`msap-dose-sweep-20260707`, 7 jul) un barrido con **exactamente la misma grilla**.
> Los umbrales no se tocaron al ver los números.

## La pregunta

El offset `theta_mc` es un dial monótono sobre dos lecturas distintas: el scoring
cold-MC (el +35 del headline) y la longitud de generación (755 → 90 tokens, E7).
¿Un mecanismo con dos caras, o dos ganancias separables sobre el mismo eje?

**La monotonía conjunta no es evidencia** — ambas curvas son monótonas en la escala
por construcción, así que correlacionarlas es vacío. Lo que discrimina es la
**forma** de cada curva contra el dial:
`f(s) = [v(s) − v(0)] / [v(1) − v(0)]`, con la longitud invertida para co-orientar.

## Datos

| fuente | qué aporta | procedencia |
|---|---|---|
| `msap-dose-sweep-20260707` | ARC n=400 `acc_norm` + loglik per-ítem en `s ∈ {0, .25, .5, .75, 1}`; gsm8k n=200 con accuracy, longitud y per-ítem en la misma grilla | HF, 2026-07-07 |
| `runs/e7_inverse/` | las mismas celdas para `s ∈ {−2, −1, −0.5, 0, 1}` | local, E7 |

**KR-A (repro) en verde:** `s=0` da acc 0.4975 / acc_norm 0.5025 — el base canónico,
idéntico a E7. La recomputación de `acc_norm` desde los loglik coincide con lm-eval
hasta 1e-6 → **el margen usado es exactamente la cantidad que `acc_norm` umbralea**.
⚠ Nota de repro: `s=1` da acc_norm 0.8550 acá y 0.8500 en E7 (2 ítems, dentro de la
tolerancia ±1 pp) — ruido de batching en bf16 entre runs, no drift de condición.

## Resultado 1 — el lado positivo denso (n=400, bootstrap pareado 2000)

| scale | acc | acc_norm | margin_norm | gsm8k len | gsm8k acc | eos |
|---|---|---|---|---|---|---|
| +0.00 | 0.4975 | 0.5025 | +0.06992 | 150.0 | 0.9700 | 0.995 |
| +0.25 | 0.6400 | 0.6600 | +0.51023 | 127.5 | 0.9850 | 0.995 |
| +0.50 | 0.7825 | 0.7775 | +0.90358 | 109.3 | 0.9750 | 1.000 |
| +0.75 | 0.8475 | 0.8550 | +1.28527 | 97.3 | 0.9500 | 0.995 |
| +1.00 | 0.8475 | 0.8550 | +1.38449 | 98.8 | 0.8150 | 0.950 |

### KR-B — **ZONA GRIS, no se concluye**

| scale | f_score [95% CI] | f_len [95% CI] | separación [95% CI] |
|---|---|---|---|
| +0.25 | +0.3349 [+0.271, +0.410] | +0.4382 [+0.328, +0.600] | −0.1033 [−0.190, −0.056] |
| +0.50 | +0.6342 [+0.569, +0.712] | +0.7952 [+0.621, +1.076] | −0.1610 [−0.362, −0.052] |
| +0.75 | +0.9245 [+0.864, +0.997] | +1.0280 [+0.819, +1.368] | −0.1035 [−0.371, +0.046] |

`max |separación| = 0.161`, dentro de la banda gris `[0.15, 0.30)` pre-registrada →
**no se concluye desacople**. La dirección pre-registrada **se sostiene** (la longitud
corre por delante del margen en las 3 celdas, y el CI excluye 0 en 2 de 3), pero la
magnitud no alcanza el umbral. **Lectura honesta: las dos curvas van bastante
acopladas, con un desfase modesto pero real.** La hipótesis "dos ganancias separables
sobre el mismo eje" **no** queda respaldada por este test.

## Resultado 2 — KR-D: **la brevedad NO causa el daño en razonamiento**

Doble disociación, con sus sigmas (McNemar pareado sobre los mismos 200 ítems):

| tramo | Δ longitud | Δ accuracy | test |
|---|---|---|---|
| `0 → 0.5` | **−40.7 tok** (SE 3.6, ~11σ) | +0.5 pp (1 ganado, 0 perdidos) | McNemar p = 1.00 |
| `0.75 → 1.0` | **+1.4 tok** (SE 6.6, ~0.2σ) | **−13.5 pp** (4 ganados, 31 perdidos) | McNemar χ²=19.3, **p = 1.1e−5** |
| `0 → 1.0` | −51.2 tok | −15.5 pp | χ²=27.3, p = 1.8e−7 |

Se comprime el 80% del rango de longitud **sin tocar** la accuracy; y después la
accuracy se desploma **sin comprimir un token más**. Esto **cierra lo que E7 dejó
explícitamente abierto** ("no se puede concluir que la brevedad cause el daño"): no lo
causa. Son mecanismos distintos.

## Resultado 3 (POST-HOC, no pre-registrado — tratar como hipótesis)

En el tramo `0.75 → 1.0`: `acc_norm` no se mueve (0.8550 → 0.8550), el margen suma
sólo 7% de su rango, la longitud no se mueve (+1.4 tok) — **y sin embargo el
razonamiento pierde 13.5 pp**. Hay un tramo del dial donde las dos caras que este
experimento venía persiguiendo ya saturaron y el daño sigue creciendo solo.

Si sobrevive verificación, la lectura no es "dos cosas" sino **al menos tres**, y la
que se separa limpio no es la brevedad: es el daño. **No pre-registrado y con una
sola grilla — no entra al paper sin su propia kill-rule.**

## Lo que queda abierto

**KR-C** (pide pod, ~1 h): grilla `s ∈ {1.25, 1.5}`. Si `f_score(1.5) < f_score(1.0)`
mientras `f_len(1.5) > f_len(1.0)` → desacople directo sin depender de las anclas.

> ⚠ **Corrección de prioridad (2026-07-27b).** La versión previa de esta sección
> despriorizaba KR-C con el argumento de que *"el Resultado 3 ya da el desacople por
> otra vía"*. Eso es **circular**: el Resultado 3 es post-hoc y este mismo documento
> declara que no entra al paper sin su propia kill-rule. No se puede despriorizar el
> experimento que lo verificaría invocando el resultado que necesita verificación.
> KR-C vuelve a la cola, y se le suma KR-E (abajo), que es el test que ataca el
> Resultado 3 **en el tramo donde ocurre**.

---

# KR-E — pre-registro de la extensión densa (escrito 2026-07-27b, ANTES de correr)

**Qué se testea.** El Resultado 3 afirma que en `0.75 → 1.0` el razonamiento pierde
13.5 pp mientras el margen suma sólo 7% de su rango y la longitud no se mueve
(+1.4 tok). Con dos puntos no se distingue **una tercera cara que crece** de **un
cliff en el extremo**. La grilla densa `s ∈ {0.80, 0.85, 0.90, 0.95}` (más 0.75 y 1.00
re-medidos dentro del mismo run, para anclar sin cruzar corridas) resuelve la forma.

**Por qué acá y no arriba de 1.0.** El fenómeno vive en este tramo. Además el
`eos_rate` acá está sano (0.995 → 0.95), así que **ninguna celda se excluye por la
regla de eos**; arriba de 1.0 el `eos` puede desplomarse (a `s=−2` fue 0.083) y KR-C
puede terminar sin concluir por diseño. KR-E no tiene ese modo de falla.

### Condiciones de saturación (las dos deben cumplirse, o el test no aplica)

- **(a) margen saturado:** `f_score(1.00) − f_score(0.75) ≤ 0.15` del rango total.
- **(b) longitud saturada:** `|Δ mean_gen_len(0.75 → 1.00)| < 10 tok` (≈1.5 SE, con
  SE ≈ 6.6 tok).

Si (a) o (b) fallan, **el tramo no está saturado** y el Resultado 3 queda **refutado
en su premisa**: el daño sería la contracara de algo que sí se está moviendo, y no hay
tercera cosa que explicar. Se reporta como refutación, no como "no concluyente".

### Lectura del daño (con (a) y (b) en verde)

Se mide gsm8k n=200, pareado por ítem, McNemar entre celdas consecutivas.

| patrón observado | lectura pre-declarada |
|---|---|
| caída **progresiva**: ≥3 de los 5 intervalos con Δacc < 0, y ningún intervalo concentra >60% de la caída total | **H3 SOBREVIVE** — hay una tercera cantidad que crece con el dial mientras las otras dos saturaron. Entra al paper con esta kill-rule citada |
| caída **de cliff**: un solo intervalo concentra >60% de la caída total | **H3 se reemplaza** por *"existe un umbral de escala por encima del cual el razonamiento colapsa"* — hallazgo distinto, igual de citable, pero NO es "una tercera cara continua" |
| caída total no significativa (McNemar `0.75 → 1.00` con p ≥ 0.01) | **H3 MUERE.** El 13.5 pp original era ruido de dos puntos |

**Ningún umbral de esta tabla se toca después de ver los números.** Si el patrón cae
justo en el borde (p.ej. un intervalo con exactamente 60%), se reporta como
indeterminado y no se elige.

### Confounds que ya sabemos que arrastra

- Una sola tarea de razonamiento (gsm8k) y una sola familia (Gemma). No se afirma
  generalidad entre tareas.
- `f_score` sale de cold-MC ARC y el daño de gsm8k 4-shot: espacios distintos, unidos
  sólo por el escalar del dial. El claim es sobre **forma contra el dial**, no sobre
  dirección compartida.
- `n=200` en gsm8k; los contrastes son pareados pero la potencia por intervalo es
  modesta — por eso el criterio primario es el **patrón** (3 de 5 intervalos), no la
  significancia intervalo a intervalo.

Runbook: [`../plans/2026-07-27-pod-runbook-dense-scale-sweep.md`](../plans/2026-07-27-pod-runbook-dense-scale-sweep.md).

## Confounds declarados

1. **Techo de ARC.** A `s=0.75` el `acc_norm` ya está en su valor final: la métrica
   primaria es el **margen** por eso mismo. Entre 0.75 y 1.0 el margen sube 7% y
   ningún ítem cruza el umbral de decisión — la accuracy plana ahí es **techo de
   readout**, no saturación del mecanismo. (Justifica ex post la elección del margen.)
2. **Dos formatos.** `f_score` sale de cold-MC (ARC) y `f_len` de gsm8k 4-shot. Son
   espacios distintos; comparar sus **formas** contra un escalar común es legítimo,
   pero el claim es "misma ganancia respecto del dial", **no** "misma dirección".
3. **`eos_rate`.** Todas las celdas positivas están sanas (≥0.95), así que la regla
   pre-registrada de excluir celdas con `eos < 0.5` no descartó ninguna. El lado
   negativo de E7 **sí** está contaminado (`eos` 0.083 en `s=−2`) y por eso no decide.
4. **n=200 en gsm8k.** Los contrastes son pareados y los sigmas están arriba, pero es
   una sola tarea de razonamiento y una sola grilla.

## Artefactos

- `runs/e8_scale_coupling/positive/` — lado positivo denso (el resultado principal).
- `runs/e8_scale_coupling/` — pre-test sobre el lado negativo de E7.
- `runs/e8_scale_coupling/dose_sweep/` — copia local de `msap-dose-sweep-20260707`.
