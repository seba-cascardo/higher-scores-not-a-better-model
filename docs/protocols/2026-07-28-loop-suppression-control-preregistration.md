# Pre-registro — control de supresión de loops: ¿el daño en CoT es emisión o razonamiento?

**Fecha:** 2026-07-28 · **Estado:** pre-registrado, sin correr
**Escrito ANTES de correr ninguna celda.** Ningún umbral ni lectura se toca después de ver
los números.
**Destino:** publicable si sobrevive — decide cómo se escribe el capítulo generativo del
paper del eje. No es diagnóstico interno (a diferencia de P-gate).
**No interfiere con R12:** artefactos en `runs/loop_ctl/`.

---

## 1. La pregunta y por qué existe ahora

R12/MMLU-Pro cerró KR3: el lift de scoring no transfiere a CoT (−6.10 pp, bounds de Manski
solapados → no declarable). Al preguntar **qué es** la masa unparsed que ensancha el bound
apareció una firma limpia
([`2026-07-28-r12-mmlu-pro-results.md`](2026-07-28-r12-mmlu-pro-results.md)):

| arm | loops (distinct-4 < 0.35) | peor ítem | mediana tokens |
|---|---|---|---|
| base | **0 / 1000** | 0.738 | 467 |
| align_lora | 10 / 1000 | 0.038 | 131 |
| v7mc | 146 / 1000 | 0.019 | 96 |
| ce_s0 | 298 / 1000 | 0.014 | 88.5 |

La tasa de loops ordena monótona con el daño en `cot`. Eso sugiere **emisión**, no
razonamiento — pero probarlo comparando accuracy entre ítems que loopearon y que no es
condicionar sobre una variable **post-tratamiento**, el mismo error que el analizador
pareado existe para evitar. La única salida es **intervenir**: suprimir el loop en decoding
y volver a medir.

## 2. Lo que ya se sabe sin gastar GPU (y de dónde sale el poder del test)

`runs/paired_gen/items_mmlu_pro_loopctl.json`, campo `reference_at_640_no_suppression` —
los valores de R12 restringidos al subset:

| estrato | n | base `cot` | v7mc `cot` | v7mc unparsed |
|---|---|---|---|---|
| **A** — v7mc loopeó | 146 | 0.5753 | **0.0616** | 101 |
| **B** — v7mc no loopeó | 154 | 0.6883 | **0.7078** | 12 |
| ALL (IPW → los 1000) | 300 | 0.6718 | 0.6135 | — |

**Donde el adapter no degenera, su CoT está al nivel del base (+1.95 pp).** Todo el daño
vive en el estrato A. El IPW reconstruye la marginal de los 1000 (−5.83 pp contra el −6.10
medido allá, dentro del ruido de muestreo) — el diseño de la muestra se valida a sí mismo.

⚠ **Esto NO identifica la causa, y es la razón de ser del experimento.** El estrato A es
también más difícil *para el base* (0.5753 vs 0.6883, −11.3 pp), así que hay selección de
dificultad además de degeneración. Lo que sí da es **potencia**: la celda de supresión
decide entre 0.06 y 0.58 en el estrato A. Un efecto de ese tamaño no necesita n grande.

## 3. Presupuesto — por qué n=300 y no 1000

Medido en el run del 2026-07-27: el canal `cot` a n=1000 cuesta **~70 min por brazo**
(4189 s v7mc, 4374 s ce_s0, 5162 s align_lora). Cinco celdas a n=1000 son ~6 h de pod para
un control mecanístico. A n=300, ~21 min por celda a cap 640 y ~34 min a cap 1024.

**Los baselines @640 sin supresión NO se re-corren** — salen de R12 restringido al subset,
y están congelados en el item file de arriba.

## 4. Muestreo (declarado antes de correr)

- **Estrato A:** los **146** ítems donde v7mc loopeó. Prob. de selección 1.0, IPW 1.0.
- **Estrato B:** **154** de los 854 restantes, `seed=0`. Prob. 154/854, IPW **5.5455**.
- `146·1.0 + 154·5.5455 = 1000` por construcción (el builder aborta si no cierra).

Los estratos se definen contra la corrida **que ya existe**, así que nada acá mira datos
que el control va a producir. Seleccionar sobre "v7mc loopeó" es post-tratamiento: por eso
los pesos están **en el archivo**, y las lecturas marginales se hacen por IPW, nunca
tratando la muestra como uniforme.

## 4b. Enmienda 2026-07-28a — la dosis, y por qué pasa a ser la celda primera

**Escrita antes de correr cualquier celda** (nada había corrido cuando se agregó), a partir
de una observación del user: **todo R12 corrió a `offset_scale: 1.0`** — verificado en los
cuatro artefactos — o sea el daño en `cot` y los 146 loops son el adapter a **dosis plena**.

La curva de dosis-respuesta ya medida (`runs/e8_scale_coupling/positive/scale_coupling.json`,
leída del artefacto crudo) dice que ésa no es la dosis que un producto elegiría:

| scale | ARC acc_norm (n=400) | gsm8k acc (n=200) | **eos_rate** | mean_gen_len |
|---|---|---|---|---|
| 0.00 | 0.5025 | 0.970 | 0.995 | 150.0 |
| 0.50 | 0.7775 | 0.975 | 1.000 | 109.3 |
| **0.75** | **0.8550** | **0.950** | **0.995** | 97.3 |
| 1.00 | 0.8550 | 0.815 | **0.950** | 98.8 |

A 0.75 el lift de ARC está **completo** (idéntico a 1.0), el daño en gsm8k es −2 pp en vez
de −15.5, y el **`eos_rate` vuelve a 0.995 — el del base**. Ese último número es la misma
cara que los loops de acá: generaciones que corren al cap sin cerrar turno. El paper ya
afirma la separabilidad (`discussion.tex`, *"The collateral reasoning loss is dose-separable
from the scoring benefit"*), con el caveat **"in steering, not in the baked adapter"**.

**El caveat no bloquea esto:** R12 inyecta v7mc por `install_lofit_hooks` con
`alpha * args.offset_scale` — es steering, no horneado. La celda es `--offset-scale 0.75`.

### Lo que NO está medido, y es la pregunta

Si la separabilidad de dosis **vale en MMLU-Pro**. El riesgo es concreto: en ARC el lift
satura a 0.75 pero `margin_norm` **sigue creciendo** (1.285 → 1.384). En MMLU-Pro hay **10
opciones** y el efecto es 3× más chico (+12.20 vs +35.83), así que la accuracy puede **no**
estar saturada a 0.75 — y entonces habría un trade-off real donde en ARC no había, y
`"saturates early"` tendría que calificarse como específico de ARC.

El canal `cold` cuesta **257 s** para los 1000 ítems (9503 pares opción, medido en el log del
2026-07-27), así que la curva entera sale por ~13 min de GPU.

### Por qué la dosis elegida es 0.75 y no 0.5 — medido, no leído de la tabla

La tabla invita a preferir 0.5: ahí gsm8k imprime **por encima** del base (0.975 vs 0.970) y
el `eos_rate` es el único 1.000 del sweep. **Las dos ventajas son 1 ítem de 200.** Los pasos
pareados sobre los mismos ítems (`scripts/analyze_dose_tradeoff.py` →
`runs/e8_scale_coupling/dose_tradeoff.json`):

| paso | ARC acc_norm | gsm8k | veredicto |
|---|---|---|---|
| 0.00 → 0.25 | **+15.75** [+12.00,+19.75] p=3.4e−15 | +1.50 [+0.00,+3.50] p=0.25 ruido | 0.25 domina 0.00 |
| 0.25 → 0.50 | **+11.75** [+8.50,+15.25] p=2e−11 | −1.00 [−2.50,+0.00] p=0.50 ruido | 0.50 domina 0.25 |
| **0.50 → 0.75** | **+7.75** [+5.00,+10.75] p=3.7e−8 | −2.50 [−5.50,**+0.00**] p=0.18 ruido | **0.75 domina 0.50** |
| **0.75 → 1.00** | +0.00 [−2.25,+2.25] ruido | **−13.50** [−19.00,−8.00] p=3.5e−6 | **0.75 domina 1.00** |

**0.75 es el único punto no dominado de la curva** — gana desde abajo y desde arriba. Y el
−2 pp de 0.75 contra base **tampoco es distinguible** (p=0.34, CI [−5.00,+1.00], **4 ítems de
200**): todo el daño de 15.5 pp aparece entero en el tramo 0.75→1.00.

⚠ **Consecuencia editorial para el paper:** `discussion.tex` escribe el 0.75 como *"at the
edge of the pre-registered 2pp window"*, lo que suena a daño chico pero presente. El test
pareado dice que **no hay daño detectable a 0.75**. La separabilidad se puede afirmar más
limpio de lo que está.

⚠ **El límite de esa afirmación, que va escrito con ella:** gsm8k base está en 0.970 con
n=200, así que el poder para detectar un decremento chico es bajísimo. **Un CI que contiene
el cero es fracaso de distinción, no evidencia de equivalencia.** No se afirma que 0.75 sea
inocuo; se afirma que estos datos no ven daño.

### Celdas D — la dosis (~35 min, PRIMERA)

| celda | brazo | dosis | canal | n | para qué | ~tiempo |
|---|---|---|---|---|---|---|
| **D-cold** | v7mc | 0.25 / 0.50 / 0.75 | `cold` | **1000** | ¿el lift satura temprano también acá? | 13 min |
| **D-cot** | v7mc | 0.75 | `cot` | 300 | la cara generativa a la dosis de producto | 21 min |

`@1.0` no se re-corre — es R12. El `cold` va sobre los **1000** (es barato y así la curva es
comparable al headline, no al subset); el `cot` sobre los 300 del subset.

### Lecturas pre-declaradas de las celdas D

| contraste | qué decide | predicción registrada |
|---|---|---|
| `cold` @0.75 vs @1.0 | ¿satura temprano en MMLU-Pro? | **sin prior firme.** ARC dice que sí; 10 opciones y un efecto 3× menor dicen que puede que no. Es el punto del experimento |
| `cold` @0.75 vs base | ¿cuánto lift queda a dosis de producto? | > 0 con CI que excluye 0; la magnitud es lo que se mide |
| `cot` @0.75: loops | ¿la degeneración es dose-dependent? | **caen.** El `eos_rate` de gsm8k vuelve al del base a 0.75 |
| `cot` @0.75 vs @1.0, estrato A | ¿se recupera el daño? | **sube.** Si no sube, la degeneración no es dose-dependent en esta tarea |

### Celda D-cot @ 0.50 — DESGATEADA (enmienda 2026-07-28b)

**Escrito con el bloque 1A en vuelo y ANTES de que exista ningún resultado de D-cot @0.75**
— la condición que hacía legítimo el gate original (decidir sin mirar el desenlace) se
cumple igual acá, y queda constatado por la fecha.

Estaba gateada con el argumento de que 0.75 **domina** a 0.5 en ARC/gsm8k, así que gastar
21 min en una dosis dominada sólo valía si 0.75 no alcanzaba. **Ese argumento estaba mal
planteado**: mide el valor de 0.5 *como dosis a elegir*, cuando lo que hace falta acá es
0.5 *como segundo punto de una curva*.

Con `cot` únicamente a 0.75 el diseño queda asimétrico — **la curva del lift tiene tres
puntos y la del daño tiene uno**. Si los loops caen de 146 a, digamos, 40, un solo punto no
distingue "0.5 los llevaría a cero" de "0.75 ya es el piso". Y la pregunta que originó todo
esto (*¿cuál es el mejor all-arounder?*) es un **trade-off**, que exige las dos caras a la
**misma** dosis. El costo marginal son 21 min sobre un bring-up ya pago.

`cot` @0.50 sobre los mismos 300 pasa a **incondicional**. El `cold` de los 300 se obtiene
restringiendo por `doc_id` el `cold` de los 1000 — no hace falta correrlo aparte.

### `direct` a dosis — DESCARTADO, con el número

Se evaluó y **no se corre a ninguna dosis**. En MMLU-Pro el canal está roto por el número de
opciones: bajo `rescore_direct_strict` el base da **0.1820** con bound **[0.182, 0.927]** y
**745/1000 sin parsear** (10 opciones ⇒ diez chances de que el matcher por containment tome
una mencionada al pasar). Un canal cuyo bound abarca 74 puntos no se vuelve declarable
porque se le agregue un eje de dosis. Ya estaba declarado provisional en la enmienda
2026-07-27b; esto lo cierra.

**Si las dos caras se separan también acá**, el marco del capítulo generativo cambia: el
daño declarable pasa a ser **una propiedad de la sobredosis**, no del adapter. Eso **no**
reabre la transferencia — a ninguna dosis hay *beneficio* en generación, y el paper ya lo
dice con cuidado ("no dose rises above base beyond noise"). Matiza el **negativo**, que es
distinto y hay que escribirlo así.

⚠ **Relación con S1, declarada para que no se lea como redundante:** dosis y supresión
contestan preguntas distintas. La dosis pregunta *¿es evitable?* (relevante para producto y
para el marco del paper); la supresión pregunta *¿el daño **ES** el loop?* (mecanístico,
causal). Si 0.75 elimina los loops, la dosis es la explicación **más simple**, y S1 sigue
siendo el único que ataca la causalidad. Ninguna reemplaza a la otra.

⚠ **Sobre subir el cap (celdas C), con el dato en contra:** el paper ya reporta que el daño
de gsm8k es *"flat across a generation-budget sweep"* — más presupuesto no recuperó nada
allá. En MMLU-Pro la pregunta del cap sigue viva por **otro** motivo: el base está contra el
cap en 22.5% de sus ítems y en 100% de sus unparsed, así que el cap **subestima al base**, no
limita al adapter. Por eso C1/C2 bajan a tercera prioridad y no se cortan del plan.

## 5. Las celdas

Supresión = `--no-repeat-ngram 8`. Restricción dura sobre repetición; preferida a
`repetition_penalty`, que re-pesa toda la distribución y cambiaría el brazo entero en vez
de sólo prohibir el loop.

| celda | brazo | cap | supresión | para qué | ~tiempo |
|---|---|---|---|---|---|
| **G1** | base | 640 | 8 | **guarda de inocuidad — corre PRIMERO** | 21 min |
| **S1** | v7mc | 640 | 8 | **la celda de interés** | 21 min |
| **C1** | base | 1024 | off | ¿cuánto del 0.6333 era techo de presupuesto? | 34 min |
| **C2** | v7mc | 1024 | off | ¿el presupuesto extra sólo compra más loop? | 34 min |
| **S2** | v7mc | 1024 | 8 | la combinación (opcional) | 34 min |

Bloque 1A = G1+S1 (~45 min, contesta la pregunta). 1B = C1+C2 (~70 min, contesta la del
cap). 1C = S2 (~35 min, opcional).

### G1 es la guarda, y sale del hallazgo de R12

**El base tiene 0 loops de 1000.** Entonces `no_repeat_ngram 8` no tiene nada que suprimir
en el base: es un **no-op esperado**. Si mueve la accuracy del base, la intervención hace
algo más que matar loops y **ningún contraste posterior es atribuible**.

- **Criterio:** G1 debe dar **0.6333 ± 2.0 pp** (el base @640 sobre estos 300 ítems).
- La tolerancia es ±2.0 pp y no ±1: n=300 tiene SE binomial ~2.8 pp, pero esto es el
  **mismo modelo, greedy, mismos ítems**, así que la única fuente de deriva es batching en
  bf16 y prohibir 8-gramas que el base casi nunca repite. Un movimiento > 2 pp es señal,
  no ruido.
- **Si G1 falla, el bloque para.** Es problema de harness, no resultado. (Precedente: el
  guard de R12 abortó un run por falta de `--apply-chat-template` y ahorró ~2 h de GPU.)
- Segundo criterio, del §8 de P-gate: si `n_unparsed` de G1 supera al del base @640 por más
  de **5 pp**, la supresión rompe formato y G1 tampoco sirve como guarda.

## 6. Lecturas pre-declaradas

Contrastes **pareados por ítem** (McNemar exacto + bootstrap pareado 10k, seed 0), dentro de
estrato; la marginal por IPW con bootstrap **estratificado**.

| contraste | qué decide | predicción registrada |
|---|---|---|
| **G1 − base@640** | ¿la intervención es inocua? | **≈ 0.** Es una guarda, no una lectura |
| **S1 − v7mc@640**, estrato A | **la pregunta** | si sube hacia ~0.5753 → el daño es **emisión**; si queda cerca de 0.0616 → **razonamiento** |
| **S1 − base@640**, estrato A | ¿recupera *todo*? | sin prior. Un cierre parcial es el resultado más probable y hay que poder decirlo |
| **S1 − v7mc@640**, estrato B | inocuidad donde no había loop | **≈ 0** |
| **S1 − v7mc@640**, IPW | el efecto marginal sobre los 1000 | derivado de A y B, no independiente |
| **C1 − base@640** | ¿el 0.6333 del base era techo de presupuesto? | **sube.** 22.5% de los ítems del base llegaban al cap y 100% de sus unparsed |
| **C2 − v7mc@640** | ¿más presupuesto ayuda al adapter? | **≈ 0 o peor**: su cola son loops, y un loop con más presupuesto es más loop |
| **C1 vs C2** | ¿el cap favorece asimétricamente al base? | si C1 sube y C2 no, el contraste de KR3 se vuelve **MÁS** desfavorable al adapter |

**Regla de honestidad sobre C1, declarada antes de correr:** si el base sube con cap 1024,
el −6.10 pp de KR3 estaba **subestimando** el daño, porque el base competía contra su propio
techo de presupuesto. Ese resultado va al paper aunque empeore el número que ya publicamos.

**Regla anti-falso-positivo:** un S1 que suba en el estrato A **y también** en el B no es
"recuperación del loop" — es que `no_repeat_ngram` mejora la generación en general, y
entonces el contraste no aísla nada. B es el control que lo detecta.

## 7. Telemetría obligatoria por celda

`loop_rate` (distinct-4), `n_unparsed`, `tok_median`, `n_at_cap`, y `distinct-4` del peor
ítem. Una celda con supresión activa cuyo `loop_rate` **no** baje respecto de su gemela sin
supresión no midió la intervención que dice medir y se reporta como no concluyente.

## 8. Confounds declarados

1. **`no_repeat_ngram 8` no es "suprimir el loop", es prohibir repetir 8-gramas.** Un
   modelo al que se le prohíbe repetir puede escribir algo peor en vez de algo mejor, o
   evitar el loop y quedar igualmente unparsed. La telemetría lo vigila; no lo elimina.
2. **El estrato A tiene selección de dificultad** (§2). Las lecturas dentro de A son
   pareadas contra el **mismo** ítem, así que la dificultad se cancela en el contraste
   `S1 − v7mc@640`; **no** se cancela en `S1 − base@640`.
3. **Una tarea (MMLU-Pro), un brazo contrastivo (v7mc), una familia (Gemma 4 31B).**
   `ce_s0` tiene el doble de loops (298) y sería la réplica natural; queda fuera por
   presupuesto y se declara como tal.
4. **n=300, y el estrato B es 154.** Un efecto de inocuidad menor a ~5 pp en B no se
   distingue de cero.
5. **El cap de 1024 no es "sin límite".** C1/C2 mueven el techo, no lo quitan.

## 9. Artefactos

- `runs/loop_ctl/cell_<name>.json` — per-ítem por celda (mismo schema que R12).
- Análisis **LOCAL**: `scripts/analyze_loop_control.py` (CPU en el pod desperdicia €/GPU).
- Item file congelado y **commiteado**: `runs/paired_gen/items_mmlu_pro_loopctl.json`.

## 10. Lo que este control NO responde

- Si el mismo mecanismo explica el **−24.2 pp de gsm8k** (otra tarea, otro harness). Si S1
  cierra a favor, ésa es la corrida siguiente, y es la que convierte a P-gate en publicable
  per la regla de decisión del handoff.
- Si la degeneración es causa o síntoma de un razonamiento que no converge. Suprimir el
  loop mide si la accuracy se recupera; no dice por qué el modelo entraba en loop.
