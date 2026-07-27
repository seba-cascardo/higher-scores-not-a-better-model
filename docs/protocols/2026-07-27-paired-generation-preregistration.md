# Pre-registro — evaluación pareada de generación (R12), pod pre-release

**Fecha de registro: 2026-07-27, ANTES de correr.** Commit de registro = el que
introduce este archivo. Nada de lo de acá se ajusta después de ver los números; si un
umbral resulta mal elegido, se reporta que se eligió mal y se declara el resultado bajo
el umbral original.

## Por qué se corre

El título del paper afirma *"Does Not Transfer to Generation"*. Lo que hoy lo sostiene es
evidencia negativa en canales separados (gsm8k $-24.2 \pm 7.3$ pp / 3 seeds, MMLU $-4.2$,
GPQA cold-MC $\sim$0 contra CoT 0.70 del base) más el árbitro, que prueba **alcance
generativo del base**, no una comparación pareada base-vs-adapter. Una auditoría externa
marcó que diferir el experimento directo y conservar el claim universal es incoherente.
La decisión del user (2026-07-27) fue **correr el pareado pre-release** en lugar de
estrechar el título.

## Diseño

Mismos ítems, todos los brazos, cuatro canales, evaluación **pareada** (por ítem, no por
media agregada).

### Brazos (4)

| brazo | qué es | seeds |
|---|---|---|
| `base` | Gemma 4 31B IT sin offsets | — |
| `v7mc` | el adapter contrastivo canónico (el del $+35$) | 1 (el publicado) |
| `ce_same_param` | control same-parameterization: mismos 48 head-pairs, mismos 12.336 parámetros, CE ordinaria | **3 en ARC-CoT**, 1 (seed 0) en el resto |
| `align_lora` | control de alta capacidad, **un solo rank** (el de la meseta $\sim$62%) | 1 |

Los 3 seeds del control CE van **sólo en el canal CoT de ARC**, que es donde la regla de
la sigma del proyecto muerde (ningún contraste cross-condición sin su $\sigma$). En el
resto, seed 0. Esto se declara acá para que nadie lo lea después como cherry-picking.

### Canales (4)

| canal | qué se mide | coste medido (batch 1, este proyecto) |
|---|---|---|
| `cold` | scoring log-lik por opción, la cara del $+35$ | 354 ms/ítem |
| `letter` | un forward greedy, "respondé sólo la letra" | 181 ms/ítem |
| `direct` | respuesta libre corta, sin CoT | ~250 ms/ítem (est.) |
| `cot` | "pensá paso a paso", parseo de la letra final | **~8.9 s/ítem**, ~285 tokens |

### Benchmarks y tamaños

| benchmark | n | canales | por qué |
|---|---|---|---|
| ARC-Challenge **full test split** | **1172** | los 4 | donde viven el $+35$ y el árbitro; el full split ya es el del árbitro re-corrido ($n{=}1172$, 444 arreglados, base 442/444 en CoT) |
| MMLU-Pro **estratificado** | **1000** | los 4 | el único sitio donde el base **no** está en su techo de CoT $\Rightarrow$ el único donde el adapter *podría* ganar |
| HellaSwag muestra | 1000 | `cold`, `letter` | — |
| Winogrande full | 1267 | `cold`, `letter` | — |

**HellaSwag y Winogrande no llevan CoT ni `direct`, y no es un recorte de presupuesto:**
son tareas de plausibilidad de continuación, no preguntas. "Razoná paso a paso y después
elegí el final más plausible" no es el mismo objeto cognitivo que en ARC, y un número así
no sería interpretable como transferencia a generación.

**Estratificación de MMLU-Pro (fijada antes de correr):** muestra proporcional por
categoría sobre el split de test, `seed=0`, con los `doc_id` congelados a un JSON
self-contained **antes** de la primera corrida. Ningún brazo re-carga el dataset (evita
el drift de slice que el proyecto ya pagó, y el gotcha de `enumerate()` de GPQA).

### Presupuesto

~6,5 h de GPU. El cuello es CoT: $2172$ ítems con CoT $\times$ 4 brazos $\approx 8700$
generaciones, más 2 seeds extra del control CE en ARC.

## Kill-rules

Se evalúan **pareados por ítem** con bootstrap pareado ($10^4$ resamples) sobre los
mismos `doc_id`. Todo contraste cross-condición se reporta con su $\sigma$.

### KR1 — el central: ¿el adapter pierde en generación sobre los ítems que "arregla"?

Sobre los **444 ítems de ARC que `v7mc` arregla en cold scoring**, comparar `base` vs
`v7mc` en el canal `cot`.

- **El claim sobrevive** si `v7mc` queda **por debajo** del `base` con CI pareado que
  **excluye el cero**.
- **El claim CAE** si el CI incluye el cero o `v7mc` queda por encima. En ese caso
  *"does not transfer"* pierde su pata más fuerte y **el título se estrecha igual**, a
  *"does not improve the measured generation channels"*.

⚠ **Asimetría declarada de antemano:** el base ya resuelve 442/444 (0.9955) de esos
ítems con CoT. Está en su techo. Este contraste puede mostrar que el adapter *pierde*;
**no puede** mostrar que gane, porque no hay margen. Se reporta como test de daño, no de
beneficio, y así se escribirá en el paper gane o pierda.

### KR2 — la dominancia del baseline trivial

Sobre los ítems ARC base-wrong, comparar `base` vs `v7mc` en el canal `letter`.

- **§5.7 sobrevive** si el `base` en `letter` sigue por encima del `v7mc` en `cold`.
- **CAE** si `v7mc` supera al `base` en `letter`: entonces la dominancia estricta del
  baseline no es estricta y la Figura 2 se rehace.

### KR3 — el que puede dar vuelta el paper

En **MMLU-Pro**, medir (a) el lift de scoring de `v7mc` sobre `base` en `cold`, y (b) la
diferencia pareada en `cot`.

- **La lectura actual sobrevive** si hay lift en `cold` y **no** lo hay en `cot`
  (CI incluye el cero o es negativo).
- **La tesis CAMBIA** si hay lift en `cold` **y** transfiere a `cot` con CI que excluye
  el cero. Entonces el efecto no es un readout de un benchmark saturado, y el paper
  cambia de **tesis**, no de redacción. Éste es el resultado que más querría ver fallar
  y el que justifica el gasto de GPU.
- **Caso nulo informativo:** si no hay lift de scoring en MMLU-Pro, se reporta como
  frontera del efecto (el $+35$ no generaliza a un benchmark no saturado) — resultado
  publicable y compatible con la tesis, pero acota su alcance y se escribe.

### KR4 — guard de reproducción (aborta el run, no lo interpreta)

Antes de gastar GPU en los cuatro canales, el harness re-mide el `cold` de `base` y
`v7mc` sobre el sub-slice $n{=}400$ ya publicado. Si no reproduce **ARC base 0.4975 /
v7mc dentro de $\pm 1$ pp de lo publicado**, el run **aborta**: hay un problema de
harness (chat template, `max_length`, stop-ids), no un resultado.

> Precedente que motiva este guard: el run E7 abortó por faltarle
> `--apply-chat-template`, y ese aborto salvó ~2 h de GPU (B45).

## Lo que este experimento NO decide

- No mide "toda generación": son cuatro canales sobre cuatro benchmarks.
- No convierte al árbitro en una comparación pareada — son evidencias distintas y se
  reportan por separado.
- No re-abre la geometría off-axis ni §5.9.

## Gotchas que el harness debe respetar

- `--apply-chat-template` **siempre** (el guard KR4 lo cazaría, pero no se corre sin él).
- `max_length` explícito: lm-eval cae a 2048 por el `text_config` anidado de Gemma 4.
- Stop-ids **resueltos desde el tokenizer**, no una constante de familia copiada
  (el zombie B15 nació de `<eos>` pelado; `<end_of_turn>`=106 es el que cierra turno).
- `nohup ... > log 2>&1 &` + `disown`; nunca `tee`/pipe (SIGPIPE silencioso).
- Ítems congelados a JSON self-contained antes del primer brazo; ningún brazo recarga
  el dataset.
- Todo análisis CPU (bootstrap pareado, agregación, tablas) corre **local**, no en el pod.
