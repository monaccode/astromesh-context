<p align="center">
  <img src="https://raw.githubusercontent.com/monaccode/astromesh-context/main/assets/logo.png" alt="Astromesh Context" width="640">
</p>

# astromesh-context

**Decidí qué entra en la ventana de contexto de tu agente, y demostrá que fue la decisión correcta.**

[![astromesh-runtime](https://img.shields.io/badge/astromesh-runtime-f472b6)](https://monaccode.github.io/astromesh/#ecosystem)
[![Estado: en diseño](https://img.shields.io/badge/estado-en%20dise%C3%B1o-fb7185)](#estado)
[![Python](https://img.shields.io/badge/python-3.12%2B-3776ab)](https://www.python.org/)
[![Licencia: Apache-2.0](https://img.shields.io/badge/licencia-Apache--2.0-blue)](https://github.com/monaccode/astromesh-context/blob/main/LICENSE)

*[Read in English](https://github.com/monaccode/astromesh-context/blob/main/README.md)*

> **En diseño: todavía no hay nada para instalar.** Este README describe el diseño contra el
> que se está construyendo la v0.1. Los ejemplos de código muestran la API planeada. La
> [hoja de ruta](#hoja-de-ruta) dice qué sale primero, y este aviso desaparece cuando salga.

La respuesta de un agente es tan buena como lo que el modelo lee antes de responder. La
recuperación trae candidatos, una base de conocimiento trae conceptos curados, la memoria
trae lo que el agente aprendió y la conversación no para de crecer. Alguien tiene que decidir
**qué de todo eso entra en la ventana, en qué orden y qué queda afuera**. Hoy esa decisión
está repartida entre plantillas de prompt, un `top_k` por acá y un recorte de historial por
allá, y nadie puede decirte si fue buena.

`astromesh-context` es esa decisión, tomada en un solo lugar:

- **Un presupuesto para todas las fuentes.** Los resultados de RAG, los conceptos OKF, los
  hechos de memoria y el historial compiten por los mismos tokens con un puntaje calibrado,
  en vez de recibir cada uno una porción fija.
- **Una traza por cada ítem.** Todo lo que propuso una fuente vuelve marcado como incluido o
  excluido, con el motivo: `budget`, `duplicate_of:…`, `below_floor`, `source_cap`,
  `deprecated`.
- **Un orden pensado para el prompt cache.** Lo estable va primero, idéntico byte a byte, para
  que las llamadas consecutivas compartan el prefijo más largo posible.
- **Una evaluación que lo demuestra.** `ContextEval` corre las mismas preguntas con distintas
  políticas y reporta exactitud, tokens gastados y reuso del prefijo, lado a lado.

El núcleo va a depender solo de PyYAML. [astromesh-rag](https://github.com/monaccode/astromesh-rag)
y [astromesh-okf](https://github.com/monaccode/astromesh-okf) se enchufan como fuentes con
extras opcionales.

## El mapa: seis preguntas sobre el contexto

Contexto y conocimiento para agentes no es una técnica. Son seis preguntas, y cada una tiene
sus patrones. Así está parada cada librería de Astromesh.

| Pregunta | Patrones | En Astromesh |
|---|---|---|
| **¿Qué fragmento sirve?** (recuperación) | naive, hybrid, multi-query, HyDE, reranking, contextual retrieval, parent-document | [astromesh-rag](https://github.com/monaccode/astromesh-rag) |
| **¿Por dónde sigo?** (navegación) | índice en el system prompt, búsqueda agéntica con tools, llms.txt, progressive disclosure | [astromesh-okf](https://github.com/monaccode/astromesh-okf) |
| **¿Cómo se relaciona?** (grafos) | GraphRAG, grafos de entidades, grafos temporales | todavía no |
| **¿Qué aprendí?** (memoria) | conversacional, hechos semánticos, episódica, memoria autogestionada, consolidación | core de Astromesh; entra acá como fuente |
| **¿Qué entra en la ventana?** (ingeniería de contexto) | presupuesto de tokens entre fuentes, compactación, offloading de resultados de tools, orden estable para el cache, deduplicación, lost-in-the-middle | **astromesh-context** |
| **¿Le puedo creer?** (procedencia) | citas, frescura, verificación, fuentes en conflicto | señales de confianza y frescura de okf, respetadas acá |

Las dos primeras deciden *qué podría ser relevante*. Esta librería decide *qué lee
realmente el modelo*.

## Cómo se arma una ventana

```mermaid
%%{init: {"theme": "base", "themeVariables": {"primaryColor": "#e0f2fe", "primaryTextColor": "#0a2463", "primaryBorderColor": "#1d4ed8", "lineColor": "#1d4ed8"}}}%%
flowchart TB
    Q([consulta]) --> S1[RAGSource]
    Q --> S2[OKFSource]
    Q --> S3[HistorySource]
    Q --> S4[tu Source]
    S1 --> C[calibrar puntajes por fuente]
    S2 --> C
    S3 --> C
    S4 --> C
    C --> L[LLMScorer]:::llm
    L --> D[deduplicar]
    D --> P[política: llenar el presupuesto]
    P --> O[ordenar para el cache]
    O --> W([ventana y traza])
    classDef llm fill:#ede9fe,stroke:#7c3aed,stroke-dasharray: 5 4
```

Las fuentes corren en paralelo y solo **proponen**: ninguna conoce el presupuesto. Todo lo que
viene después es una función pura, así que los mismos candidatos con la misma política dan
siempre la misma ventana. El paso punteado es opcional y es el único que llama a un modelo.

1. **Calibrar.** Un coseno de 0.82, un BM25 de 14 y "el tercer turno más reciente" no son
   comparables. Los puntajes se normalizan por rango dentro de cada fuente y se ponderan por
   cuánto confiás en esa fuente. Las señales los ajustan a la vista: un concepto vencido
   puntúa menos, uno revisado por una persona puntúa más y uno deprecado queda afuera.
2. **Puntuar con un modelo (opcional).** `LLMScorer` vuelve a puntuar los mejores candidatos
   en una sola llamada por lote y mezcla su puntaje con el calibrado. Si falla o se pasa del
   tiempo, la ventana se arma igual con los puntajes calibrados y la traza lo dice.
3. **Deduplicar.** Copias exactas por hash y casi-copias por shingles de palabras. Esto es lo
   que atrapa a RAG y OKF trayendo la misma política dos veces.
4. **Llenar el presupuesto.** Lo fijado (el system prompt, el índice raíz de OKF) entra
   primero. Después un mínimo por fuente (el último turno del usuario, siempre) y el resto por
   valor por token, respetando un tope por fuente y un puntaje mínimo.
5. **Ordenar para el cache.** Primero lo estable en un orden fijo, después lo de la sesión y
   al final lo volátil, con lo más fuerte en los extremos de cada bloque, donde los modelos
   leen mejor. El historial queda cronológico y al final.

## API planeada

```python
from astromesh_context import ContextEngine, ScoredKnapsack, StaticSource, HistorySource
from astromesh_context.sources.rag import RAGSource
from astromesh_context.sources.okf import OKFIndexSource, OKFSource

engine = ContextEngine(
    sources=[
        StaticSource(system_prompt, kind="instruction", pinned=True),
        OKFIndexSource("./kb"),            # el índice del bundle, fijado en el prefijo estable
        OKFSource("./kb", limit=10),       # conceptos completos, con confianza y frescura
        RAGSource(pipeline, k=20),         # chunks de un pipeline de astromesh-rag
        HistorySource(messages),           # la conversación; el último turno siempre entra
    ],
    policy=ScoredKnapsack(),
    budget=8_000,
)

window = await engine.build("¿Me devuelven la plata después de 3 semanas en el plan Team?")
messages = window.to_messages()            # formato OpenAI o Anthropic
print(window.trace.explain())              # qué entró, qué quedó afuera y por qué
```

El mismo motor desde YAML:

```yaml
apiVersion: astromesh/v1
kind: ContextPolicy
metadata:
  name: support
spec:
  budget: 8000
  policy: scored_knapsack
  sources:
    - { type: okf_index, bundle: ./kb, pinned: true }
    - { type: okf, bundle: ./kb, limit: 10, weight: 1.2 }
    - { type: rag, pipeline: support-docs, k: 20, max_tokens: 3000 }
    - { type: history, min_items: 2 }
```

## Políticas

Una política decide cómo los candidatos llenan el presupuesto. Todas comparten la misma
firma, así la evaluación las compara de igual a igual.

| Política | Qué hace | Para qué |
|---|---|---|
| `ScoredKnapsack` | Un presupuesto, puntajes calibrados, valor por token, topes y pisos por fuente | La opción por defecto |
| `FixedSlots` | Una porción fija del presupuesto por fuente, llenada por prioridad | La línea base que usa todo el mundo hoy |
| `Everything` | Todos los candidatos, o un error claro si no entran | El control: cuánto cuesta "meter todo" |

`LLMScorer` no es una política: es un paso de puntuación que funciona con cualquiera de ellas.

## Dentro del loop del agente

Dos piezas opcionales para agentes de larga duración. Usan los mismos tipos y no necesitan el
motor: cada una trabaja sobre una lista común de mensajes de OpenAI o Anthropic.

- **`Compactor`** resume los turnos viejos cuando el historial pasa un umbral y deja los
  últimos turnos textuales. Nunca separa una llamada a una tool de su resultado, porque los
  proveedores lo rechazan. Es incremental: el resumen anterior alimenta el siguiente.
- **`Offloader`** mueve a un store un resultado de tool que pasa el límite y deja en su lugar
  una vista previa y un handle. El agente lee lo que necesita con
  `ctx_read(handle, offset, limit)` y `ctx_grep(handle, pattern)`.

## Demostrarlo: `ContextEval`

Una estrategia de contexto es una afirmación sobre exactitud y costo, así que tiene que venir
con números. `ContextEval` pasa un set de preguntas etiquetadas por varias configuraciones y
reporta, para cada una, exactitud de la respuesta, tokens por llamada, latencia y cuánto del
prefijo del prompt comparten las llamadas consecutivas. La primera comparación publicada van
a ser las cuatro configuraciones de arriba (`Everything`, `FixedSlots`, `ScoredKnapsack` y
`ScoredKnapsack` con `LLMScorer`) sobre el corpus etiquetado que trae astromesh-rag.

## Integraciones

La v0.1 apunta a LangChain/LangGraph (un nodo antes del modelo, y las piezas del loop como
`pre_model_hook`), PydanticAI (`history_processor`) y un servidor MCP que expone `ctx_build`,
`ctx_read` y `ctx_grep`. Los demás frameworks que soporta astromesh-okf vienen después.

## Hoja de ruta

**v0.1:** `ContextEngine`; las tres políticas; calibración, deduplicación y orden para el
cache; `LLMScorer`; fuentes para RAG, OKF, historial, contenido estático y callables;
`Compactor` y `Offloader`; `ContextEval` con una primera comparación publicada; las tres
integraciones de arriba; YAML `kind: ContextPolicy`.

**Después:** más integraciones con frameworks; una fuente de grafos; adopción por el core de
Astromesh, donde reemplaza el recorte de historial y el prefetch por plantillas de hoy.

## Estado

En diseño. El repositorio tiene el logo y este README; el diseño escrito de la v0.1 se está
terminando, y el código viene después. Seguí el [ecosistema de Astromesh](https://monaccode.github.io/astromesh/#ecosystem)
para enterarte del primer release.

## Licencia

Apache-2.0.
