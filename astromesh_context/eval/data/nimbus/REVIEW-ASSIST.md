# Asistencia para revisar `answers.jsonl` (Nimbus, nivel 2)

> Borrador de apoyo para una revisión humana, generado el 2026-10-20. **No es una revisión**:
> `answers.jsonl` y `SOURCE.md` quedan sin tocar y `reviewed by` sigue en `PENDING`.
> No se commitea; este archivo queda sin trackear.

## Cómo se califica en el nivel 2 (para qué tiene que servir cada referencia)

- `dataset.py::_references()` lee `answers.jsonl` como un mapa `id -> reference`. Cada `EvalCase`
  lleva su `reference` (texto libre, sin campos de hechos clave).
- `runner.py`: el modelo de respuesta recibe `window.to_messages()` (system prompt: "Answer only
  from the context provided. Cite the context id you used.") y su respuesta va a `grade_answer`.
- `judge.py`: **juez LLM**, sin exact match ni key facts. El prompt pide una sola palabra:
  `CORRECT` si la respuesta "states the same facts as the reference", `PARTIAL` si le faltan
  datos pero no se equivoca, `WRONG` en otro caso. Puntaje 1.0 / 0.5 / 0.0; una respuesta que no
  sea exactamente una de esas palabras da 0.0 y cuenta en `judge unparsed`.
- Qué implica para el formato: una referencia larga, con datos secundarios, empuja hacia
  `PARTIAL` a las respuestas cortas que son correctas. Una referencia que **no menciona** un dato
  sensible a la versión (por ejemplo el plazo de la papelera) no deja que el juez castigue una
  respuesta que use la política 2024 reemplazada. Las 60 referencias son prosa en el idioma de la
  pregunta (15 en inglés, 45 en español): el formato está bien.
- Integridad verificada: 60 líneas, los ids coinciden 1:1 y en orden con `queries.jsonl`, y no
  hay duplicados. El sha256 actual
  (`1901bf48…f4b5`) coincide con el fijado en `tests/test_eval_smoke.py`.

Corpus (`corpus.jsonl`, 200 documentos): `err-*` (40), `sku-*` (20) y `faq-*` (50) son vigentes.
`pol-*-2026` (15) están vigentes; `pol-*-2024` (15) están reemplazadas (en el bundle OKF salen
como `status: deprecated` y `superseded_by`). `rn-*` y `blog-*` (60) son ruido
(`trap: noise`). Revisé los 60 documentos de ruido: ninguno contradice un dato de las referencias.

## Resumen

| Clasificación | Cantidad |
|---|---|
| OK | 56 |
| DUDOSA | 4 |
| MAL | 0 |

- Ninguna referencia depende de una política 2024 (sin trampa invertida). Las 15 `q-dist-*` usan
  los valores 2026, que siempre difieren del 2024.
- El riesgo principal no es que haya errores sino que falte discriminación: `q-noise-05` no
  menciona el plazo de la papelera, así que una respuesta con el valor 2024 pasaría como correcta.

## DUDOSA / MAL

| id | Pregunta | Referencia actual | Propuesta | Evidencia |
|---|---|---|---|---|
| q-noise-05 (DUDOSA) | ¿Cómo recupero un archivo que mandé a la papelera? | "Abre el archivo en la papelera de la web y elige Restaurar; vuelve a su carpeta original y conserva sus enlaces." | "Abre el archivo en la papelera de la web y elige Restaurar; vuelve a su carpeta original y conserva sus enlaces. Se puede restaurar mientras siga en la papelera: 60 días, o 120 en los planes de equipo." | `err-4040`: "Ábrelo en la papelera de la web y elige Restaurar". `pol-papelera-2026`: "permanecen en la papelera durante 60 días… En los planes de equipo el plazo es de 120 días". Sin el plazo, una respuesta que diga 30/60 días (`pol-papelera-2024`, reemplazada) no se castiga. Si preferís no agregar un hecho fuera de `relevant` (sólo `err-4040`), dejala igual pero anotalo como limitación. |
| q-noise-03 (DUDOSA) | How do I replay failed webhook deliveries? | "You can replay failed webhook deliveries from Developers > Webhooks once your endpoint has recovered. Failed events are retried with exponential backoff for up to 24 hours." | "Replay them from Developers > Webhooks. Failed events are also retried automatically with exponential backoff for up to 24 hours, and once your endpoint recovers the pending events are delivered in order." | `err-8010` / `err-8011`: "once the endpoint recovers, pending events are delivered in order. You can replay failed deliveries from Developers > Webhooks." La referencia condiciona el replay a que el endpoint se recupere (el doc no lo dice) y omite la entrega automática en orden. |
| q-para-14 (DUDOSA) | soy profe en una escuela, ¿hay precio especial? | "Sí: las instituciones educativas acreditadas con documentación oficial acceden a una tarifa especial, como Nimbus Team para Educación (SKU-63E) a 3 USD por usuario al mes con mínimo 10 usuarios. La validación demora hasta cinco días hábiles…" | "La tarifa especial es para la institución, no para docentes a título individual: las instituciones educativas acreditadas con documentación oficial acceden a Nimbus Team para Educación (SKU-63E), 3 USD por usuario al mes con mínimo 10 usuarios. La validación demora hasta cinco días hábiles y la tarifa se aplica desde la siguiente factura." | `faq-descuento-educativo`: "Las instituciones educativas acceden a una tarifa especial previa acreditación". `sku-63e`: "mínimo 10 usuarios… 3 USD por usuario al mes". El "Sí" inicial puede hacer que el juez marque WRONG una respuesta correcta del tipo "no hay descuento individual para docentes". |
| q-dist-10 (DUDOSA) | Si cancelo el plan pago, ¿cuánto tiempo guardan mis archivos? | "Los archivos que superan la cuota gratuita se conservan 90 días en modo de solo lectura; después se eliminan de forma definitiva, empezando por los más antiguos." | "La cuenta pasa al plan gratuito y conserva lo que entra en la cuota gratuita. Los archivos que la superan se conservan 90 días en modo de solo lectura; después se eliminan de forma definitiva, empezando por los más antiguos." | `pol-conservacion-tras-baja-2026`: "los archivos que superan la cuota gratuita se conservan durante 90 días en modo de solo lectura". `faq-baja-suscripcion`: "la cuenta pasa a la modalidad gratuita". La pregunta dice "mis archivos" y la referencia sólo cubre el excedente. Además, `faq-free-tier-downgrade` dice que el excedente "stays accessible… until usage falls below the limit", sin plazo, y choca con la política (ver inconsistencias). |

## Preguntas ambiguas, solapadas e inconsistencias del corpus

**Ambiguas** (la referencia elige una lectura razonable; no hace falta cambiarla):
- `q-para-10`: "dejarle la cuenta a mi socio". `faq-transferencia-titularidad` cubre sólo
  **cuentas de equipo**, y el socio tiene que ser administrador. El corpus no documenta una
  transferencia de cuenta personal.
- `q-para-12`: "más gigas sin cambiarme de plan". Los complementos son sólo para planes de pago
  (`faq-capacidad-adicional`: "Los planes de pago admiten complementos"). Para un usuario Free la
  respuesta no aplica. Opcional: agregar "(en planes de pago)".
- `q-para-20`: "weird amount". También podría ser un cargo duplicado (`faq-cargo-duplicado`) o un
  prorrateo. La referencia cubre moneda e impuestos, que es lo que marca `relevant`.

**Solapadas** (no son duplicadas; ninguna cae en la misma sesión de 5 casos):
`q-exact-04` / `q-noise-03` (`err-8011`), `q-exact-15` / `q-noise-08` (`err-9002`),
`q-exact-02` / `q-noise-04` (`err-4032`), `q-exact-05` / `q-noise-09` (`sku-77a`),
`q-exact-11` / `q-dist-15` (límites de API diario vs. por minuto),
`q-para-02` / `q-para-19` / `q-dist-10` (pasar a Free).

**Inconsistencias del corpus** (vendorizado, no se edita acá; anotarlas río arriba en astromesh-rag):
- `faq-free-tier-downgrade` ("Content above the free storage limit stays accessible… until usage
  falls below the limit") contra `pol-conservacion-tras-baja-2026` (90 días y después se borra).
  Afecta a `q-para-02`, `q-para-19` y `q-dist-10`.
- `faq-deleted-account-recovery` (14 días de gracia tras pedir la eliminación) contra
  `faq-eliminacion-cuenta` ("Una vez confirmada, la operación es irreversible"). Ninguna pregunta
  lo toca.

## OK (56)

Formato: id: documento, cita (≤ 20 palabras). Notas menores entre corchetes, sin cambios necesarios.

**Exactas**
- q-exact-01: `err-4012`, "un token de sesión que superó los 30 días sin actividad en ese dispositivo".
- q-exact-02: `err-4032`, "Solo un administrador puede ampliar la capacidad del equipo desde la consola".
- q-exact-03: `err-5004`, "llegó después de los 120 segundos que el servidor espera por cada fragmento".
- q-exact-04: `err-8011`, "your endpoint did not answer within 10 seconds… retried with exponential backoff for up to 24 hours".
- q-exact-05: `sku-77a`, "SAML single sign-on and data residency… 5 TB per user… 200 USD per user per year". [omite "in Europe or North America"]
- q-exact-06: `sku-52m`, "1 TB por usuario / mínimo 3 usuarios / 180 días… 9 USD por usuario al mes".
- q-exact-07: `sku-15t`, "10 TB adicionales, se comparte entre los usuarios del plan… 40 USD al mes".
- q-exact-08: `err-2102`, "el archivo editado se guarda con su nombre anterior junto al renombrado".
- q-exact-09: `err-3601`, "Solo el administrador que creó la retención legal puede levantarla".
- q-exact-10: `err-7003`, "Copy the Entity ID shown on that page into the Audience field of your identity provider".
- q-exact-11: `err-4291`, "superó el límite diario… El contador se reinicia a las 00:00 UTC".
- q-exact-12: `sku-34a`, "2 TB compartidos / hasta 6 miembros… 120 USD al año".
- q-exact-13: `err-6121`, "Actualiza la fecha de vencimiento o registra una tarjeta nueva… siguientes 24 horas".
- q-exact-14: `sku-41b`, "Copia de seguridad automática de un servidor completo, con copias cada 15 minutos".
- q-exact-15: `err-9002`, "links stay valid for 7 days after the archive is ready".

**Paráfrasis**
- q-para-01: `faq-cargo-duplicado`, "preautorización que el emisor libera en un plazo de 3 a 5 días hábiles".
- q-para-02: `faq-baja-suscripcion`, "Configuración > Plan > Finalizar suscripción… la cuenta pasa a la modalidad gratuita". [no menciona el excedente de cuota; ver inconsistencias]
- q-para-03: `faq-reembolso-prorrateado`, "se calcula de forma proporcional y se acredita como saldo a favor".
- q-para-04: `faq-medio-de-pago`, "El nuevo medio se utiliza a partir de la siguiente renovación".
- q-para-05: `faq-datos-fiscales`, "razón social, el identificador tributario y el domicilio fiscal… nota de crédito".
- q-para-06: `faq-historial-facturacion`, "Configuración > Facturación > Comprobantes… archivo comprimido por ejercicio".
- q-para-07: `faq-renovacion-automatica`, "La renovación automática puede desactivarse en Configuración > Plan".
- q-para-08: `faq-periodo-gracia` + `faq-suspension-impago`, "período de gracia de 10 días"; "los archivos se conservan y pueden descargarse".
- q-para-09: `faq-reactivacion-cuenta`, "La reactivación es inmediata una vez acreditado el importe".
- q-para-10: `faq-transferencia-titularidad`, "designar a otro administrador como nuevo titular en Consola > Organización > Titularidad". [ver ambiguas]
- q-para-11: `faq-perdida-segundo-factor`, "el ingreso puede completarse con uno de los códigos de respaldo".
- q-para-12: `faq-capacidad-adicional` + `sku-15s` + `sku-15t`, "1 TB adicional… 5 USD al mes"; "10 TB adicionales… 40 USD al mes". [ver ambiguas]
- q-para-13: `faq-descuento-anual`, "se abonan 10 meses y se obtiene el servicio durante 12". [coincide con todos los pares de SKU mensual/anual]
- q-para-15: `faq-licencias-no-asignadas`, "disminuir la cantidad de licencias en Consola > Licencias antes de la renovación".
- q-para-16: `faq-cargo-no-reconocido`, "formulario de transacciones no reconocidas, indicando los últimos cuatro dígitos de la tarjeta".
- q-para-17: `faq-cierre-sesiones-remoto`, "«Finalizar sesión» revoca el acceso de forma inmediata; se recomienda además actualizar la contraseña".
- q-para-18: `faq-upgrade-proration` + `faq-cambio-plan-superior`, "only the prorated difference for the remaining days of the current term".
- q-para-19: `faq-subscription-pause`, "Subscriptions cannot be suspended temporarily… move to the free tier". [el excedente sobre la cuota Free se borra a los 90 días por `pol-conservacion-tras-baja-2026`; opcional mencionarlo]
- q-para-20: `faq-charge-currency` + `faq-impuestos-moneda`, "the issuer applies its own exchange rate and may add a foreign transaction fee".

**Distractores** (todas con el valor 2026; entre paréntesis, el valor 2024 que la trampa ofrece)
- q-dist-01: `pol-reembolsos-2026`, "dentro de los 30 días… anuales y también a las mensuales" (2024: 14 días, sin mensuales).
- q-dist-02: `pol-papelera-2026`, "durante 60 días… En los planes de equipo el plazo es de 120 días" (2024: 30/60).
- q-dist-03: `pol-cuentas-inactivas-2026`, "sin inicios de sesión durante 24 meses… aviso… 30 días antes" (2024: 12 meses/14 días).
- q-dist-04: `pol-contrasenas-2026`, "al menos 12 caracteres y no pueden coincidir con las últimas cinco" (2024: 8).
- q-dist-05: `pol-uso-aceptable-2026`, "hasta 3 TB por día entre todos sus enlaces compartidos" (2024: 1 TB).
- q-dist-06: `pol-registros-acceso-2026`, "dirección IP, dispositivo y fecha… durante 180 días" (2024: 90).
- q-dist-07: `pol-soporte-2026`, "4 horas para Business… Los plazos se cuentan en horas hábiles" (2024: 8 h).
- q-dist-08: `pol-disponibilidad-2026`, "99,95 %… crédito del 25 %… dentro de los 30 días siguientes" (2024: 99,9 %/10 %).
- q-dist-09: `pol-cambios-precio-2026`, "avisamos a los clientes afectados con 60 días de antelación" (2024: 30).
- q-dist-11: `pol-enlaces-compartidos-2026`, "caducan a los 30 días. La caducidad máxima… es de 180 días" (2024: 90/365).
- q-dist-12: `pol-edad-minima-2026`, "hay que tener al menos 16 años" (2024: 13).
- q-dist-13: `pol-subprocessors-2026`, "at least 30 days before engaging a new subprocessor" (2024: 15).
- q-dist-14: `pol-incident-notification-2026`, "within 48 hours of confirming it" (2024: 72).
- q-dist-15: `pol-api-fair-use-2026`, "up to 1,200 requests per minute and 250,000 requests per day… HTTP 429" (2024: 600/100,000).

**Ruido**
- q-noise-01: `faq-activar-dos-pasos`, "se vincula una aplicación de autenticación mediante el código QR… diez códigos de respaldo".
- q-noise-02: `faq-contacto-facturacion` (+ `faq-invoice-delivery`), "dirección de correo adicional, por ejemplo la del área contable".
- q-noise-04: `faq-exceso-cuota` + `err-4031` + `err-4032`, "No se aplican cargos adicionales automáticos por exceso"; "vaciando la papelera, que también cuenta para la cuota".
- q-noise-06: `err-3307` + `err-3308`, "tras cinco intentos fallidos el enlace se bloquea durante 15 minutos". [redacción: "Es probable que el enlace haya caducado"]
- q-noise-07: `err-2999`, "Approve the new device from one that already has access… Nimbus support cannot recover the key".
- q-noise-08: `faq-portabilidad-datos` + `err-9002`, "metadatos de la cuenta, el historial de actividad y el contenido almacenado"; "valid for 7 days".
- q-noise-09: `sku-77a` + `sku-77m`, "everything in Team plus SAML single sign-on". [el corpus no dice si Enterprise (`sku-90e`) incluye SAML]
- q-noise-10: `faq-dispositivos-vinculados`, "El plan Free admite hasta tres dispositivos… no se conectan durante 180 días".

## Pasos para completar la revisión (los hace la persona revisora)

1. **Decidir cada DUDOSA** de la tabla y, si se acepta, editar `answers.jsonl`. Hay que mantener
   una línea por caso, el mismo orden de ids, las claves `id` y `reference`, UTF-8 y **finales de
   línea LF** (`tests/test_eval_judge.py::test_answers_file_covers_every_query` falla si aparece
   `\r`; ojo con los editores de Windows). Recorrer también las 56 OK: esto es una asistencia, no
   reemplaza leerlas.
2. **Actualizar el hash fijado** si cambió `answers.jsonl`: `tests/test_eval_smoke.py`, la clave
   `"answers.jsonl"` de `PINNED`, con el valor de
   `sha256sum astromesh_context/eval/data/nimbus/answers.jsonl`. Si no se actualiza, falla
   `test_vendored_corpus_hash`.
3. **Marcar la revisión** en `SOURCE.md`, en la última línea:
   `- answers.jsonl: drafted from the relevant documents by the implementer; reviewed by: <nombre>, <AAAA-MM-DD>`.
   El test sólo exige que `SOURCE.md` siga conteniendo `a86bfba`.
4. **Verificar**: `uv run pytest tests/test_eval_smoke.py tests/test_eval_judge.py tests/test_cli.py`
   (requiere `astromesh-rag` y `astromesh-okf`; si no están, el smoke se saltea: usar
   `uv run --extra eval pytest …`).
5. **Correr el nivel 2** (`README.md:315`, `README.es.md:323`, `astromesh_context/cli.py`).
   Hacen falta `CTX_EVAL_BASE_URL` (endpoint compatible con OpenAI), `CTX_EVAL_API_KEY` y
   `CTX_EVAL_MODEL` (modelo que responde). `CTX_EVAL_JUDGE_MODEL` es opcional: por defecto usa
   `CTX_EVAL_MODEL`, y además puntúa la fila `scored_knapsack+llm`. La clave no se escribe en el
   reporte, que guarda sólo el host y los ids de modelo.
   ```bash
   CTX_EVAL_BASE_URL=... CTX_EVAL_API_KEY=... CTX_EVAL_MODEL=... CTX_EVAL_JUDGE_MODEL=... \
     uv run --extra eval astromesh-context eval --level 2 --embedder st --out results/
   ```
   En PowerShell: `$env:CTX_EVAL_BASE_URL='...'; $env:CTX_EVAL_API_KEY='...'; $env:CTX_EVAL_MODEL='...'`
   y después el mismo `uv run …`. Para probar el endpoint antes con pocos casos, agregar
   `--subset 5`. Los budgets por defecto son `750,1000,2000` (`--budgets`).
6. **Revisar la salida** (`results/<fecha>-st-L2.md` / `.json`): `judge unparsed` tiene que dar
   0; si no, el juez no responde con una sola palabra y la `accuracy` sale subestimada. El
   `answers_sha256` del JSON tiene que coincidir con el hash nuevo. Después hay que commitear
   `answers.jsonl`, el hash, `SOURCE.md` y los resultados, y completar la sección de nivel 2 del
   README ("se van a agregar cuando se revisen las respuestas de referencia").
