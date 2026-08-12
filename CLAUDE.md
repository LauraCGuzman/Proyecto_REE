# Reglas de esta sesión — refactor REE Fase 1

Este repo está en refactor siguiendo `PLAN_REFACTOR_REE_FASE1.md` (pliego externo,
no versionado aquí). Reglas fijas para cualquier trabajo de refactor en este repo:

- **Al mover funciones a `src/`: traslado literal.** No cambiar firmas, valores por
  defecto ni comportamiento. Si algo pide mejora, se anota y se pregunta — no se
  aplica por iniciativa propia.
- **Si un número del gate se mueve, se para y se reporta. NUNCA se ajusta el
  baseline.** El JSON de baseline (`modelos/baseline_numeros.json`, movido desde
  `reports/` en Fase 1.5) es la verdad congelada; una discrepancia significa que
  el refactor rompió algo, no que el baseline esté desactualizado.
- **No borrar ni mover `sandbox/` sin confirmación explícita de Laura.**
- **No editar notebooks leyendo el JSON completo.** Usar `nbformat` y localizar
  celdas por la cadena de búsqueda que indica el pliego, nunca por índice numérico
  (los índices rotan al borrar/fundir celdas).
- **Tier 2 y los bloqueantes B1/B2 requieren autorización explícita de Laura.**
  No se ejecutan por iniciativa propia, aunque parezcan la continuación lógica del
  trabajo en curso.
