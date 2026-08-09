# src/modeling.py
"""Barrido de hiperparámetros sobre validación, usado por todas las rejillas
de modelo_demanda.ipynb. Traslado literal desde el notebook (antes definido
en la celda de la rejilla nivel/5f y reutilizado desde ahí en las demás).

_TEST_MEDIDO se queda en el notebook, no aquí: su semántica es "una lectura
de test por pasada del notebook", y en un módulo importado el reinicio deja
de ser observable en cada Run All.
"""
import itertools

import pandas as pd
from sklearn.metrics import mean_absolute_error
from sklearn.tree import DecisionTreeRegressor


def barrido_validacion(features, objetivo, train_fit, val,
                       rejilla_depth, rejilla_leaf, semilla=42):
    """Rejilla ordenada por MAE de VALIDACIÓN. Test no aparece dentro de esta función.

    objetivo: callable df -> Series.
        nivel -> lambda d: d["demanda_real"]
        delta -> lambda d: d["demanda_real"] - d["demanda_lag_24"]
    """
    y_fit, y_val = objetivo(train_fit), objetivo(val)
    filas = []
    for depth, leaf in itertools.product(rejilla_depth, rejilla_leaf):
        # 'est', no 'modelo': que la variable del bucle no sobreviva y se cuele
        # en una predicción posterior (la trampa que este nombre evita).
        est = DecisionTreeRegressor(max_depth=depth,
                                    min_samples_leaf=leaf,
                                    random_state=semilla)
        est.fit(train_fit[features], y_fit)
        pred_val = est.predict(val[features])
        filas.append({
            # params como dict en la propia fila: si max_depth=None viaja por una
            # columna se convierte en NaN, la columna pasa a float y el ganador
            # vuelve como float donde sklearn espera int o None.
            "params": {"max_depth": depth, "min_samples_leaf": leaf},
            "max_depth": str(depth),
            "orden": depth if depth is not None else 10**6,
            "min_samples_leaf": leaf,
            "mae_fit": mean_absolute_error(y_fit, est.predict(train_fit[features])),
            "mae_val": mean_absolute_error(y_val, pred_val),
            "sesgo_val": (y_val - pred_val).mean(),
        })
    df_rejilla = pd.DataFrame(filas)          # df_rejilla, no df: no pisar el nombre genérico
    df_rejilla["gap"] = df_rejilla["mae_val"] - df_rejilla["mae_fit"]
    return df_rejilla.sort_values("mae_val").reset_index(drop=True)


def dispersion_top(ranking, k=5):
    """Separación entre el ganador y el k-ésimo. Si sale ~0 se está eligiendo
    entre empates y la dimensión barrida no está midiendo nada."""
    top = ranking["mae_val"].head(k)
    return top.max() - top.min()
