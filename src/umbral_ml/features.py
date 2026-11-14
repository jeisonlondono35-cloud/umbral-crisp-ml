"""Feature helpers that are part of a saved Pipeline (they must live in an importable module so the
saved model can be loaded by the Flask app)."""
import numpy as np
import pandas as pd

from .config import HIT_X, MOVES

H_X = ["x_" + x for x in HIT_X[1:]]          # the bias column is constant: the models add their own


def interactions(X: pd.DataFrame) -> pd.DataFrame:
    """attack x situation products: lets a pooled logistic regression learn one slope per attack,
    the same structure as the in-game bandit (one logistic regression per attack). The attack list is
    fixed from the game code (not learned from data: no leakage)."""
    mv = X["mv"].to_numpy()
    xs = X[H_X].to_numpy(dtype=float)
    on = (mv[:, None] == np.array(MOVES)[None, :]).astype(float)          # rows x moves
    prod = (on[:, :, None] * xs[:, None, :]).reshape(len(X), -1)          # rows x (moves*features)
    names = [f"{m}*{c}" for m in MOVES for c in H_X]
    return pd.concat([X, pd.DataFrame(prod, columns=names, index=X.index)], axis=1)
