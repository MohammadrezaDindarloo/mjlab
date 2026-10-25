"""Sparse SGS reward, paper Eq. 5 (Step 4).

r = 100*1[success] - 1*1[failure] - 1e-5*C(|a|^2) - 1e-4*C(|a_t - a_{t-1}|^2)
    - 1e-4*C(|qdot_arm|^2),  with C(x) = min(x, 1e4).
"""
