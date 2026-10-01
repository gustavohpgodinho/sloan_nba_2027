# -*- coding: utf-8 -*-
# =========================================================================
# Painel 2x2 real vs simulado — equivalente ao geom_density(alpha = 0.35)
# do ggplot: poligono preenchido translucido, com a curva por cima.
# Conta como UMA figura. Cole no 22_graficos_sloan.ipynb.
#
# Usa scipy.gaussian_kde, que e exatamente o que o pandas chama por tras do
# kind="density". A unica diferenca e que aqui a curva e preenchida.
#
# Nao mexe em moldura, titulo nem eixo y: o seu estilo prevalece.
# =========================================================================
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde, wasserstein_distance

COR_SIM, COR_REAL = "C0", "C1"
ALPHA = 0.35

# ------------------------------------------------------------------ dados
sim_nplays = (df_num_plays_
              .query("period <= 4")
              .reset_index(drop=True).reset_index()
              .assign(parte=lambda d: d.groupby(["seasondata", "sim_id", "period"])["index"].rank())
              .groupby(["seasondata", "sim_id", "parte"], as_index=False)["size"].sum()["size"])

real_nplays = (dict_real_values["nplays_period"]
               .query("period <= 4")
               .groupby("game_id")["size"].sum())

PARES = [
    dict(titulo="Home team final score", xlabel="Points",
         real=dict_real_values["final_points"]["hpts"],
         sim=df_final_points_["home_points"], minimo=None),
    dict(titulo="Final score difference", xlabel="Home minus away points",
         real=dict_real_values["final_points"]["dif"],
         sim=df_final_points_["dif"], minimo=None),
    dict(titulo="Lead changes per game", xlabel="Number of lead changes",
         real=dict_real_values["lead_changes"]["real_lead_changes"],
         sim=df_lead_changes_["lead_changes"], minimo=0),
    dict(titulo="Events per game", xlabel="Number of play-by-play tokens",
         real=real_nplays, sim=sim_nplays, minimo=None),
]

# ------------------------------------------------------------------ figura
fig, axes = plt.subplots(2, 2, figsize=(12, 7))

for ax, p in zip(axes.ravel(), PARES):
    real = np.asarray(p["real"], dtype=float)
    sim = np.asarray(p["sim"], dtype=float)

    j = np.concatenate([real, sim])
    lo, hi = np.quantile(j, 0.002), np.quantile(j, 0.998)
    if p["minimo"] is not None:
        lo = max(lo, p["minimo"])
    grade = np.linspace(lo, hi, 400)

    d_sim = gaussian_kde(sim)(grade)
    d_real = gaussian_kde(real)(grade)

    ax.fill_between(grade, d_sim, color=COR_SIM, alpha=ALPHA, lw=0,
                    label="Simulated games")
    ax.plot(grade, d_sim, color=COR_SIM, lw=1.6)

    ax.fill_between(grade, d_real, color=COR_REAL, alpha=ALPHA, lw=0,
                    label="Real games")
    ax.plot(grade, d_real, color=COR_REAL, lw=1.6)

    w = wasserstein_distance(real, sim)
    ax.text(0.975, 0.95, "Wasserstein = %.2f" % w, transform=ax.transAxes,
            ha="right", va="top", fontsize=9)
    ax.text(0.975, 0.865, "Real: %.1f (%.1f)" % (real.mean(), real.std()),
            transform=ax.transAxes, ha="right", va="top", fontsize=8.5,
            color=COR_REAL)
    ax.text(0.975, 0.785, "Simulated: %.1f (%.1f)" % (sim.mean(), sim.std()),
            transform=ax.transAxes, ha="right", va="top", fontsize=8.5,
            color=COR_SIM)

    ax.set_title(p["titulo"])
    ax.set_xlabel(p["xlabel"])
    ax.set_ylabel("Density")
    ax.set_xlim(lo, hi)
    ax.set_ylim(0, max(d_sim.max(), d_real.max()) * 1.30)

handles, labels = axes.ravel()[0].get_legend_handles_labels()
fig.legend(handles, labels, loc="lower center", ncol=2)
fig.tight_layout(rect=[0, 0.06, 1, 1])
fig.savefig("figura_sloan_real_vs_simulado.png", dpi=200, bbox_inches="tight",
            facecolor="white")
plt.show()

# ------------------------------------------------------------------ numeros

for p in PARES:
    r, s = np.asarray(p["real"], float), np.asarray(p["sim"], float)
    print("%-26s W = %6.2f | real %7.2f +-%5.2f | sim %7.2f +-%5.2f | n %6d / %7d"
          % (p["titulo"], wasserstein_distance(r, s),
             r.mean(), r.std(), s.mean(), s.std(), len(r), len(s)))
