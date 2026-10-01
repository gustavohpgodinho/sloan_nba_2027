# -*- coding: utf-8 -*-
"""
Trajetoria de probabilidades de desfecho — script independente.

NAO usa nada do notebook 21b: monta a grade de tempo, preenche o estado,
cruza com o indexador e calcula o ponto de definicao internamente.

Depende apenas de numpy, pandas e matplotlib.

ENTRADAS
--------
df_jogo : DataFrame de UMA partida real, com as colunas
          period, remaining_time, home_points, away_points
          e, opcionalmente, dif e posse.
          (e uma fatia de df_real_games_YYYY)

df_indexador : DataFrame com
          period, remaining_time, dif, posse, total, home, away, draw
          (e um dos df_indexador_YYYY)

USO
---
    traj = prepara_trajetoria(
        df_real_games_2223.query("game_id == 22200800"),
        df_indexador_2122,
        marginalizar_posse=True)

    figura_trajetoria(traj, periodos=(4,),
                      titulo="Golden State Warriors 119 x 113 Dallas Mavericks",
                      caminho="figura_sloan_trajetoria.png")
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

COR = {"home": "#2a78d6", "away": "#eb6834", "draw": "#1baf7a"}
ROTULO = {"home": "Home win", "away": "Away win", "draw": "Tie at end of regulation"}
DESFECHOS = ["home", "away", "draw"]

DURACAO_PERIODO = 7200      # decimos de segundo
LIMITE_DIF = 25


def _mmss(decimos):
    seg = int(round(float(decimos) / 10.0))
    return "%d:%02d" % (seg // 60, seg % 60)


def marginaliza_posse(df_indexador):
    """Remove o estrato de posse, ponderando pelo numero de simulacoes da celula.

    A trajetoria indexada por posse salta a cada troca de posse, o que numa figura
    pequena le como ruido. Marginalizando, a curva fica lisa e continua sendo
    probabilidade estimada -- de um cenario que nao condiciona em quem tem a bola.
    A legenda da figura precisa dizer isso.
    """
    d = df_indexador.copy()
    for c in DESFECHOS:
        d[c] = d[c] * d["total"]
    g = (d.groupby(["period", "remaining_time", "dif"], as_index=False)
           .agg(dict([(c, "sum") for c in DESFECHOS] + [("total", "sum")])))
    for c in DESFECHOS:
        g[c] = g[c] / g["total"]
    return g


def prepara_trajetoria(df_jogo, df_indexador, periodos=(1, 2, 3, 4),
                       marginalizar_posse=False, limiar=0.95):
    """Grade completa da partida, com as probabilidades e o ponto de definicao."""
    d = df_jogo.copy()
    if "dif" not in d.columns:
        d["dif"] = d["home_points"] - d["away_points"]
    if "posse" not in d.columns:
        d["posse"] = 0.0

    d = d[d["period"].isin(periodos)]
    d = d[["period", "remaining_time", "home_points", "away_points", "dif", "posse"]]
    # um evento por instante: vale o estado apos o ultimo deles
    d = (d.sort_values(["period", "remaining_time"], ascending=[True, False])
           .drop_duplicates(["period", "remaining_time"], keep="last"))

    # grade completa de (periodo, tempo restante)
    grade = pd.DataFrame(
        [(p, t) for p in periodos for t in range(DURACAO_PERIODO, -1, -1)],
        columns=["period", "remaining_time"])

    traj = (grade.merge(d, on=["period", "remaining_time"], how="left")
                 .sort_values(["period", "remaining_time"], ascending=[True, False])
                 .reset_index(drop=True))
    # estado so depende do passado: ffill, nunca bfill
    for c in ("home_points", "away_points", "dif", "posse"):
        traj[c] = traj[c].ffill()
    traj["dif"] = traj["dif"].fillna(0).clip(-LIMITE_DIF, LIMITE_DIF)
    traj["posse"] = traj["posse"].fillna(0)

    idx = marginaliza_posse(df_indexador) if marginalizar_posse else df_indexador.copy()
    chaves = ["period", "remaining_time", "dif"] + ([] if marginalizar_posse else ["posse"])
    for c in chaves:
        traj[c] = traj[c].astype(float)
        idx[c] = idx[c].astype(float)

    traj = traj.merge(idx[chaves + DESFECHOS], on=chaves, how="left")
    traj = traj.sort_values(["period", "remaining_time"],
                            ascending=[True, False]).reset_index(drop=True)
    # celula vazia no indexador: carrega a ultima estimativa conhecida
    for c in DESFECHOS:
        traj[c] = traj[c].ffill()

    # ponto de definicao: primeiro instante a partir do qual o maximo das tres
    # probabilidades permanece >= limiar ate o fim
    acima = (traj[DESFECHOS].max(axis=1) >= limiar).fillna(False).values
    traj["condicao"] = np.minimum.accumulate(acima[::-1])[::-1].astype(int)
    return traj


def figura_trajetoria(traj, periodos=(4,), titulo=None, caminho=None,
                      figsize=(10, 5.5), limiar=0.95,
                      loc_legenda="center left", bbox_legenda=None,
                      y_anotacao=0.55, lado_anotacao="auto", desloc_anotacao=8,
                      intervalo_ticks=600,
                      marcar_mudancas_placar=True, cor_marcas="0.88",
                      lw_marcas=0.7, marcas_nos_dois=False):
    """
    loc_legenda     : qualquer `loc` do matplotlib ("upper left", "lower right",
                      "center", "best", ...). Com bbox_legenda, posiciona livre.
    bbox_legenda    : tupla (x, y) em fracao dos eixos, ex. (1.02, 1) joga a
                      legenda para fora, a direita. None usa so o loc.
    y_anotacao      : altura do texto do ponto de definicao, em probabilidade (0 a 1)
    lado_anotacao   : "auto", "left" ou "right" — de que lado da linha vertical
    desloc_anotacao : afastamento do texto em relacao a linha, em pontos
    intervalo_ticks : espacamento das marcas do eixo x, em decimos de segundo.
                      600 = 1 minuto, 300 = 30 segundos, 1200 = 2 minutos.
    marcar_mudancas_placar : traco vertical de fundo em cada instante em que a
                      diferenca de pontos muda, para o leitor ligar os dois
                      paineis. Vai no zorder 0, abaixo de tudo.
    cor_marcas      : cinza desses tracos. "0.88" e bem claro; "0.80" marca mais.
    marcas_nos_dois : repete os tracos tambem no painel de baixo.
    """
    d = (traj[traj["period"].isin(periodos)]
         .sort_values(["period", "remaining_time"], ascending=[True, False])
         .reset_index(drop=True))

    um = len(periodos) == 1
    x = (d["remaining_time"].values.astype(float) if um
         else ((d["period"] - 1) * DURACAO_PERIODO
               + (DURACAO_PERIODO - d["remaining_time"])).values.astype(float))

    fig, (ax, ax2) = plt.subplots(
        2, 1, figsize=figsize, sharex=True,
        gridspec_kw={"height_ratios": [3, 1], "hspace": 0.12})

    # tracos de fundo nos instantes em que o placar muda
    if marcar_mudancas_placar:
        muda = np.flatnonzero(np.diff(d["dif"].values) != 0) + 1
        for j in muda:
            ax.axvline(x[j], color=cor_marcas, lw=lw_marcas, zorder=0)
            if marcas_nos_dois:
                ax2.axvline(x[j], color=cor_marcas, lw=lw_marcas, zorder=0)

    for c in DESFECHOS:
        ax.plot(x, d[c].values, lw=1.8, color=COR[c], label=ROTULO[c], zorder=3)
    ax.axhline(limiar, color="0.4", lw=1, ls="--", label="0.95 threshold", zorder=2)

    tp = d[d["condicao"] == 1]
    if len(tp):
        i = tp.index[0]
        ax.axvline(x[i], color="0.1", lw=1.3)
        if lado_anotacao == "auto":
            meio = (x.min() + x.max()) / 2
            a_direita = (x[i] < meio) if um else (x[i] > meio)
        else:
            a_direita = (lado_anotacao == "right")
        ax.annotate("Decision point: %s left in Q%d"
                    % (_mmss(d.loc[i, "remaining_time"]), int(d.loc[i, "period"])),
                    xy=(x[i], y_anotacao),
                    xytext=(-desloc_anotacao if a_direita else desloc_anotacao, 0),
                    textcoords="offset points",
                    ha="right" if a_direita else "left", va="center", fontsize=9)

    ax.set_ylim(0, 1)
    ax.set_yticks(np.arange(0, 1.01, 0.10))
    ax.set_ylabel("Outcome probability")
    ax.grid(axis="y", lw=0.6, alpha=0.4)
    ax.set_axisbelow(True)
    if bbox_legenda is None:
        ax.legend(loc=loc_legenda, fontsize=9, framealpha=0.9)
    else:
        ax.legend(loc=loc_legenda, bbox_to_anchor=bbox_legenda, fontsize=9,
                  framealpha=0.9)
    if titulo:
        ax.set_title(titulo)

    ax2.step(x, d["dif"].values, where="post", lw=1.4, color="0.25")
    ax2.axhline(0, color="0.4", lw=1, ls=":")
    ax2.set_ylabel("Score\ndifference")
    ax2.grid(axis="y", lw=0.6, alpha=0.4)
    ax2.set_axisbelow(True)
    lim = max(6, int(np.abs(d["dif"]).max()) + 2)
    ax2.set_ylim(-lim, lim)

    if um:
        ax2.invert_xaxis()
        ax2.set_xlabel("Time remaining in Q%d" % periodos[0])
        topo = int(np.ceil(x.max() / intervalo_ticks) * intervalo_ticks)
        t = np.arange(topo, -1, -intervalo_ticks)
        ax2.set_xticks(t)
        ax2.set_xticklabels([_mmss(v) for v in t])
    else:
        for p in periodos[1:]:
            ax.axvline((p - 1) * DURACAO_PERIODO, color="0.8", lw=1)
            ax2.axvline((p - 1) * DURACAO_PERIODO, color="0.8", lw=1)
        ax2.set_xlabel("Game time")
        ax2.set_xticks([(p - 0.5) * DURACAO_PERIODO for p in periodos])
        ax2.set_xticklabels(["Q%d" % p for p in periodos])
        ax2.set_xlim(0, DURACAO_PERIODO * len(periodos))

    fig.tight_layout()
    if caminho:
        fig.savefig(caminho, dpi=300, bbox_inches="tight", facecolor="white")
    return fig, (ax, ax2)
