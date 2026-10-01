# -*- coding: utf-8 -*-
"""
Diagnostico do limiar de 0,95 — cole como celula nova no notebook 21b.

Responde tres perguntas distintas que vinham sendo confundidas:

  (A) PERGUNTA DO PEDRO, versao por partida
      Em quantas partidas ALGUM instante teve alguma probabilidade >= 0,95
      apontando um desfecho que nao foi o realizado?

  (B) CALIBRACAO, versao por instante
      Entre os instantes em que o indexador diz >= 0,95, em que fracao o
      desfecho apontado e o que de fato ocorreu? Se calibrado, ~95%.

  (C) ERRO DA MEDIDA
      No proprio ponto de definicao (primeiro instante com condicao == 1),
      o desfecho apontado e o realizado? Este e o unico que mede a medida.
      Pela construcao de calcula_turning_point a condicao e sobre o MAXIMO
      das tres probabilidades, nao sobre um desfecho fixo, entao (C) pode
      ser diferente de zero: e o caso da cesta no estouro que vira o jogo.

Alem disso conta as SAIDAS do limiar: quantas vezes o maximo sobe acima de
0,95 e depois volta a cair, e com quanto tempo restante isso acontece.
"""
import numpy as np
import pandas as pd

DESFECHOS = ["home", "away", "draw"]


def _trajetoria(df_jogo, df_indexador, df_full_, passo=10):
    """Trajetoria completa de uma partida, sem filtrar pelo trecho resolvido.

    passo=10 reduz a grade de decimos para segundos. A grade cheia tem 28.800
    instantes por partida e o diagnostico nao precisa dessa resolucao.
    """
    t = (df_jogo
         .reset_index(drop=True)
         .pipe(preenche_dataframe, df_full=df_full_)
         .merge(df_indexador, how="left")
         .pipe(calcula_turning_point)
         .sort_values(["period", "remaining_time"], ascending=[True, False])
         .reset_index(drop=True))
    if passo > 1:
        t = t[t["remaining_time"] % passo == 0]
    return t


def diagnostico_limiar(df_real, df_indexador, df_games, df_full_=None,
                       limiar=0.95, passo=10, max_jogos=None):
    if df_full_ is None:
        df_full_ = df_full

    finais = (df_real
              .groupby("game_id")
              .agg(hpts=("home_points", "max"), apts=("away_points", "max")))
    finais["outcome"] = np.where(finais["hpts"] > finais["apts"], "home",
                         np.where(finais["hpts"] < finais["apts"], "away", "draw"))

    ids = df_real["game_id"].unique()
    if max_jogos:
        ids = ids[:max_jogos]

    linhas, conf = [], []
    for gid in ids:
        t = _trajetoria(df_real[df_real["game_id"] == gid].query("period <= 4"),
                        df_indexador, df_full_, passo=passo)
        t = t.dropna(subset=DESFECHOS)
        if t.empty:
            continue

        real = finais.loc[gid, "outcome"]
        probs = t[DESFECHOS].to_numpy()
        pmax = probs.max(axis=1)
        pred = np.array(DESFECHOS)[probs.argmax(axis=1)]
        acima = pmax >= limiar
        certo = pred == real

        # saidas: sobe acima do limiar e depois volta a cair
        saidas = int(np.sum(acima[:-1] & ~acima[1:])) if len(acima) > 1 else 0
        rt_saida = (t["remaining_time"].to_numpy()[:-1][acima[:-1] & ~acima[1:]]
                    if len(acima) > 1 else np.array([]))

        # ponto de definicao: primeiro instante com condicao == 1
        tp = t[t["condicao"] == 1]
        pred_tp = (DESFECHOS[int(tp[DESFECHOS].iloc[0].to_numpy().argmax())]
                   if len(tp) else None)

        linhas.append({
            "game_id": gid,
            "outcome": real,
            "cruzou": bool(acima.any()),
            "cruzou_errado": bool((acima & ~certo).any()),          # (A)
            "n_acima": int(acima.sum()),
            "n_acima_errado": int((acima & ~certo).sum()),           # (B)
            "saidas": saidas,
            "rt_1a_saida": float(rt_saida[0]) if len(rt_saida) else np.nan,
            "pred_tp": pred_tp,
            "tp_errado": (pred_tp is not None and pred_tp != real),   # (C)
        })

        # pares (probabilidade, acerto) para o diagrama de confiabilidade,
        # um por desfecho por instante
        for k, y in enumerate(DESFECHOS):
            conf.append(pd.DataFrame({"game_id": gid, "p": probs[:, k],
                                      "hit": (y == real)}))

    res = pd.DataFrame(linhas)
    conf = pd.concat(conf, ignore_index=True)

    print("=" * 68)
    print("partidas analisadas: %s" % format(len(res), ",d"))
    print("=" * 68)
    print("(A) PERGUNTA DO PEDRO, por partida")
    cruz = res["cruzou"].sum()
    print("    cruzaram o limiar em algum instante : %s (%.1f%%)"
          % (format(cruz, ",d"), 100 * cruz / len(res)))
    print("    cruzaram apontando desfecho ERRADO  : %s (%.2f%% das que cruzaram)"
          % (format(res["cruzou_errado"].sum(), ",d"),
             100 * res["cruzou_errado"].sum() / max(cruz, 1)))
    print()
    print("(B) CALIBRACAO, por instante acima do limiar")
    na, ne = res["n_acima"].sum(), res["n_acima_errado"].sum()
    print("    instantes acima do limiar : %s" % format(na, ",d"))
    print("    frequencia observada      : %.2f%%   (esperado ~%.0f%%)"
          % (100 * (na - ne) / max(na, 1), 100 * limiar))
    print()
    print("(C) ERRO DA MEDIDA, no ponto de definicao")
    print("    pontos de definicao apontando desfecho errado: %s (%.2f%%)"
          % (format(res["tp_errado"].sum(), ",d"),
             100 * res["tp_errado"].sum() / len(res)))
    print()
    print("SAIDAS DO LIMIAR")
    com = (res["saidas"] > 0).sum()
    print("    partidas com ao menos uma saida : %s (%.1f%%)"
          % (format(com, ",d"), 100 * com / len(res)))
    print("    saidas por partida, media       : %.2f" % res["saidas"].mean())
    if res["rt_1a_saida"].notna().any():
        q = res["rt_1a_saida"].dropna().quantile([.25, .5, .75]) / 10.0
        print("    tempo restante na 1a saida (s)  : p25 %.0f | mediana %.0f | p75 %.0f"
              % (q.iloc[0], q.iloc[1], q.iloc[2]))

    # diagrama de confiabilidade
    faixas = [0, .1, .2, .3, .4, .5, .6, .7, .8, .9, .95, 1.0001]
    conf["faixa"] = pd.cut(conf["p"], bins=faixas, right=False)
    diag = (conf.groupby("faixa", observed=True)
            .agg(n=("hit", "size"), prevista=("p", "mean"), observada=("hit", "mean")))
    print()
    print("DIAGRAMA DE CONFIABILIDADE")
    print((diag.assign(prevista=lambda d: (100 * d["prevista"]).round(1),
                       observada=lambda d: (100 * d["observada"]).round(1))
           ).to_string())

    return res, diag


# res, diag = diagnostico_limiar(df_real_games_2223, df_indexador_2122, df_games_nba)
