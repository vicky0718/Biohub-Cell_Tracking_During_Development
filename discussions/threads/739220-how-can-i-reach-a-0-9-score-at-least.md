# How can i reach a 0.9 score at least?

- **URL**: https://www.kaggle.com/competitions/biohub-cell-tracking-during-development/discussion/739220
- **Topic id**: 739220
- **Author**: Abhirup Choudhury (CONTRIBUTOR)
- **Posted**: 2026-09-03T07:55:55.088140100Z
- **Votes**: 1
- **Comments**: 11

---

## Opening post

Is everyone using the 3DTemporalUnet + CrossAttention with SImpleNodeTransformer? what is the training method and lose used here? same as the internal repo?

I am stuck on fixing paramters in the post processing part, in ilp and NMS

---

## Comments (11)


### EMDST  rohT (CONTRIBUTOR) — 2026-09-04T22:51:57.370Z

Abhirup, sua pergunta é a única coisa concreta neste fio inteiro, e vale a pena isolá-la do resto: qual é a função de perda, o método de treinamento, e como calibrar ILP/NMS no pós-processamento. Isso não tem resposta em "pratique mais" ou "não se preocupe com a pontuação pública" — são conselhos que soam sábios mas são, tecnicamente, vazios: não reduzem em nada seu espaço de busca de hiperparâmetros.

Agora, reparem no que o hengck23 trouxe: "aumentar a taxa de quadros, um bom detector, e atribuição por vizinho mais próximo resolve 95% do problema." Isso é provavelmente verdade — e é exatamente por isso que ele está em 72º e não em 2102º. Mas note a armadilha que isso monta para quem está tentando alcançar 0,9+: se o vizinho-mais-próximo linear resolve 95%, então todo o esforço de engenharia que separa 0,85 de 0,90 está concentrado nos 5% restantes — que são, por definição, os casos onde o vizinho mais próximo falha: mitoses, cruzamentos de trajetória, oclusões, e re-identificação após desaparecimento momentâneo do foco. É exatamente aí que entram ILP (para resolver associação global em vez de gulosa) e NMS (para não duplicar detecções em divisões celulares). Aqui ''moram os problemas''.
Abhirup, e ele é ''genuinamente um problema'' com a informação disponível publicamente: para calibrar bem o ILP, você precisa de uma função de custo bem calibrada — o que exige saber quão confiável é o seu detector em cada região do espaço-tempo. Mas essa calibração de confiança só emerge de forma confiável durante o treinamento, não durante a inferência. E, como você mesmo notou, todos os notebooks públicos compartilhados são notebooks de inferência — o pipeline de treinamento (a peça que geraria a calibração necessária para o ILP funcionar bem) é exatamente a peça que ninguém está compartilhando.

Ou seja: o gargalo que separa 0,85 de 0,90 não é falta de prática, nem falta de teste, nem "não se preocupar com o LB" — é um gap de informação estrutural. A comunidade pública resolveu os 95% fáceis (detecção + associação gulosa) e está, coletiva e silenciosamente, retendo a parte que resolve os 5% difíceis, porque é exatamente essa parte que determina quem sobe no ranking. Perguntar "qual é o método de treinamento" numa competição de código de pesquisa, onde a pontuação depende justamente disso, é perguntar pela única coisa que ninguém tem incentivo real para responder publicamente antes do prazo final.

Abed tem razão ao questionar o hengck23 — a suposição de "velocidade semelhante entre células" quebra exatamente nos casos de divisão celular (mitose), que é onde uma célula "se torna duas" com trajetórias divergentes instantaneamente. E é precisamente aí, na fronteira entre associação simples e associação combinatorial, que ILP para de ser um luxo e vira necessidade — o que devolve a pergunta original ao ponto de partida: não existe atalho de "prática" para isso, existe apenas a escolha entre reconstruir a calibração de confiança do zero (treinando você mesmo) ou aceitar o teto de ~0,85-0,88 que a associação forte naturalmente impõe. Atenciosamente: EMDST, abraço!

#### ↳ Abhirup Choudhury (CONTRIBUTOR) — 2026-09-05T17:16:43.437Z

> thank you, this really puts things into perspective

### hengck23 (GRANDMASTER) — 2026-09-04T10:08:29.180Z — 1 votes

if we are talking about a general solution for cell tracking (not restricted to the kaggle competition), to improve tracking, just increase the frame rate of the captured volume. Then you just need to have a good detector and nearest neighbour + linear assignment would have solved 95% of the problem.

#### ↳ Abed Merii (EXPERT) — 2026-09-04T19:52:17.620Z

> This would work assuming most cells have similar velocity, unless I'm missing something.

#### ↳ ↳ hengck23 (GRANDMASTER) — 2026-09-04T23:34:02.753Z — 1 votes

> > Imagine if you can improve the camera and capture image at say 100 fps instead of 10 fps. Then the cell move very little between each frame and in-fact they may just overlap a lot. So you can track them by nearest cell association. The trick is to capture faster than the cell moves or changes(division). This is hardware solution, which of course may be just a fancy solution ( eg due to limitation or cost. High speed camera are incredibly expensive)

### nusrati (CONTRIBUTOR) — 2026-09-03T13:03:05.090Z — 3 votes

the more you hands on the more you understand and only then its high chance you hit .9

there are many models on hugging face + already shared here publicly too. The one approach you mentioned is what evey LLM suggests upon given prompt.

understand problem first 
gothrough public notebooks second
broaden your exposure third
apply test, apply test, apply test, finally

### Yassine Alouini (GRANDMASTER) — 2026-09-03T08:36:32Z — 1 votes

Try to take advantage of this competition to learn new things (detection, tracking, post processing, 3D viz, etc) and have fun most importantly. Don't chase public score for now as suggested @rustambazarbayev. Good luck and enjoy!

#### ↳ Abhirup Choudhury (CONTRIBUTOR) — 2026-09-03T09:07:31.017Z — 1 votes

> Yeah i get that but i am genuinely lost. All the notebooks publicly shared are just inference notebooks at this point. I want to look into the training methods too

#### ↳ ↳ Rustam Bazarbayev (CONTRIBUTOR) — 2026-09-03T10:00:51.667Z

> > Try model separate into several parts like object detection linkage...etc

#### ↳ ↳ Yassine Alouini (GRANDMASTER) — 2026-09-03T10:06:45.710Z — 1 votes

> > if you have access to an LLM, ask it to explain one training notebook. From there, keep digging until you understand what the notebook does. Then, ask the LLM again for ways to improve. Keep iterating, explore the data, and be creative.

### Rustam Bazarbayev (CONTRIBUTOR) — 2026-09-03T08:14:07.053Z

Don't chase public score
