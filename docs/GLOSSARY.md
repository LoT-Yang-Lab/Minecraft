# GLOSSARY - Minecraft notation governance

> **AI 对齐文件**：每次开始本课题的实质性工作前，先读此文件。  
> 本文件定义科学对象、符号层级和使用规则。LaTeX 宏实现见
> [`LATEX_MACROS.md`](LATEX_MACROS.md)。  
> 当前 notation 分成三层：task/world、agent internal MDP/SMDP、
> analysis / identification。符号首先由其 epistemic owner 决定：真实任务、
> participant 的内部模型，还是 analyst 恢复出的结构。

---

## 0. 顶层规则

### 0.1 三层 ownership

同一种数学对象可以在多个层中出现，但必须用不同 notation 标出其 epistemic
status。

| Layer | 问题 | 核心符号 | 禁止混入 |
|-------|------|----------|----------|
| Task / world | 实验环境真实是什么？ | $\mathcal E,\mathcal S,\mathcal A,P_{\mathrm{env}},R$ | $M,\mathcal T_M,\Theta_M,\widehat{\cdot}$ |
| Agent internal model | participant 主观上用什么 MDP/SMDP 和 policy？ | $M,\mathcal T_M,\Omega_M,P_M,R_M,\mathcal K_M,\pi_M$ | $\widehat{\mathcal G}_M,\mathrm{BDeu},\widetilde p(M\mid D)$ |
| Analysis / identification | analyst 如何从数据估计或比较 agent models？ | $D,\mathcal G_M,\Theta_M,\widehat{\mathcal G}_M,\widehat{\mathcal T}_M,\widetilde p(M\mid D),d^*,L_n^{(M)}$ | 把 EM、BDeu 或 posterior 当成 participant cognition |

### 0.2 最小 spine

正文中的最小表达应优先使用：

$$
\boxed{
\mathcal E
\longrightarrow
\mathcal T_M
\longrightarrow
\mathcal K_M
\longrightarrow
\pi_M(s;s_\star)
\longrightarrow
D
\longrightarrow
\widehat{\mathcal G}_M,\widehat{\mathcal T}_M,\widetilde p(M\mid D)
}
$$

含义：

- $\mathcal E$ 是真实 task environment；
- $\mathcal T_M$ 是 family $M$ 假设 participant 构造的 internal MDP/SMDP；
- $\mathcal K_M$ 是该 internal model 诱导的 cognitive-map coordinate set；
- $\omega\in\Omega_M$ 是 agent 选择的 internal action token，可为 primitive action
  或 temporally extended token；
- $\pi_M$ 是由 shared model-based planning procedure 诱导的 policy；
- $D$ 是 observed behaviour；
- hats、posteriors、$d^*$、$L_n^{(M)}$、$\widehat{\mathcal G}_M$ 等都属于
  analysis layer。

### 0.3 Family names

The official candidate family names are

$$
\mathcal M
=
\{
M_{\mathrm{primitive}},
M_{\mathrm{state}},
M_{\mathrm{temporal}}
\}.
$$

Use the shorthand labels `primitive`, `state`, and `temporal` throughout the glossary.
Do not use `option`, `chunk`, `chunking`, or `landmark` as family labels. Temporally
extended action sequences and landmark-mediated transitions may still be discussed as
mechanisms inside the temporal and state abstraction agents.

### 0.4 不再作为顶层 spine 的旧链条

旧 notation

$$
\mathcal R_M
\to
P_M
\to
c_M
\to
d_M^\to
\to
\psi_M
\to
\mathrm A^*
$$

不再作为 glossary 的主 spine。Its useful pieces may appear locally as policy or
analysis diagnostics, but $d_M^\to$、$\psi_M$、A* should not be treated as independent
object layers parallel to task, agent, and analysis.

---

## 1. 时间粒度与索引

```text
participant p
  -> session / block b
       -> trial / trajectory n
            -> primitive step t
            -> local analysis span m, if needed
```

| 索引 | 唯一含义 | 例子 |
|------|----------|------|
| $p$ | participant | $D_p$ |
| $b$ | session / block | $D_{p,b}$ |
| $n$ | trial / trajectory | $\tau_n,D_n,L_n^{(M)}$ |
| $t$ | primitive step within trial | $s_{n,t},a_{n,t}$ |
| $m$ | local analysis span index | $\omega_{n,m},\sigma_{n,m}$ |
| $u,v$ | primitive boundary / span endpoints | $a_{n,u:v}$ |
| $i,j,k$ | landmark / abstract-node indices | $\ell_i,\ell_j$ |
| $q$ | optimization / EM iteration only | $\Theta_M^{(q)}$ |

正文在无歧义时可以省略 participant 和 session 下标，但 trial index $n$ 与
primitive-step index $t$ 不得互换。

---

## 2. Notation grammar

### 2.1 字体与字母保留

| 形式 | 含义 | 例子 |
|------|------|------|
| calligraphic uppercase | 集合、结构、环境、模型 | $\mathcal E,\mathcal S,\mathcal A,\mathcal T_M,\mathcal G_M$ |
| roman uppercase $M$ | agent family | $M_{\mathrm{primitive}}$ |
| Greek uppercase $\Theta$ | analysis-side continuous parameters | $\Theta_M$ |
| lowercase Roman / Greek | 单个状态、动作、token 或参数 | $s,a,\omega,\ell,\beta$ |
| uppercase $D$ | observed data bundle | $D,D_n,D_{1:n}$ |

保留规则：

1. $\mathcal E$ 表示真实 task environment；不要用 $\mathcal T$ 表示 task。
2. $\mathcal T_M$ 表示 agent internal MDP/SMDP；下标 $M$ 必须出现。
3. $M$ 只表示 agent family；candidate family set 写为 $\mathcal M$。
4. $R$ 表示 task-level state reward / cost；$R_M$ 表示 agent-internal reward / cost。
5. $\omega$ 表示 agent internal action token；$a$ 只表示 primitive action。
6. $\mathcal G_M$ 表示 family-specific dependency graph；do not use $\mathcal G$ as
   a standalone grammar for action-sequence tokens.
7. $\widehat{\cdot}$ 表示 analysis-recovered estimate；不能用于 task truth 或 agent-internal
   primitives before fitting。
8. $\widetilde{\cdot}$ 表示明确标注的 approximation。
9. 上标 $(q)$ 只表示 optimization / EM iteration。
10. 上标 $(M)$ 可用于 family-specific evidence quantity，例如 $L_n^{(M)}$。

### 2.2 Layer prefixes by epistemic status

| 对象                    | Task truth     | Agent internal   | Analysis estimate        |
| --------------------- | -------------- | ---------------- | ------------------------ |
| Environment / model   | $\mathcal E$   | $\mathcal T_M$   | $\widehat{\mathcal T}_M$ |
| Policy                | not task-owned | $\pi_M$          | $\widehat\pi_M$          |
| Dependency graph | not task-owned | ${\mathcal G}_M$ | $\widehat{\mathcal G}_M$ |

Avoid double hats in prose. If a fitted internal transition or reward must be named,
write "the fitted transition/reward inside $\widehat{\mathcal T}_M$."

---

## 3. Task / world layer

### 3.1 Task environment

The task is a finite MDP with state-only reward / cost:

$$
\mathcal E
=
(\mathcal S,\mathcal A,P_{\mathrm{env}},R,\rho_0,\mathcal S_\star,H).
$$

| 符号 | 含义 |
|------|------|
| $\mathcal E$ | true task environment |
| $\mathcal S$ | primitive state space |
| $\mathcal A$ | primitive action space |
| $s,s'\in\mathcal S$ | primitive states |
| $a\in\mathcal A$ | primitive action |
| $P_{\mathrm{env}}(s'\mid s,a)$ | task-defined transition |
| $R(s)$ | state-only reward / cost |
| $r$ | realized scalar reward at a primitive step |
| $R_t$ | random variable for the reward at primitive step $t$ |
| $\rho_0$ | start-state distribution, if randomized |
| $\mathcal S_\star$ | possible goal states |
| $H$ | horizon or maximum primitive steps |

When the trial goal determines which state is rewarding, use the local shorthand

$$
R_n(s)
$$

for trial $n$. Typically $R_n(s)$ gives reward at $s=s_{\star,n}$ and a step or state
cost otherwise. Do not use action-conditioned or next-state-conditioned reward notation
in the glossary.

For deterministic tasks one may write

$$
f_{\mathrm{env}}(s,a)=s'
$$

and for an action sequence $\mathbf a=(a_1,\ldots,a_k)$,

$$
f_{\mathrm{env}}(s,\mathbf a)
$$

denotes the final state reached by executing the sequence from $s$.

### 3.2 Trial trajectory

Observed primitive trajectory:

$$
\tau_n
=
(s_{n,0},a_{n,1},r_{n,1},s_{n,1},\ldots,a_{n,T_n},r_{n,T_n},s_{n,T_n}).
$$

| 符号                | 含义                                                 |
| ----------------- | -------------------------------------------------- |
| $s_{n,0}$         | trial start                                        |
| $s_{\star,n}$     | trial goal                                         |
| $T_n$             | observed primitive-action count                    |
| $s_{n,t},a_{n,t}$ | primitive state and primitive action               |
| $r_{n,t}$         | realized reward at primitive step $t$ in trial $n$ |
| $R_{n,t}$         | random variable whose realization is $r_{n,t}$     |

Do not introduce a separate trial-return symbol unless a local analysis explicitly needs
one. Path efficiency, likelihood, and structural evidence should not depend on a glossary-
level return variable.

---

## 4. Agent internal MDP/SMDP layer

### 4.1 Internal model

Each family defines an internal model:

$$
\mathcal T_M
=
(\mathcal S,\Omega_M,P_M,R_M).
$$

| 符号                             | 含义                                                                                                                                                 |
| ------------------------------ | -------------------------------------------------------------------------------------------------------------------------------------------------- |
| $\mathcal T_M$                 | family $M$ 的 agent internal MDP/SMDP                                                                                                               |
| $\mathcal S$                   | agent uses the task state space as its planning state space in the current paper                                                                    |
| $\Omega_M$                     | internal action-token set                                                                                                                          |
| $\omega\in\Omega_M$            | primitive or temporally extended internal token                                                                                                    |
| $P_M(s'\mid s,\omega)$         | agent-internal token transition model                                                                                                              |
| $R_M(s)$ or $R_M(s,\omega,s')$ | agent-internal reward / cost, depending on local derivation                                                                                        |
| $\psi_M(s)$                    | coordinate of state $s$ in the family-specific cognitive map                                                                                       |

Do not introduce a separate main-layer internal state space for the current paper.
State abstraction, when discussed here, means that the agent simplifies the transition
structure over $\mathcal S$; it does not replace $\mathcal S$ by a quotient state space.

For token-level planning, use the conditional kernel $P_M(s'\mid s,\omega)$. If a
token-marginal transition is needed for deriving a map or likelihood, write it locally as

$$
P_M(s'\mid s).
$$

The cognitive map is represented only by coordinates for all states:

$$
\psi_M:\mathcal S\to\mathbb R^d.
$$

If a named cognitive-map object is useful, use

$$
\mathcal K_M
=
\{\psi_M(s):s\in\mathcal S\}.
$$

Do not define $\mathcal K_M$ as a tuple of edges, distances, costs, and embeddings in the
glossary. Planning costs are induced by the internal reward / cost function $R_M$; they
are not independent representational objects.

The primitive baseline uses only primitive actions:

$$
\Omega_{\mathrm{primitive}}=\mathcal A.
$$

### 4.2 Temporal Abstraction Agent Internal Model

The temporal abstraction agent stores an action-level dependency graph over primitive
actions. Let

$$
\mathcal E_{\mathcal A}\subseteq\mathcal A\times\mathcal A
$$

be the supported action-to-action edges. Each supported edge $(a,a')\in\mathcal E_{\mathcal A}$
has a separate positive edge parameter

$$
\eta_{aa'}>0.
$$

The edge parameters are not part of $\mathcal G_{\mathrm{temporal}}$; they may induce the
row-normalized action transition

$$
\Pr_{\mathrm{temporal}}(a'\mid a)
=
\begin{cases}
\dfrac{\eta_{aa'}}
{\sum_{b:(a,b)\in\mathcal E_{\mathcal A}}\eta_{ab}},
& (a,a')\in\mathcal E_{\mathcal A},\\
0, & (a,a')\notin\mathcal E_{\mathcal A}.
\end{cases}
$$

The agent augments primitive actions with temporally extended internal action tokens.
An internal token $\omega$ may execute for multiple primitive steps. Its primitive
implementation is

$$
\mathbf a(\omega)
=
(a_1,\ldots,a_{k_\omega}),
\qquad
k_\omega=|\mathbf a(\omega)|.
$$

The induced internal transition is

$$
P_M(s'\mid s,\omega)
=
\Pr(s_{t+k_\omega}=s'\mid s_t=s,\omega,\mathcal T_M).
$$

For one-step primitive actions, $\omega=a$ and $k_\omega=1$.

If a token has stochastic or policy-dependent implementation, $\mathbf a(\omega)$ denotes
a realized primitive sequence and $P_M(\cdot\mid s,\omega)$ is the marginal transition distribution over realizations.

$$
\Omega_{\mathrm{temporal}}
=
\mathcal A
\cup
\{\omega:\mathbf a(\omega)=(a_1,\ldots,a_k),\ k>1,
(a_i,a_{i+1})\in\mathcal E_{\mathcal A}\ \text{for all }i<k\}.
$$

Here $\omega$ itself is the temporally extended token, derived as a supported path or
motif in the action dependency graph. In the main glossary, a token is not a separate
Bayesian graph or constituency-grammar object. Constituency grammar can be introduced
only as a later extension if the paper explicitly needs it.
Do not introduce $(\mathcal I_o,\pi_o,\beta_o)$ unless the paper explicitly returns to a
formal options derivation.

### 4.3 State Abstraction Agent Internal Transition Model

The state abstraction agent simplifies the transition model over $\mathcal S$ by treating
landmarks as special states. Define

$$
\mathcal L\subset\mathcal S,
\qquad
\mathcal U=\mathcal S\setminus\mathcal L.
$$

The state-abstraction dependency edge set is

$$
\mathcal E_{\mathrm{state}}
=
\mathcal E_{\mathcal L\mathcal L}
\cup
\mathcal E_{\mathcal L\mathcal U}
\cup
\mathcal E_{\mathcal U\mathcal L},
\qquad
\mathcal E_{\mathcal U\mathcal U}=\varnothing.
$$

Each supported edge $(s,s')\in\mathcal E_{\mathrm{state}}$ has a separate positive edge
parameter

$$
\eta_{ss'}>0.
$$

The edge parameters are not part of $\mathcal G_{\mathrm{state}}$; they induce the
row-normalized transition kernel

$$
P_{\mathrm{state}}(s'\mid s)
=
\begin{cases}
\dfrac{\eta_{ss'}}
{\sum_{y:(s,y)\in\mathcal E_{\mathrm{state}}}\eta_{sy}},
& (s,s')\in\mathcal E_{\mathrm{state}},\\
0, & (s,s')\notin\mathcal E_{\mathrm{state}}.
\end{cases}
$$

Thus landmark-to-landmark, landmark-to-non-landmark, and non-landmark-to-landmark
transitions may be supported, while direct non-landmark-to-non-landmark transitions are
absent. Every transition out of a non-landmark must therefore pass through the landmark
layer.

To connect the state-level landmark transition to primitive action sequences, define

$$
\mathcal A_{s\to s'}^{+}
=
\{\mathbf a=(a_1,\ldots,a_k): f_{\mathrm{env}}(s,\mathbf a)=s'\}.
$$

For supported landmark transitions, use a uniform implementation distribution:

$$
p_M(\mathbf a\mid s,s')
=
\frac{1}{|\mathcal A_{s\to s'}^{+}|}.
$$

Therefore

$$
p_M(s',\mathbf a\mid s)
=
P_{\mathrm{state}}(s'\mid s)\,
p_M(\mathbf a\mid s,s').
$$

This is an agent-internal construction. If the same block structure is fitted from data,
it belongs to $\widehat{\mathcal T}_M$ and $\widehat{\mathcal G}_{\mathrm{state}}$ in
the analysis layer.

### 4.4 Family-specific dependency graphs

$\mathcal G_M$ is a family-specific dependency graph. It records which temporal or
state-transition dependencies are allowed; it does not contain transition parameters.
Those parameters live on edges and induce $P_M$ inside $\mathcal T_M$. The same structure
symbol can appear in two epistemic positions:

- agent-internal: the structure implicitly stored by the participant as part of
  $\mathcal T_M$;
- analysis-side: the structure the analyst estimates as $\widehat{\mathcal G}_M$.

For the current families:

$$
\mathcal G_{\mathrm{primitive}}
=
\varnothing
$$

or the degenerate one-step transition structure over primitive actions.

For temporal abstraction agents,

$$
\mathcal G_{\mathrm{temporal}}
=
(\mathcal A,\mathcal E_{\mathcal A})
$$

where $\mathcal E_{\mathcal A}\subseteq\mathcal A\times\mathcal A$ encodes allowable
action-to-action dependencies. Edge parameters $\eta_{aa'}$ are separate from the graph
and may induce $\Pr_{\mathrm{temporal}}(a'\mid a)$; supported action paths in this graph
induce the temporally extended tokens in $\Omega_{\mathrm{temporal}}$.

For state abstraction agents,

$$
\mathcal G_{\mathrm{state}}
=
(\mathcal S,\mathcal L,
\mathcal E_{\mathcal L\mathcal L},
\mathcal E_{\mathcal U\mathcal L},
\mathcal E_{\mathcal L\mathcal U}),
$$

where $\mathcal U=\mathcal S\setminus\mathcal L$ and
$\mathcal E_{\mathcal U\mathcal U}=\varnothing$ by definition. The parameters
$\eta_{ss'}$ on these edges are not elements of $\mathcal G_{\mathrm{state}}$.

### 4.5 Policy layer

Agent behaviour is generated by a policy over internal tokens. In RL terms,
$\mathcal T_M$ is the agent's internal model and $\omega$ plays the role of the action in
that internal MDP/SMDP. The policy selects a token for the current state and target:

$$
\pi_M(s;s_\star)\in\Omega_M(s),
$$

where $\Omega_M(s)\subseteq\Omega_M$ is the set of tokens available at $s$.

In this glossary, A* is a model-based planning / heuristic-search procedure defined on
the internal model $\mathcal T_M$. With nonnegative transition costs and an admissible
heuristic, A* returns an optimal minimum-cost path. It is not a general RL learning
algorithm; it is a way to compute a target-directed policy from a known or hypothesized
model.

For a candidate token $\omega\in\Omega_M(s)$, let $s_\omega'$ be the endpoint predicted by
the internal transition model. Define the one-step token cost from the internal reward /
cost function:

$$
c_M(s,\omega,s_\omega')
=
R_M(s,\omega,s_\omega'),
$$

when $R_M$ is written as as cost. Lower $c_M$ is better
for planning. The target-directed heuristic is

$$
h_M(s;s_\star)
=
\lambda_h\|\psi_M(s)-\psi_M(s_\star)\|_2,
$$

where $\lambda_h$ is the heuristic scale. The A*-style cost-to-go action value is

$$
Q_M^{\mathrm{A^*}}(s,\omega;s_\star)
=
c_M(s,\omega,s_\omega')
+
h_M(s_\omega';s_\star).
$$

Lower $Q_M^{\mathrm{A^*}}$ is better. The deterministic planning policy is the greedy
minimum-cost token choice

$$
\pi_M(s;s_\star)
\in
\underset{\omega\in\Omega_M(s)}{\arg\min}
\ Q_M^{\mathrm{A^*}}(s,\omega;s_\star).
$$

Stochastic choice rules, response temperatures, lapse rates, and trajectory likelihoods are analysis-side observation models. They are not part of the agent's main policy definition in this section.

---

## 5. Analysis / identification layer

### 5.1 Observed data

| 符号 | 含义 |
|------|------|
| $D$ | all observed data |
| $D_n$ | trial $n$ data bundle |
| $D_{1:n}$ | cumulative data through trial $n$ |
| $D^{\mathrm{explore}}$ | free-exploration trajectories |
| $D^{\mathrm{search}}$ | start-goal search trajectories |
| $D^{\mathrm{near}}$ | directional nearness judgments |
| $D^{\mathrm{temporal}}$ | temporal-token judgments |
| $\mathrm{RT}_n$ | optional response-time data |

### 5.2 Structure-learning target

The primary analysis object is the family-specific dependency graph:

$$
\mathcal G_M.
$$

Recovered structures and induced fitted objects are

$$
\widehat{\mathcal G}_M,
\qquad
\widehat{\Omega}_M,
\qquad
\widehat{\mathcal T}_M.
$$

Family-specific interpretation:

| Family | Recovered structure |
|--------|---------------------|
| $M_{\mathrm{temporal}}$ | $\widehat{\mathcal G}_{\mathrm{temporal}}$ is the recovered action-to-action dependency graph; fitted edge parameters induce temporal-token probabilities and recovered internal tokens $\widehat\Omega_{\mathrm{temporal}}$ |
| $M_{\mathrm{state}}$ | $\widehat{\mathcal G}_{\mathrm{state}}$ is the recovered dependency graph over landmark-mediated state transitions; fitted edge parameters induce $\widehat P_{\mathrm{state}}$ |
| $M_{\mathrm{primitive}}$ | no nontrivial structure beyond primitive actions |

Local token assignments, spans, or latent variables may be introduced as computational
devices derived from $\widehat{\mathcal G}_M$. They are not the top-level
structure-learning target.

If local span notation is needed, use

| 符号 | 含义 |
|------|------|
| $\omega_{n,m}$ | token assigned to local span $m$ in trial $n$ |
| $\sigma_{n,m}$ | observed primitive content of span $m$ |
| $u_m,v_m$ | primitive span boundaries |
| $\delta_{n,m}=|\sigma_{n,m}|$ | derived span duration |

Do not introduce a generic top-level $Z$ section. If a method requires latent variables,
define them locally as a consequence of $\mathcal G_M$ or $\widehat{\mathcal G}_M$.

### 5.3 Hypotheses and fitted quantities

The analysis hypothesis can be written as

$$
h=(M,\mathcal G_M,\mathcal T_M).
$$

Posterior target:

$$
p(M,\mathcal G_M,\mathcal T_M,\Theta_M\mid D).
$$

Fitted quantities:

| 符号 | 含义 |
|------|------|
| $\widehat{\mathcal G}_M$ | fitted / recovered family-specific dependency graph |
| $\widehat{\Omega}_M$ | recovered internal token set induced by $\widehat{\mathcal G}_M$ |
| $\widehat{\mathcal T}_M$ | recovered internal model for family $M$ |
| $\widehat\Theta_M$ | fitted continuous analysis parameters |
| $\widehat\pi_M$ | fitted policy implied by $\widehat{\mathcal T}_M,\widehat\Theta_M$ |
| $\widetilde p(M\mid D)$ | approximate family posterior |

Use $\mathcal H$ for the candidate hypothesis space if needed:

$$
\mathcal H
=
\{(M,\mathcal G_M,\mathcal T_M):M\in\mathcal M\}.
$$

### 5.4 Normative and diagnostic quantities

The following are analysis-derived quantities, not task primitives:

| 符号 | 含义 |
|------|------|
| $d^*(s,s_\star)$ | true primitive shortest-path distance computed from $\mathcal E$ |
| $E_n$ | path efficiency |
| $L_n^{(M)}$ | trial-window structural evidence for family $M$ |
| $N_{\mathrm{expand}}$ | node expansions under a specified planning algorithm |
| $\mu_{\mathrm{sp}}(s,s_\star)$ | primitive shortest-path multiplicity |
| $d_{\mathrm{grid}}(s,s_\star)$ | grid-only shortest-path distance, if used |
| $d_{\mathrm{loop}}(s,s_\star)$ | shortest distance requiring the directed loop, if used |

Path efficiency:

$$
E_n
=
\begin{cases}
\dfrac{d^*(s_{n,0},s_{\star,n})}{T_n},
& \text{trial $n$ successful},\\[6pt]
0,
& \text{trial $n$ failed}.
\end{cases}
$$

Structural evidence:

$$
L_n^{(M)}
=
\log p(D_{n-w+1:n}\mid M)
-
\log p(D_{n-w+1:n}\mid M_{\mathrm{primitive}}),
$$

where $w$ is the trial-window size. $L_n^{(M)}$ is an analyst summary of evidence, not a
participant variable.

### 5.5 Structural learning and Bayesian program induction

Use Bayesian program induction / PGM structure-learning language for the analysis layer:

$$
p(M,\mathcal G_M,\mathcal T_M,\Theta_M\mid D)
\propto
p(M)
p(\mathcal G_M,\mathcal T_M\mid M)
p(\Theta_M\mid M)
p(D\mid M,\mathcal G_M,\mathcal T_M,\Theta_M).
$$

Exact family evidence is conceptually

$$
p(M\mid D)
\propto
p(M)
\sum_{\mathcal G_M,\mathcal T_M}
\int
p(D,\mathcal G_M,\mathcal T_M,\Theta_M\mid M)
\,d\Theta_M.
$$

Approximate family probability:

$$
\widetilde p(M\mid D)
=
\frac{
p(M)\exp(-\tfrac12\mathrm{IC}_M)
}{
\sum_{M'\in\mathcal M}
p(M')\exp(-\tfrac12\mathrm{IC}_{M'})
}.
$$

$\widetilde p(M\mid D)$ must be labeled as approximate. Do not write it as the exact
posterior.

### 5.6 BDeu, structural EM, and proposal scores

BDeu and structural EM are analysis algorithms. They do not describe participant
cognition.

| 符号 | 含义 |
|------|------|
| $\Theta_M$ | continuous likelihood / policy / execution parameters |
| $\theta_\pi$ | analyst-side policy-observation parameters, if a stochastic choice wrapper is used |
| $\theta_{\mathrm{exec}}$ | token execution / emission parameters |
| $\theta_{\mathrm{RT}}$ | optional response-time parameters |
| $\alpha_{\mathrm{BD}}$ | BDeu equivalent sample size |
| $\mathcal B_M^{(q)}$ | within-family BDeu proposal diagnostic |
| $\mathcal C_M$ | observed-data acceptance objective |

Generic structure-learning update:

$$
(\widehat{\mathcal G}_M,\widehat{\mathcal T}_M,\widehat\Theta_M)
=
\underset{\mathcal G_M,\mathcal T_M,\Theta_M}{\arg\max}
\left[
\log p(D\mid M,\mathcal G_M,\mathcal T_M,\Theta_M)
+
\log p(\mathcal G_M,\mathcal T_M\mid M)
-
\mathcal P_M(\Theta_M)
\right].
$$

Within-family proposal diagnostic:

$$
\mathcal B_M^{(q)}(\mathcal G'_M,\mathcal T'_M)
=
\log p_{\mathrm{BDeu}}
\left(
\bar N_M^{(q)}(\mathcal G'_M,\mathcal T'_M)
\mid \mathcal G'_M,\alpha_{\mathrm{BD}}
\right)
+
\log p(\mathcal G'_M,\mathcal T'_M\mid M).
$$

Final proposal acceptance should use an observed-data objective:

$$
\mathcal C_M(\mathcal G'_M,\mathcal T'_M)
=
\log
p(D\mid M,\mathcal G'_M,\mathcal T'_M,\widehat\Theta_M(\mathcal G'_M,\mathcal T'_M))
+
\log p(\mathcal G'_M,\mathcal T'_M\mid M).
$$

Raw BDeu scores are not used for cross-family comparison.

---

## 6. Policy parameterization details

This section is optional notation for manuscripts or analyses that need to expose the
analyst-side stochastic choice model. Keep it local unless the paper text requires it.

### 6.1 Analyst-side stochastic choice and likelihood

The main agent policy is the deterministic planning policy in Section 4.5. If an analysis
needs probabilistic choice data, wrap the A*-style cost-to-go value in an observation
model. This is analyst-side likelihood notation, not the definition of the agent family.

$$
\widehat{\pi}_M^{\mathrm{obs}}(\omega\mid s,s_\star;\Theta_M)
=
(1-\epsilon)
\frac{
\exp\{-\beta_\pi Q_M^{\mathrm{A^*}}(s,\omega;s_\star)\}
}{
\sum_{\omega'\in\Omega_M(s)}
\exp\{-\beta_\pi Q_M^{\mathrm{A^*}}(s,\omega';s_\star)\}
}
+
\epsilon\frac{1}{|\Omega_M(s)|}.
$$

Here lower $Q_M^{\mathrm{A^*}}$ means higher choice probability. Optional fitted
parameters live in $\Theta_M$:

| 符号 | 含义 |
|------|------|
| $\beta_\pi$ | inverse temperature for the analyst-side choice model |
| $\epsilon$ | lapse / random-action rate |
| $B$ | optional search-expansion budget if bounded planning is fitted |
| $\lambda_h$ | heuristic scale if estimated rather than fixed |
| $\rho_{\mathrm{tie}}$ | tie-breaking rule or parameter |

For observed search data, a compact likelihood can be written as

$$
p(D^{\mathrm{search}}\mid M,\mathcal G_M,\mathcal T_M,\Theta_M)
=
\prod_{n,t}
\widehat{\pi}_M^{\mathrm{obs}}
(\omega_{n,t}\mid s_{n,t},s_{\star,n};\Theta_M)
\,
p_{\mathrm{exec}}
(a_{n,t}\mid \omega_{n,t},s_{n,t};\theta_{\mathrm{exec}}),
$$

with marginalization over $\omega_{n,t}$ if internal token assignments are latent. The
execution model $p_{\mathrm{exec}}$ maps internal-token choices to observed primitive
actions and belongs to the fitted analysis layer.

### 6.2 Cognitive-map coordinates

If coordinates are used as an A* heuristic, write

$$
\psi_M:\mathcal S\to\mathbb R^d.
$$

The heuristic term for target $s_\star$ is

$$
h_M(s;s_\star)
=
\lambda_h\|\psi_M(s)-\psi_M(s_\star)\|_2.
$$

Use this coordinate heuristic inside $Q_M^{\mathrm{A^*}}$ as defined in Section 4.5. Do
not define a separate glossary-level distance object unless a later analysis explicitly
needs one.

---

## 7. Family evidence patterns

| Pattern | Main interpretation |
|---------|---------------------|
| repeated action string across starts or goals | supports temporally extended internal tokens in $\Omega_{\mathrm{temporal}}$ |
| behaviour routes through stable landmark states | supports state abstraction through landmark-mediated $P_M$ or $R_M$ |
| fewer decision points with similar primitive path length | supports temporally extended internal tokens |
| route choice / RT follows reduced search expansions | supports a planner-policy account |
| improved later-domain starting evidence | supports cross-task transfer of $\mathcal T_M$ ingredients |

No empirical pattern should be described as final human evidence before behavioural
analysis and simulation recovery have passed the gates in `STATUS.md`.

---

## 8. Current naming decisions

| Old / tempting notation | Current rule |
|-------------------------|--------------|
| $\mathcal T$ for task | Avoid; use $\mathcal E$ so $\mathcal T_M$ can mean internal model |
| $\mathcal R_M$ as generic stored representation | Retire as top-level symbol; use $\mathcal T_M$ plus family-specific ingredients |
| hatted agent transitions/rewards | Avoid; use $P_M,R_M$ for agent-internal quantities |
| action-conditioned reward notation | Avoid; use state-only $R(s)$ and trial-local $R_n(s)$ |
| option, chunk, or landmark as family labels | Avoid in glossary; use $M_{\mathrm{temporal}}$ or $M_{\mathrm{state}}$ |
| $d^*$ in task layer | Move to analysis diagnostics |
| option tuple $(\mathcal I_o,\pi_o,\beta_o)$ | Do not use unless formal options are explicitly reintroduced |
| $\mathcal G_M$ | Use for family-specific dependency graph; parameters live on edges |
| $Z$ as top-level analysis object | Avoid; local latent variables may be derived from $\mathcal G_M$ |
| A* as representation layer | Treat as a policy parameterization |

---

## 9. Minimal symbol table

| Symbol | Layer | Meaning |
|--------|-------|---------|
| $\mathcal E$ | Task | true task environment |
| $\mathcal S,\mathcal A$ | Task | primitive states and actions |
| $P_{\mathrm{env}}$ | Task | true transition function |
| $R,R_n$ | Task | state reward / cost function |
| $\tau_n$ | Task / data | observed primitive trajectory |
| $\mathcal M$ | Agent / analysis | candidate family set |
| $M$ | Agent / analysis | model / agent family |
| $\mathcal T_M$ | Agent | internal MDP/SMDP for family $M$ |
| $\Omega_M$ | Agent | internal action-token set |
| $\omega$ | Agent | primitive or temporally extended token |
| $P_M$ | Agent | internal transition model |
| $R_M$ | Agent | internal reward / cost |
| $\psi_M$ | Agent | cognitive-map coordinate function on $\mathcal S$ |
| $\mathcal K_M$ | Agent | optional set of cognitive-map coordinates $\{\psi_M(s):s\in\mathcal S\}$ |
| $\pi_M$ | Agent | policy over internal tokens |
| $\mathcal L$ | Agent | landmark set |
| $\eta_{aa'}$ | Agent / analysis | edge parameter for supported temporal action edge $(a,a')$ |
| $\eta_{ss'}$ | Agent / analysis | edge parameter for supported state dependency edge $(s,s')$ |
| $D$ | Analysis | observed data bundle |
| $\mathcal G_M$ | Analysis | family-specific dependency graph |
| $\widehat{\mathcal G}_M$ | Analysis | recovered family-specific dependency graph |
| $\widehat{\Omega}_M$ | Analysis | recovered internal token set |
| $\widehat{\mathcal T}_M$ | Analysis | recovered internal model |
| $\Theta_M$ | Analysis | fitted continuous parameters |
| $d^*$ | Analysis | primitive shortest-path benchmark |
| $E_n$ | Analysis | path efficiency |
| $L_n^{(M)}$ | Analysis | structural evidence |
| $\widetilde p(M\mid D)$ | Analysis | approximate family posterior |
