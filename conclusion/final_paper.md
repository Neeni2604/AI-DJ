Final Project Report
Title:

Abstract:
Getting a machine to DJ well is not a simple task. Picking the next song requires juggling harmonic compatibility, energy pacing, and transition timing all at once, and any hand-written rules for doing this tend to break down quickly across different genres and contexts. We propose an RL-based AI DJ that learns to sequence tracks using PPO, with rewards learned from human preference comparisons rather than a fixed reward function. We plan to evaluate our agent by comparing sets generated under a proxy reward against sets generated under a Deep RLHF preference model, using both quantitative metrics and human preference win rates.
Introduction:
DJing is a sequential decision-making task. A good DJ set is not defined only by whether each individual song is good, but by how tracks fit together over time. Tempo changes, energy progression, transition style, repetition, and genre context all affect whether a sequence feels coherent. These criteria are difficult to encode as fixed rules because listener judgments are subjective and depend on the surrounding sequence.
We model this task as simplified DJ sequencing over a track library. At each step, the agent receives a normalized feature vector for the current track and chooses the next track plus a transition type: cut, fade, or beatmatch. The environment is built on Free Music Archive metadata and audio features stored in SQLite. This abstraction does not attempt to synthesize full audio mixes; instead, it focuses on the high-level planning problem of deciding what should come next and how the transition should be treated.
The main challenge is reward design. A purely hand-written reward can encourage useful musical properties such as smooth BPM changes and reasonable energy movement, but it can also overfit to measurable proxies that do not fully capture human taste. To address this, we use a two-stage training pipeline. First, PPO is trained with a proxy reward so that the agent can generate reasonable candidate sequences. Second, human annotators compare generated sequence pairs, and these labels train a reward model that is used for RLHF fine-tuning.
The contribution of this project is an end-to-end prototype for preference-based AI DJ sequencing. The system includes an FMA-backed Gymnasium environment, PPO training, preference-pair generation, a terminal annotation workflow, majority-vote label merging, Bradley-Terry reward-model training, RLHF fine-tuning, and evaluation artifacts. This makes the project a small but complete testbed for studying how reinforcement learning and human preference feedback can be applied to subjective music sequencing tasks.

Related Work:
Prior work on automatic DJ systems has focused mainly on song mixing and transition generation. MusicMixer uses audio similarity between song sections to suggest and perform seamless mixes for inexperienced users \cite{hdm-MusicMixercomputeraidedDJsystem-2015}. Vande Veire and De Bie present a more comprehensive raw-audio DJ system for drum and bass, showing that rule-based and MIR-driven methods can produce full mixes from a local song library \cite{vd-rawaudioseamlessmix-2018}. More recent AI DJ systems combine audience reaction analysis, similarity-based song selection, beatmatching, and equalizer mixing for electronic dance music \cite{hfnlc-AIDJSystemElectronic-2022}. These systems are closest to the practical goal of our project, but they primarily rely on engineered audio-processing pipelines rather than learning a sequencing policy from preference feedback.

Recent work has also explored playlist sequencing and transition quality beyond traditional recommendation. Liebman et al. argue that music preference depends on temporal context, not just isolated song choice, and show that online adaptation to sequential preferences can improve listener experience \cite{lss-RightMusicRightTime-2019}. Wang proposes a reinforcement-learning hybrid recommender that models both song preferences and transition preferences \cite{w-HybridRecommendationMusicBased-2020}. Tomasi et al. develop a simulation-based RL framework for playlist generation, using a modified DQN to optimize user-satisfaction metrics in large action spaces \cite{tckcrd-AutomaticMusicPlaylistGeneration-2023}. Our work shares this sequential framing, but focuses specifically on DJ-style track selection with explicit transition actions.

Several projects use learning methods directly for DJ automation. Kronvall's DeepFADE system applies hierarchical deep reinforcement learning to music mixing, with separate tiers for song selection, cue-point loading, and transition generation \cite{k-MixingMusicUsingDeep-2019}. Its reward functions include beat alignment, volume stability, tonal consonance, and transition timing, but the system suffers from reward hacking and limited policy expressiveness. DJ-AI systems use graph optimization, audio embeddings, and generative models such as MusicGen to arrange playlists and synthesize smoother transitions \cite{ktgyabb-DJAIIntelligentPlaylistSorting-2025,ktgyab-DJAIOptimizingPlaylist-2025}. Compared with these approaches, our project uses a simpler feature-level environment but directly studies how human preferences can improve the reward signal for sequencing.
Human feedback is especially relevant because musical quality is subjective. Justus applies human-in-the-loop reinforcement learning to music generation, using user feedback as the reward signal for iteratively improving generated compositions \cite{j-MusicGenerationusingHumanInTheLoop-2023}. We apply the same motivation to DJ sequencing rather than composition: instead of asking the agent to create new music, we ask it to choose better sequences of existing tracks. Our contribution is therefore a compact RLHF pipeline for DJ-style sequencing, combining proxy-reward PPO with a Bradley-Terry reward model trained from human pairwise comparisons.

Problem Statement:
A DJ set is a sequence of tracks arranged in a way such that each transition feels natural to listeners. Good sets tend to have smooth tempo changes, coherent energy arcs, and transitions that fit the musical context. Achieving this manually requires lots of experience and intuition that is difficult to encode explicitly. The goal of this work is to automate track sequencing in a way that produces sets a human listener would judge as high quality. 

We formalize this as finding an optimal policy $\pi^*$ over a sequence of track selections:
\begin{equation}
    \pi^* = \arg\max_{\pi}\ \mathbb{E}_{\sigma \sim \pi}\left[R(\sigma)\right]
\end{equation}
where $\sigma$ is a full DJ set produced by following policy $\pi$, and $R(\sigma)$ is a measure of overall set quality. The core difficulty is that $R$ is not known in closed form because listener preferences are subjective and context-dependent, so no hand-designed reward function can fully capture what makes a set good. Therefore, our work focuses on the central challenge of estimating $R$ from human judgement and using it to train a sequencing agent that reflects genuine listener preferences. 
Method:
The main idea behind our work is simple. People can usually recognize when a DJ set flows well, even if they struggle to explain why. We use this observation to split the problem into two parts. First, we train a policy to generate reasonable track sequences using simple reward signals based on interpretable metrics. Then, we replace those metrics with a reward signal learned directly from human judgements. This avoids the task of manually specifying every quality that makes a set feel natural. 

\subsection{Environment}
We model track-sequencing as a Markov Decision Process using a custom Gymnasium environment. At each step, the agent observes the current track as a normalized feature vector $s_t \in \mathbb{R}^9$ which includes tempo, key, mode, energy, valence, danceability, loudness, chroma mean, and RMS mean. Based on that state, it picks an action $a_t = (\text{track}_{t+1},\ \tau_t)$ where $\text{track}_{t+1}$ is the next track chosen from a candidate pool and $\tau_t \in \{\text{cut, fade, beatmatch}\}$ determines how the transition is performed. Episodes run for a fixed number of steps over tracks drawn from the Free Music Archive dataset.


\subsection{Stage 1: Proxy Reward Training}
Before collecting human preferences, we need the agent to generate sequences that are worth comparing. To do this, we train a PPO policy using a hand-designed proxy reward designed around three basic qualities of a good transition. 

\begin{equation}
    r_t = 0.5 \cdot r_{\text{bpm}} + 0.3 \cdot r_{\text{transition}} + 
    0.2 \cdot r_{\text{energy}} - p_{\text{repeat}}
\end{equation}

The term $r_{\text{bpm}}$ penalizes abrupt tempo jumps between consecutive tracks. The transition reward, $r_{\text{transition}}$, captures whether the selected transition type makes sense for the relationship between tracks (for example, rewarding beatmatching when BPM differences are small and favoring cuts when energy changes are more dramatic). The energy term encourages sequences that evolve in a musically coherent way rather than moving randomly between intensity levels. We also include a repeat penalty, $p_{\text{repeat}}$, to discourage replaying the same track. The policy itself is a two-layer MLP trained with PPO in Stable Baselines3.


\subsection{Stage 2: Preference Data Collection}

The proxy reward provides a useful starting point, but it misses qualities a human listener cares about. To capture those, we collect pairwise preference labels from human annotators. We generate pairs by rolling out two different policies in the environment. One sequence comes from the trained PPO policy and the other comes from a random policy. This contrast gives annotators a meaningful choice rather than comparing two sequences of similar quality. Any pair containing repeated tracks or very large BPM jumps is filtered out, since those sequences are broken and would only introduce noise to the labels. The A and B sides are randomly assigned to reduce position bias. Annotators choose the sequence with better overall flow, or skip if the choice is genuinely unclear. When multiple annotators label the same pair, we take the majority vote and discard ties. 


\subsection{Stage 3: Reward Model Training}

We train a small MLP on these winner-loser pairs. The loss is the Bradley-Terry ranking objective.

\begin{equation}
    \mathcal{L} = -\log \sigma\!\left(R(\sigma_w) - R(\sigma_l)\right)
\end{equation}

Here $\sigma_w$ and $\sigma_l$ are the preferred and rejected sequences, and $R(\cdot)$ is the scalar output of the reward model. Rather than feeding in raw step-by-step features, each sequence is summarized using five aggregate statistics. These are mean tempo, tempo smoothness, mean energy, energy trend, and beatmatch fraction. This aggregation keeps the input small, which matters when the annotation dataset is limited. Training is done with Adam, and we keep the checkpoint with the lowest validation loss. 


\subsection{Stage 4: RLHF Fine-Tuning}

Once the reward model is trained, we go back to the Stage 1 PPO checkpoint and continue training against the new signal. To keep the policy from forgetting what good transitions look like, we combine the proxy reward with the learned reward instead of replacing it entirely. At each step the agent still receives the proxy signal, and at the end of the episode the full sequence is scored by the reward model. This lets PPO optimize for long-run sequence quality as judged by human preferences, while staying grounded in the transition-level 
structure the proxy reward enforces.

Experimental Design:

We study **automatic DJ sequencing** as a sequential decision-making problem.
Given the current track, an agent chooses the next track and a transition type
(`cut`, `fade`, or `beatmatch`). The environment is built from the Free Music
Archive data stored in `fma_db/data/fma.db`, with track-level features such as
tempo, key, energy, valence, danceability, loudness, chroma, and RMS.

The project tests three hypotheses:

- **H1:** PPO trained on the proxy reward in `DJEnv` will outperform a random
 policy.
- **H2:** Larger PPO runs will improve policy quality, although the learned
 policy may still trail a greedy heuristic baseline.
- **H3:** RLHF fine-tuning with a learned reward model can improve a
 proxy-trained PPO checkpoint, but only if the reward model generalizes well.

The proxy reward used for PPO is:

`0.5 * bpm_smoothness + 0.3 * transition_reward + 0.2 * energy_flow - repeat_penalty`

This reward encourages smooth tempo changes, context-appropriate transitions,
coherent energy flow, and avoidance of repeated tracks. For RLHF, the proxy
reward is replaced with a terminal reward from a learned preference model.

The main experiments use a **500-track subset**, **12-step episodes**, and a
human-preference dataset built from **50 generated sequence pairs**. Three
annotators labeled these pairs; after majority voting, `43` usable preference
examples remained, `7` ties were dropped, and pairwise agreement was `0.5982`.

We ran three experiments:

1. **PPO vs. random:** test whether PPO beats the random baseline and shows an
  upward learning curve.
2. **PPO scaling and heuristic comparison:** compare saved PPO runs at
  increasing timestep budgets and measure the gap to the heuristic baseline.
3. **RLHF fine-tuning:** start from a PPO checkpoint, train a reward model on
  merged preference labels, then fine-tune PPO with terminal reward from that
  model.

The evaluation criteria are:

- **For H1:** PPO must exceed the random baseline and show upward reward trends.
- **For H2:** larger PPO runs should improve reward and ideally narrow the gap
 to the heuristic policy.
- **For H3:** RLHF should improve the starting PPO checkpoint on proxy reward,
 learned reward, or direct human preference. The saved repository mainly
 supports the first two.


Results and Discussion
Core Results
Result
Value
Preference pairs generated
50
Usable merged preference pairs
43
Pairwise annotator agreement
0.5982
Reward-model best validation epoch
1
Reward-model best validation loss
0.7070


Run
Timesteps
Random
Heuristic
PPO / RLHF score
Improvement
ppo_smoke
2,048
8.1183
n/a
8.4567
+0.3384 vs random
ppo_20260406_220618
16,384
8.4582
n/a
9.3478
+0.8896 vs random
ppo_20260410_002422
65,536
7.8289
11.2695
10.2409
+2.4120 vs random
ppo_20260413_004855
65,536
7.4781
11.3320
8.0709
+0.5928 vs random
rlhf_20260413_000436
32,768
start 9.9051
n/a
9.1785
-0.7266 vs start
rlhf_20260413_005446
32,768
start 8.1051
n/a
8.2579
+0.1528 vs start

Analysis
The strongest result is that PPO consistently beats random, which supports H1. All saved PPO runs improve on the random baseline and all show upward learning curves. The best PPO checkpoint, ppo_20260410_002422, reaches 10.2409, a gain of +2.4120 over random. This shows that the proxy-reward environment is learnable and that the PPO pipeline works end to end.
H2 is partially supported. Increasing training scale improves the best observed PPO result, but performance is not stable across runs. Two runs with the same 65,536 timestep budget differ sharply: one reaches 10.2409, while the later run reaches only 8.0709. This suggests sensitivity to seed, initialization, or training variance. In addition, PPO still trails the heuristic baseline, which stays near 11.3. The learned policy is useful, but it has not surpassed the strongest hand-designed strategy under the same proxy objective.
H3 is also only partially supported. One RLHF run improves its starting PPO checkpoint slightly (8.1051 -> 8.2579), but another degrades substantially (9.9051 -> 9.1785). The RLHF effect is therefore not robust. The most likely reason is the reward model: its best validation loss occurs at epoch 1, and validation loss worsens afterward despite lower training loss. That pattern indicates overfitting or weak supervision.
The reward-model bottleneck is consistent with the dataset size. The reward model was trained from only 43 usable preference pairs, and human agreement was moderate rather than high. That is enough to prove the pipeline works, but not enough to provide a strong, stable learning signal for RLHF. The current results therefore suggest that proxy-reward PPO is the strongest component of the system, while RLHF is limited primarily by reward-model quality.
Follow-On Experiments
Three follow-on experiments deepen this interpretation.
PPO scaling: comparing 2,048, 16,384, and 65,536 timesteps shows that more training can help substantially, but also exposes variance across large runs.
Different RLHF starting checkpoints: the two RLHF runs suggest that a weak reward model can either slightly help or actively hurt an existing PPO policy depending on the starting point.
Reward-model diagnostics: the validation-loss curve itself is a critical follow-on experiment, because it localizes the main problem to preference modeling rather than PPO implementation.
Overall, the results confirm that the system works technically, but they do not yet show that RLHF reliably outperforms the best PPO-only policy.

Conclusion:

This project explored AI DJ sequencing with PPO and RLHF. The main problem is interesting because good DJ behavior depends on both objective transition quality and subjective human judgments of flow. That makes it a natural domain for combining reinforcement learning with preference learning.
The main finding is that PPO successfully learns meaningful sequencing behavior from a structured proxy reward, consistently outperforming a random baseline. A second important finding is that RLHF is not yet reliable in the current system. It can produce small gains, but it can also degrade a strong starting checkpoint. The clearest explanation is that the reward model was trained on too little and too noisy preference data.
We learned three main lessons:
proxy rewards are strong enough to make the task learnable;
RLHF quality depends more on reward-model supervision than on PPO itself;
small preference datasets are not enough to consistently beat the best proxy-trained policy.
With another month or two, the most important next steps would be:
collect several hundred preference pairs instead of a few dozen;
improve annotation consistency with better instructions and calibration;
train a stronger reward model, ideally one that models sequence structure more explicitly;
run larger held-out human evaluations comparing PPO directly to RLHF;
stabilize PPO with repeated seeds and more systematic checkpoint selection;
explore hybrid training that mixes proxy reward and learned reward instead of replacing the proxy reward entirely.
In short, the project already demonstrates a complete AI DJ pipeline, but its most important scientific result is that the hard part is not getting PPO to learn. The hard part is learning a reward model from human preferences that is strong enough to guide further improvement.


References:

```
@inproceedings{hdm-MusicMixercomputeraidedDJsystem-2015,
  title = {{{MusicMixer}}: Computer-Aided {{DJ}} System Based on an Automatic Song Mixing},
  shorttitle = {{{MusicMixer}}},
  booktitle = {Proceedings of the 12th {{International Conference}} on {{Advances}} in {{Computer Entertainment Technology}}},
  author = {Hirai, Tatsunori and Doi, Hironori and Morishima, Shigeo},
  date = {2015-11-16},
  series = {{{ACE}} '15},
  pages = {1--5},
  publisher = {Association for Computing Machinery},
  location = {New York, NY, USA},
  doi = {10.1145/2832932.2832942},
  url = {https://dl.acm.org/doi/10.1145/2832932.2832942},
  urldate = {2026-04-26},
  abstract = {In this paper, we present MusicMixer, a computer-aided DJ system that helps DJs, specifically with song mixing. MusicMixer continuously mixes and plays songs using an automatic music mixing method that employs audio similarity calculations. By calculating similarities between song sections that can be naturally mixed, MusicMixer enables seamless song transitions. Though song mixing is the most fundamental and important factor in DJ performance, it is difficult for untrained people to seamlessly connect songs. MusicMixer realizes automatic song mixing using an audio signal processing approach; therefore, users can perform DJ mixing simply by selecting a song from a list of songs suggested by the system, enabling effective DJ song mixing and lowering entry barriers for the inexperienced. We also propose personalization for song suggestions using a preference memorization function of MusicMixer.},
  isbn = {978-1-4503-3852-3}
}

@inproceedings{hfnlc-AIDJSystemElectronic-2022,
  title = {{{AI DJ System}} for {{Electronic Dance Music}}},
  booktitle = {2022 {{International Symposium}} on {{Electronics}} and {{Smart Devices}} ({{ISESD}})},
  author = {Huang, Hao-Wei and Fadli, Muhammad and Nugraha, Achmad Kripton and Lin, Chih-Wei and Cheng, Ray-Guang},
  date = {2022-11},
  pages = {1--6},
  doi = {10.1109/ISESD56103.2022.9980591},
  url = {https://ieeexplore.ieee.org/abstract/document/9980591},
  urldate = {2026-04-27},
  abstract = {Disk jockey (DJ), the primary performer of electronic dance music, is often known to perform by observing the audience’s reaction and liven them by playing suitable EDM songs. DJ mixes two EDM songs without any breaks or silences, making the mix structurally coherent and seamless. However, DJs sometimes make mistakes during performances. An unsuitable song selection may occur due to the DJ’s inability to assess the audience’s reaction during a busy schedule. In the long run, hiring a DJ is quite expensive. A DJ is also limited in their ability to work continuously for long periods as a human being. Also, being a DJ requires expertise and experience, which most general music consumers lack. This paper proposes an automated DJ system featuring artificial intelligence (AI) named AI DJ system by combining action recognition, song selection, and beatmatching and equalizer mixing. The proposed system watches the audience’s reaction and selects the next song by using song similarity and choosing the similar key and suitable energy level. Additionally, beatmatching and equalizer mixing is applied to improve the song transition. Based on evaluations from 15 professional DJs in Taiwan using 400 EDM songs, the song mixes from the proposed system were confirmed to have good quality, with mean opinion score (MOS) as the metric.},
  eventtitle = {2022 {{International Symposium}} on {{Electronics}} and {{Smart Devices}} ({{ISESD}})},
  keywords = {AI DJ,Artificial Intelligence,Disc Jockey,Diversity reception,Electronic Dance Music,Equalizers,Handwriting recognition,Humanities,Measurement,Schedules,System performance}
}

@inproceedings{j-MusicGenerationusingHumanInTheLoop-2023,
  title = {Music {{Generation}} Using {{Human-In-The-Loop Reinforcement Learning}}},
  booktitle = {2023 {{IEEE International Conference}} on {{Big Data}} ({{BigData}})},
  author = {Justus, Aju Ani},
  date = {2023-12},
  pages = {4479--4484},
  doi = {10.1109/BigData59044.2023.10386567},
  url = {https://ieeexplore.ieee.org/abstract/document/10386567},
  urldate = {2026-04-27},
  abstract = {This paper presents an approach that combines Human-In-The-Loop Reinforcement Learning (HITL RL) with principles derived from music theory to facilitate real-time generation of musical compositions. HITL RL, previously employed in diverse applications such as modelling humanoid robot mechanics and enhancing language models, harnesses human feedback to refine the training process. In this study, we develop a HILT RL framework that can leverage the constraints and principles in music theory. In particular, we propose an episodic tabular Q-learning algorithm with an epsilon-greedy exploration policy. The system generates musical tracks (compositions), continuously enhancing its quality through iterative human-in-the-loop feedback. The reward function for this process is the subjective musical taste of the user.},
  eventtitle = {2023 {{IEEE International Conference}} on {{Big Data}} ({{BigData}})},
  keywords = {Algorithmic Music,Audio Machine Learning,Heuristic algorithms,HITL RL,Human Feedback,Human in the loop,Human-Agent Teaming,Human-In-The-Loop,Machine learning algorithms,Music,Music Generation,Q-learning,Reinforcement Learning,RLHF,Shape,Training}
}

@thesis{k-MixingMusicUsingDeep-2019,
  title = {Mixing {{Music Using Deep Reinforcement Learning}}},
  author = {Kronvall, Viktor},
  date = {2019},
  url = {https://urn.kb.se/resolve?urn=urn:nbn:se:kth:diva-266349},
  urldate = {2026-04-27},
  abstract = {Deep Reinforcement Learning has recently seen good results in tasks such as board games, computer games and the control of autonomous vehicles. Stateof-the-art autonomous DJ-systems generating mixed audio hard-code the mixing strategy commonly with a cross-fade transition. This research investigates whether Deep Reinforcement Learning is an appropriate method for learning a mixing strategy that can yield more expressive and varied mixes than the hard-coded mixing strategies by adapting the strategies to the songs played. To investigate this, a system named the DeepFADE system was constructed. The DeepFADE system was designed as a three-tier system of hierarchical Deep Reinforcement Learning models. The first tier selects an initial song and limits the song collection to a smaller subset. The second tier selects when to transition to the next song by loading the next song at pre-selected cue points. The third tier is responsible for generating a transition between the two loaded songs according to the mixing strategy. Two Deep Reinforcement Learning algorithms were evaluated, A3C and Dueling DQN. Convolutional and residual neural networks were used to train the reinforcement learning policies. Rewards functions were designed as combinations of heuristic functions that evaluate the mixing strategy according to several important aspects of a DJ-mix such as alignment of beats, stability in output volume, tonal consonance, and time between transitions of songs. The trained models yield policies that are either unable to create transitions between songs or strategies that are similar regardless of playing songs. Thus the learnt mixing strategies were not more expressive than hard-coded cross-fade mixing strategies. The training suffers from reward hacking which was argued to be caused by the agent’s tendency to focus on optimizing only some of the heuristics. The reward hacking was mitigated somewhat by the design of more elaborate rewards that guides the policy to a larger extent.A survey was conducted with a sample size of n = 11. The small samplesize implies no statistically significant conclusions can be drawn. However, the mixes generated by the trained policy was rated more enjoyable compared to a randomized mixing strategy. The convergence rate of the training is slow and training time is not only limited by the optimization of the neural networks but also by the generation of audio used during training. Due to the limited available computational resources it is not possible to draw any clear conclusions whether the proposed method is appropriate or not when constructing the mixing strategy.},
  langid = {english}
}

@inproceedings{ktgyab-DJAIOptimizingPlaylist-2025,
  title = {{{DJ AI}}: {{Optimizing Playlist Alignment}} and {{Generating Transitions}} with {{Generative}} and {{Embedding Models}}},
  shorttitle = {{{DJ AI}}},
  booktitle = {Proceedings of the 20th {{International Audio Mostly Conference}}},
  author = {Kınay, Orkun and Tekdemir, Barış and Gökyılmaz, Göktuğ and Yavuz, Ekmel and Ay, Berk and Balcisoy, Selim},
  date = {2025-12-12},
  series = {{{AM}} '25},
  pages = {394--399},
  publisher = {Association for Computing Machinery},
  location = {New York, NY, USA},
  doi = {10.1145/3771594.3771640},
  url = {https://dl.acm.org/doi/10.1145/3771594.3771640},
  urldate = {2026-04-26},
  abstract = {Digital music platforms have transformed the listening experience through curated playlists and transitions, yet many transition creating systems primarily serve professional DJs with many manual features to be set while overlooking amateur performers and everyday listeners. In this study, we introduce DJ-AI, a novel framework that bridges this gap by analyzing detailed musical features to optimize song sequences and create harmonic transitions between those songs. Our approach employs graph based optimization techniques to efficiently arrange playlists by mapping song relationships and determining the best transition paths. Additionally, we integrate MusicGen—a generative model for generating coherent musical continuations—and MERT audio embedding model, which capture nuanced musical attributes, to enhance the smoothness of transitions. Experimental evaluations reveal that DJ-AI outperforms traditional crossfade methods in generating smooth and coherent transitions. This framework paves the way for AI-driven adaptive mixing solutions, making seamless music transitions more accessible to a broader audience.},
  isbn = {979-8-4007-2065-9}
}

@inproceedings{ktgyabb-DJAIIntelligentPlaylistSorting-2025,
  title = {{{DJ-AI}}: {{Intelligent Playlist Sorting}} and {{Seamless Generative Transitions}}},
  shorttitle = {{{DJ-AI}}},
  booktitle = {2025 33rd {{Signal Processing}} and {{Communications Applications Conference}} ({{SIU}})},
  author = {Kınay, Orkun and Tekdemir, Barış and Gökyılmaz, Göktuğ and Yavuz, Ekmel and Ay, Berk and Balcısoy, Selim and Bilimleri Fakültesi, Doğa},
  date = {2025-06},
  pages = {1--4},
  issn = {2165-0608},
  doi = {10.1109/SIU66497.2025.11112407},
  url = {https://ieeexplore.ieee.org/abstract/document/11112407},
  urldate = {2026-04-27},
  abstract = {The music listening experience has improved and become more accessible over time with digital playlists and real-time DJ transitions. However, existing systems do not cover fully amateur DJs and end users. Unlike traditional recommendation algorithms, DJ-AI analyzes songs not only based on user preferences but also their musical features to generate optimal sequences and seamless transitions. Graph-based optimization methods have been used for playlist arrangement, while MusicGen and MERT embeddings have been integrated to enhance transition smoothness. Experimental results show that the proposed method improves the listener experience by increasing transition compatibility.},
  eventtitle = {2025 33rd {{Signal Processing}} and {{Communications Applications Conference}} ({{SIU}})},
  keywords = {Artificial intelligence,Artificial Intelligence,Multiple signal classification,Music,Music Transition,Optimization methods,Playlist Optimization,Real-time systems,Signal processing,Signal processing algorithms,Sorting}
}

@article{lss-RightMusicRightTime-2019,
  title = {The {{Right Music}} at the {{Right Time}}: {{Adaptive Personalized Playlists Based}} on {{Sequence Modeling1}}},
  shorttitle = {The {{Right Music}} at the {{Right Time}}},
  author = {Liebman, Elad and Saar-Tsechansky, Maytal and Stone, Peter},
  date = {2019-09-01},
  journaltitle = {Management Information Systems Quarterly},
  shortjournal = {MIS Quarterly},
  volume = {43},
  number = {3},
  pages = {765--786},
  publisher = {MIS Quarterly},
  issn = {0276-7783},
  doi = {10.25300/MISQ/2019/14750},
  url = {https://dx.doi.org/10.25300/MISQ/2019/14750},
  urldate = {2026-04-27},
  abstract = {Recent years have seen a growing focus on automated personalized services, with music recommendations a particularly prominent domain for such contributions. However, while most prior work on music recommender systems has focused on preferences for songs and artists, a fundamental aspect of human music perception is that music is experienced in a temporal context and in sequence. Hence, listeners’ preferences also may be affected by the sequence in which songs are being played and the corresponding song transitions. Moreover, a listener’s sequential preferences may vary across circumstances, such as in response to different emotional or functional needs, so that different song sequences may be more satisfying at different times. It is therefore useful to develop methods that can learn and adapt to individuals’ sequential preferences in real time, so as to adapt to a listener’s contextual preferences during a listening session. Prior work on personalized playlists either considered batch learning from large historical data sets, attempted to learn preferences for songs or artists irrespective of the sequence in which they are played, or assumed that adaptation occurs over extended periods of time. Hence, this prior work did not aim to adapt to a listener’s current song and sequential preferences in real time, during a listening session. This paper develops and evaluates a novel framework for online learning of and adaptation to a listener’s current song and sequence preferences exclusively by interacting with the listener, during a listening session. We evaluate the framework using both real playlist datasets and an experiment with human listeners. The results establish that the framework effectively learns and adapts to a listeners’ transition preferences during a listening session, and that it yields a significantly better listener experience. Our research also establishes that future advances of online adaptation to listener’s temporal preferences is a valuable avenue for research, and suggests that similar benefits may be possible from exploring online learning of temporal preferences for other personalized services.},
  langid = {english}
}

@inproceedings{tckcrd-AutomaticMusicPlaylistGeneration-2023,
  title = {Automatic {{Music Playlist Generation}} via {{Simulation-based Reinforcement Learning}}},
  booktitle = {Proceedings of the 29th {{ACM SIGKDD Conference}} on {{Knowledge Discovery}} and {{Data Mining}}},
  author = {Tomasi, Federico and Cauteruccio, Joseph and Kanoria, Surya and Ciosek, Kamil and Rinaldi, Matteo and Dai, Zhenwen},
  date = {2023-08-04},
  series = {{{KDD}} '23},
  pages = {4948--4957},
  publisher = {Association for Computing Machinery},
  location = {New York, NY, USA},
  doi = {10.1145/3580305.3599777},
  url = {https://dl.acm.org/doi/10.1145/3580305.3599777},
  urldate = {2026-04-26},
  abstract = {Personalization of playlists is a common feature in music streaming services, but conventional techniques, such as collaborative filtering, rely on explicit assumptions regarding content quality to learn how to make recommendations. Such assumptions often result in misalignment between offline model objectives and online user satisfaction metrics. In this paper, we present a reinforcement learning framework that solves for such limitations by directly optimizing for user satisfaction metrics via the use of a simulated playlist-generation environment. Using this simulator we develop and train a modified Deep Q-Network, the action head DQN (AH-DQN), in a manner that addresses the challenges imposed by the large state and action space of our RL formulation. The resulting policy is capable of making recommendations from large and dynamic sets of candidate items with the expectation of maximizing consumption metrics. We analyze and evaluate agents offline via simulations that use environment models trained on both public and proprietary streaming datasets. We show how these agents lead to better user-satisfaction metrics compared to baseline methods during online A/B tests. Finally, we demonstrate that performance assessments produced from our simulator are strongly correlated with observed online metric results.},
  isbn = {979-8-4007-0103-0}
}

@article{vd-rawaudioseamlessmix-2018,
  title = {From Raw Audio to a Seamless Mix: Creating an Automated {{DJ}} System for {{Drum}} and {{Bass}}},
  shorttitle = {From Raw Audio to a Seamless Mix},
  author = {Vande Veire, Len and De Bie, Tijl},
  date = {2018-09-24},
  journaltitle = {EURASIP Journal on Audio, Speech, and Music Processing},
  shortjournal = {J AUDIO SPEECH MUSIC PROC.},
  volume = {2018},
  number = {1},
  pages = {13},
  issn = {1687-4722},
  doi = {10.1186/s13636-018-0134-8},
  url = {https://doi.org/10.1186/s13636-018-0134-8},
  urldate = {2026-04-27},
  abstract = {We present the open-source implementation of the first fully automatic and comprehensive DJ system, able to generate seamless music mixes using songs from a given library much like a human DJ does.},
  langid = {english},
  keywords = {Computational creativity,DJ,Drum and Bass,Machine learning,MIR}
}

@inproceedings{w-HybridRecommendationMusicBased-2020,
  title = {A {{Hybrid Recommendation}} for {{Music Based}} on {{Reinforcement Learning}}},
  booktitle = {Advances in {{Knowledge Discovery}} and {{Data Mining}}},
  author = {Wang, Yu},
  editor = {Lauw, Hady W. and Wong, Raymond Chi-Wing and Ntoulas, Alexandros and Lim, Ee-Peng and Ng, See-Kiong and Pan, Sinno Jialin},
  date = {2020},
  pages = {91--103},
  publisher = {Springer International Publishing},
  location = {Cham},
  doi = {10.1007/978-3-030-47426-3_8},
  abstract = {The key to personalized recommendation system is the prediction of users’ preferences. However, almost all existing music recommendation approaches only learn listeners’ preferences based on their historical records or explicit feedback, without considering the simulation of interaction process which can capture the minor changes of listeners’ preferences sensitively. In this paper, we propose a personalized hybrid recommendation algorithm for music based on reinforcement learning (PHRR) to recommend song sequences that match listeners’ preferences better. We firstly use weighted matrix factorization (WMF) and convolutional neural network (CNN) to learn and extract the song feature vectors. In order to capture the changes of listeners’ preferences sensitively, we innovatively enhance simulating interaction process of listeners and update the model continuously based on their preferences both for songs and song transitions. The extensive experiments on real-world datasets validate the effectiveness of the proposed PHRR on song sequence recommendation compared with the state-of-the-art recommendation approaches.},
  isbn = {978-3-030-47426-3},
  langid = {english},
  keywords = {Hybrid recommendation,Markov decision process,Music recommendation,Reinforcement learning,Weighted matrix factorization}
}
```

