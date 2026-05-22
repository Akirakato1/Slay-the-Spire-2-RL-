# RL Agent Architecture Reading List

This is the reading list for designing the Slay the Spire 2 RL agent, especially
the combat/strategy split, candidate-action policy heads, transformer memory,
and staged training curriculum.

## Core RL Algorithms And Architecture

1. **Proximal Policy Optimization Algorithms**
   - Link: https://arxiv.org/abs/1707.06347
   - Why read: likely first online RL optimizer baseline. Useful before trying
     more complex distributed or offline methods.

2. **Stabilizing Transformers for Reinforcement Learning**
   - Link: https://arxiv.org/abs/1910.06764
   - Why read: introduces GTrXL, a transformer variant designed to make
     transformer memory more stable in RL.

3. **IMPALA: Scalable Distributed Deep-RL with Importance Weighted Actor-Learner Architectures**
   - Link: https://arxiv.org/abs/1802.01561
   - Google Research page: https://research.google/pubs/impala-scalable-distributed-deep-rl-with-importance-weighted-actor-learner-architectures/
   - Why read: useful when scaling to many simulator workers and separating
     actors from learners.

4. **The Option-Critic Architecture**
   - Link: https://ojs.aaai.org/index.php/AAAI/article/view/10916
   - Why read: conceptual foundation for hierarchical policies and learned
     options, relevant to the combat-agent/strategy-agent split.

5. **Decision Transformer: Reinforcement Learning via Sequence Modeling**
   - NeurIPS page: https://papers.nips.cc/paper_files/paper/2021/hash/7f489f642a0ddb10272b5c31057f0663-Abstract.html
   - arXiv: https://arxiv.org/abs/2106.01345
   - Why read: useful if we collect trajectories and want offline pretraining
     or return-conditioned sequence modeling.

6. **Grandmaster Level in StarCraft II Using Multi-Agent Reinforcement Learning**
   - Nature: https://www.nature.com/articles/s41586-019-1724-z
   - DeepMind summary: https://deepmind.google/en/blog/alphastar-grandmaster-level-in-starcraft-ii-using-multi-agent-reinforcement-learning/
   - Why read: high-level reference for action heads, league training,
     large-scale RL systems, and complex strategic games.

## Card-Game And Deck-Building References

7. **Two-Step Reinforcement Learning for Multistage Strategy Card Game**
   - Link: https://arxiv.org/abs/2311.17305
   - Journal PDF: https://journals.pan.pl/Content/132310/PDF/BPASTS_2024_72_6_4317.pdf
   - Why read: directly relevant to staged training: simplified/foundational
     phase first, then full game, with separate agents for different decision
     phases.

8. **Playing Various Strategies in Dominion with Deep Reinforcement Learning**
   - AAAI AIIDE page: https://ojs.aaai.org/index.php/AIIDE/article/view/27518
   - DOI: https://doi.org/10.1609/aiide.v19i1.27518
   - Why read: deck-building RL reference. Especially relevant for representing
     decks as multisets and handling variable-size action sets.

9. **RLCard: A Toolkit for Reinforcement Learning in Card Games**
   - Link: https://arxiv.org/abs/1910.04376
   - Project docs: https://rlcard.org/
   - Why read: useful for environment API conventions, card-game RL baselines,
     and sparse-reward/game-interface design.

## Suggested Reading Order

1. Two-Step Reinforcement Learning for Multistage Strategy Card Game
2. Proximal Policy Optimization Algorithms
3. Stabilizing Transformers for Reinforcement Learning
4. Playing Various Strategies in Dominion with Deep Reinforcement Learning
5. The Option-Critic Architecture
6. IMPALA
7. Decision Transformer
8. AlphaStar
9. RLCard

## Decisions To Extract While Reading

For each paper, write down:

- what architecture idea it supports,
- what idea it warns against,
- what should be implemented now,
- what should wait until after a baseline works,
- what experiment would prove whether it applies to this simulator.

