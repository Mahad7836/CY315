# IRS-Aided Physical Layer Security — DRL Project
## CY315 Wireless and Mobile Security | GIK Institute | Spring 2026

### Team
- M. Daniyal     2023406  — DRL agent (DDPG)
- M. Afeef Bari  2023356  — Results & report
- Mahad Aqeel    2023286  — Channel environment

### Project structure
```
project/
├── env/
│   ├── channel.py       ← IRS channel simulation (Week 1)
│   └── verify_env.py    ← sanity check & Week 1 plots
├── agent/
│   └── ddpg.py          ← DDPG agent (Week 2)
├── results/             ← all saved plots go here
├── report/              ← LaTeX files (Week 5)
└── main.py              ← full training pipeline
```

### Week-by-week plan
- Week 1: Channel environment ✓
- Week 2: Baseline DDPG (known Eve CSI, continuous phases)
- Week 3: Contribution 1 — remove Eve CSI, worst-case reward
- Week 4: Contribution 2 — discrete 2-bit phase shifts
- Week 5: Report + slides

### Baseline paper
Yang et al., "Deep Reinforcement Learning-Based Intelligent Reflecting
Surface for Secure Wireless Communications," IEEE Trans. Wireless Commun.,
vol. 20, no. 1, pp. 375-388, Jan. 2021.
