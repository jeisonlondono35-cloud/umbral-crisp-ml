# Final presentation · script (20 minutes, English)

Speakers: **J.A** Jeison (≈7½ min) · **J.P** Juan Pablo (≈7 min) · **J.Q** Johan (≈5½ min).
Every member explains technical content and shows evidence on screen. Record the screen at 1280×720 or more, with the
browser zoom at 110–125% so the text is readable. Use the test accounts that the teacher received (never show a
password on screen: type it with the field hidden, or cut that part).

Before recording:
- Open https://umbral-crisp-ml.onrender.com 2–3 minutes before (the free server sleeps and takes ~1 minute to wake up).
- Tabs ready: homepage (PGC) · Phase 1, 2, 3, 4, 6 pages · the final report PDF · GitHub repository · a private window
  for the new player account · a normal window signed in as the analyst.
- Prepare 40 actions to paste in the session form, for example:
  `shoot, shoot, dodge, melee, shoot, block, shoot, dodge, jump, hide` (repeat it 4 times).

---

## 0:00 – 4:00 · Team, problem, objectives, reference framework — **J.Q**

**Screen:** homepage → section 1 (Project Title) → 2 → 3 → 4 → 5.

1. (0:00) "Hello, we are Jeison Arboleda, Juan Pablo Lugo and Johan Quintero, from the Machine Learning course at
   Universidad de Cundinamarca. Our project is *UMBRAL: Machine Learning Player Modeling for Fair Adaptive Boss Fights,
   Integrated Into a Player Information System*."
2. (0:30) **Problem.** "UMBRAL is our pixel-art game for web, Android and PC. Its bosses learn from the player during the
   fight with three small online models. Nobody had checked if those models really work, and a boss that reads the
   player too well or too badly makes the fight unfair. The player data was also in different places."
   Show *Affected population* and *Supporting evidence*: flow theory (Csikszentmihalyi; Chen), dynamic difficulty
   adjustment (Hunicke; Zohaib), and our own measurement: the same kind of boss cost between 4 and 13 lives.
3. (1:45) **Justification.** "The learning bosses are the main idea of the game, and they only work if they are fair. The
   intelligent system evaluates the models, puts the model inside a real decision with a person who approves, and
   monitors it."
4. (2:30) **Objectives.** Read the general objective. Scroll the table of specific objectives SO1–SO7: "each one has an
   indicator and the result that we really obtained; six are met and one is partly met — we come back to it at the end."
5. (3:15) **Reference framework.** "Player modeling (Yannakakis and Togelius), related studies like Missura and Gärtner,
   who predicted the right difficulty from short sessions, gradient boosting (Friedman; LightGBM), CRISP-ML(Q) by Studer
   and others, and production ML and security: technical debt (Sculley), concept drift (Gama) and OWASP. All the sources
   are in section 7 in APA format." → hand over to Jeison.

## 4:00 – 10:00 · The six CRISP-ML phases — **J.A**

**Screen:** Phase pages 1 → 6, `experiments.csv`, figures.

1. (4:00) **Phase 1.** "I created the game, so I was the domain expert. We translated the business goals into four ML
   goals: next action, hit probability, dodge position and speed on phones. We fixed six success criteria **before**
   training." Show the success criteria table.
2. (5:00) **Phase 2 · data preparation.** "We added an anonymous play log to the game: no names or emails. We found
   duplicated packages, the value 9999 for 'no enemy', and long pauses." Show the before/after table and Appendix B rows.
   "The most important decision: we split by **player**, 28/10/10, so the same person is never in training and test. The
   real log is still below the minimum we fixed, so this version uses simulated players and says SIMULATED everywhere."
3. (6:30) **Phase 3 · model comparison.** "We tried seven kinds of models for the next action — from the majority class
   and the game's own Markov model to logistic regression, random forest and gradient boosting — with group
   cross-validation by player. 22 experiments in the log." Open `experiments.csv` for two seconds.
4. (7:30) **Phase 4 · evaluation and selection.** Show the comparison figure and the final test table.
   "Gradient boosting reached 54.9% with players it never saw, against 53.1% for the game model and 36.7% for the
   simplest guess. For attacks, the logistic regression had the best Brier score, 0.189 against 0.196. We checked
   overfitting, bias by difficulty, and I reviewed the results as the domain expert. All criteria were met."
   One honest sentence: "the difference with the game model is small; the bigger gain is that now we know."
5. (9:00) **Phases 5 and 6.** "We saved each model with its preprocessing and a model card, and published it on Render.
   For monitoring we use the PSI: in a simulated test with a change in week 4, the PSI found it before the accuracy
   fell. GitHub Actions retrains every Monday." → "Juan Pablo will show how the model works inside the system."

## 10:00 – 17:00 · Live demo of the published system — **J.P**

**Screen:** the published site. Follow the steps exactly; this is the complete case asked by the activity.

1. (10:00) **Registration** (private window): Create account → new user (for example `demo_player`) → "the form has no
   role field: public registration always creates a player, and the password is stored as a scrypt hash."
2. (10:45) **Login and permissions:** as the new player, open `/app/users` → *Forbidden* (show it). "Authorization is
   checked on the server for each route and each object; this attempt is now in the audit log."
3. (11:15) **Business data management:** in *My account* → **+ Saved game** (world 2, 3 stars). Show one rule: try
   relics greater than the world → the error message. Then **+ Upload session**: world II, Normal, 10 minutes, 6 deaths,
   lost, 0 hearts, paste the 40 actions → Save. "The session is *registered*; the player can still edit it."
4. (12:30) **Model execution** (analyst window): Dashboard → the new session is in *Sessions waiting for the model* →
   **Analyze**. "The server rebuilds the features with the same code as training and gives the model the earlier
   sessions of the player, as in training."
5. (13:15) **Interpretation of the output:** read the three boxes: predictability (for example 51% vs a 38% baseline =
   the bosses read this player), next action with its probability, and hit risk. Read the *How to read it* box and the
   limits: simulated training data, fewer than 30 actions moves a lot.
6. (14:30) **Use in a decision:** *Decision supported by the model*: "the rule says LOWER Normal → Easy because the
   player died 6 times per 10 minutes and the hit risk is 39%, above the top quarter (36%)." Show that **Reject** needs
   a note. Press **Approve and apply**.
7. (15:30) **The result:** open the player profile → difficulty is now **Easy** and the change is in the history. Open
   **Model history**: date, user, model, version, latency of this inference. Open **Reports** (by difficulty and world)
   and show **Users** as admin (roles, cannot change own role). → hand over to Jeison.

## 17:00 – 18:30 · Monitoring and maintenance evidence — **J.A**

**Screen:** `/app/monitoring`, then GitHub Actions.

- "Monitoring shows the inferences, response time (p95 under 200 ms), errors, the accuracy of each week compared with
  the real next actions, and the PSI. The PSI needs 200 actions in 30 days, so the system says *not computed yet*
  instead of showing a number without meaning." Show the model version table.
- Show the GitHub Actions tab: tests on every push and the weekly retraining. "If a new model is worse, we revert the
  commit and the old version comes back."

## 18:30 – 20:00 · Schedule, objectives, limitations, conclusions — **J.Q**

**Screen:** homepage section 6 (schedule and contributions) → final report PDF (Results table) → section 7 and the
resource cards.

- Schedule: "planned against actual dates, taken from the Git history; each member's contribution is on the page and in
  the report."
- Objectives: "six of seven objectives are met; monitoring is partly met because drift needs more real data."
- **Demonstrated vs expected:** "We demonstrated that the process and the system work end to end on the public URL — the
  operational review passed 13 of 13 checks. What we still expect, and must measure with real players, is that the
  fights really feel fairer."
- Limitations and future work: simulated training data, few real sessions, manual upload of sessions; next: retrain
  with the real log, let the game upload sessions, a short fairness survey.
- (19:40) "Everything is on the homepage: the PGC documentation, the final report, the CRISP-ML evidence, the
  repository, the user manual and this video. Thank you."

---

Checklist after recording: total time 19–21 minutes · the three voices are heard · no password on screen · upload as
YouTube *unlisted* (or Google Drive with "anyone with the link") · put the link in `VIDEO_URL` in
`src/umbral_ml/pgc.py`, regenerate the final report and push.
