# User Manual

## 1. Before you start

Open the address of the system. The cover page and the six phases of the project are public. To use the system you need an account: press **Create account** (you get a player account) or **Sign in**. Analyst and administrator accounts are given by an administrator.

## 2. For players

1. **Create your account.** Choose a user name, the name other people see, your gamer tag and JAY or JAKE. The password needs 8 characters with letters and numbers.
2. **My account** shows your profile, how many saved games, sessions and records you have, and the difficulty changes made to your account.
3. **Saved games.** In your profile press **+ Saved game**. Slot 0 is AUTO; 1 to 24 are the save windows of the game. The system checks the game rules: one relic per world, at most 16 stars per world.
4. **Upload a play session.** Press **+ Upload session** and write the date, mode, world, difficulty, minutes, deaths, result, hearts at the end and the combat actions in order (shoot, melee, block, dodge, jump, hide). You can edit or delete the session until an analyst analyzes it.
5. **See what the model learned.** When your session is analyzed, its page shows how predictable you are for the bosses, your next probable action and your hit risk, with an explanation and its limits. If the model proposes a change of difficulty, you see it and its result in your profile.
6. **Records.** Send a record from your profile. It appears on the leaderboard (Records) after an analyst verifies it.
7. **Support.** Use **Support → + New ticket** to report a problem or give your opinion (score 1 to 5).

## 3. For analysts

1. **Dashboard** shows the work queue: sessions waiting for the model and recommendations waiting for a decision.
2. **Analyze a session.** Open the session and press **Analyze with the model** (also from the dashboard or the Sessions list). In about one second you see predictability, next action, hit risk and the decision supported by the model.
3. **Decide.** If there is a recommendation, read the reason. **Approve and apply** changes the player's difficulty. To **Reject** you must write why. Pending decisions are also in **Recommendations**.
4. **Follow the result.** When the next session of that player is analyzed, the recommendation says if the change worked.
5. **Players.** Create profiles for testers without an account, edit them or suspend them. The difficulty of an existing account changes only through a recommendation.
6. **Records.** Verify or reject the records that wait for verification.
7. **Support.** Take a ticket, move it from open to in progress to resolved (with a resolution) to closed.
8. **Reports** show sessions, deaths, wins and model accuracy by difficulty and by world, recommendations that worked, tickets and opinions. **Download sessions (CSV)** exports the data.
9. **Monitoring** shows inferences, response time, errors, accuracy per week compared with real actions, data drift (PSI) and model versions. **Model history** lists every inference.

## 4. For administrators

Everything an analyst can do, plus **Users**: change roles (player, analyst, administrator) and disable accounts. You cannot change your own role, and one active administrator must always exist. The audit log shows sign-ins, role changes, locked accounts, difficulty changes and forbidden attempts. Only an administrator can delete an analyzed session or a player.

## 5. Short training guide (20 minutes)

1. (3 min) Open the cover page and read Phase 1 and Phase 4 to know what the model predicts and how good it is.
2. (4 min) Create a player account, add one saved game and upload a session with at least 20 actions.
3. (5 min) Sign in as the analyst test account, open the dashboard, analyze the session and read each number with the "How to read it" box.
4. (4 min) Approve or reject the recommendation and check the change in the player's profile.
5. (4 min) Open Reports and Monitoring and find: the model accuracy this week, the PSI and the last errors.

Demonstration script for the class: player uploads → analyst analyzes → recommendation approved → profile shows the new difficulty → Monitoring shows the inference with user, version and time.

## 6. Incident reporting and feedback

- **Who can report:** any signed-in user (Support → New ticket), players in the game (Pause → Report an error) and anyone in the GitHub repository (Issues).
- **Severity:** High = the system or a main function does not work (for example sign-in or the model). Medium = a function works badly or a model result looks unfair. Low = small problems and opinions.
- **Response time:** High the same day (or go back to the last good version); Medium within 3 days; Low in the weekly review.
- **Flow:** open → in progress (someone takes it) → resolved (with the resolution written) → closed. A resolved ticket can be opened again if the problem returns.
- **Unfair model results:** use the category "Model / unfair boss" from the session page. These tickets are read with the rejected recommendations in the weekly model review.
- **Opinions:** category "Opinion" with a score from 1 to 5. The mean score is in Reports.
