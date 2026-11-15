"""PGC documentation of the whole project (Activity 3): ONE source for the homepage (/pgc) and the final report.

Text may use <i>, <b> and <br/> only (valid in HTML and in reportlab). Numbers that come from the pipeline or the
system are read from reports/*.json by `numbers()`, so the page, the PDF and the repository say the same thing.
Edit the planned dates of SCHEDULE and the VIDEO_URL here; nothing else needs to change.
"""
from __future__ import annotations

import json

from .config import RAW, REPORTS, ROOT

PROJECT_TITLE = ("UMBRAL: Machine Learning Player Modeling for Fair Adaptive Boss Fights, "
                 "Integrated Into a Player Information System")
SHORT_TITLE = "UMBRAL: Player Modeling for Fair Adaptive Boss Fights"
VIDEO_URL = ""          # the 20-minute presentation (YouTube unlisted or Drive); empty = "coming soon"
TEAM = [("J.A", "Jeison Arboleda Londoño"), ("J.P", "Juan Pablo Lugo Cárdenas"),
        ("J.Q", "Johan Sebastián Quintero Ramírez")]


def numbers() -> dict:
    """The few numbers that the PGC text quotes, read from the result files."""
    R = json.loads((REPORTS / "results.json").read_text(encoding="utf-8"))
    A, B = R["A"], R["B"]
    pub = json.loads((RAW / "ml_public.json").read_text(encoding="utf-8"))
    out = {"source": R["source"], "n_exp": len(R["experiments"]), "rows": R["data"]["rows"],
           "split": R["data"]["split"], "real": R["data"]["real_available"],
           "acc": A["test"][A["chosen"]]["accuracy"], "acc_game": A["test"]["In-game Markov-2 (decay 0.97)"]["accuracy"],
           "acc_major": A["test"]["Majority class"]["accuracy"], "chosen_a": A["chosen"], "chosen_b": B["chosen"],
           "brier": B["test"][B["chosen"]]["brier"], "brier_game": B["test"]["In-game bandit (SGD, lr 0.3)"]["brier"],
           "next_ms": R["serving"]["next_ms"], "real_players": pub.get("players", 0), "real_made": pub.get("made", 0),
           "real_acc": pub["right"] / pub["made"] if pub.get("made") else None, "review": None, "tests": None}
    p = REPORTS / "operational_review.json"
    if p.exists():
        v = json.loads(p.read_text(encoding="utf-8"))
        if v.get("environment") == "DEPLOYED":
            out["review"] = (v["passed"], v["total"])
    tests = ROOT / "tests" / "test_system.py"
    out["tests"] = tests.read_text(encoding="utf-8").count("\ndef test_") if tests.exists() else None
    snap = REPORTS / "system_snapshot.json"
    out["snap"] = json.loads(snap.read_text(encoding="utf-8")) if snap.exists() else None
    return out


def pct(v) -> str:
    return "-" if v is None else f"{100 * v:.1f}%"


# ------------------------------------------------------------------------------------------------ sections
def problem(n: dict) -> list[tuple[str, list[str]]]:
    return [
        ("Context", [
            "UMBRAL is a pixel-art action game made by our team with the Godot engine. People play it on the web, on "
            "Android phones and on computers, and the same account works on all of them. The player fights bosses that "
            "learn during the fight: three small online models inside the game guess the next action of the player, "
            "where the player moves when a boss charges, and which attack will probably hit. The bosses use these "
            "guesses to fake an attack when the player always blocks, or to aim where the player usually dodges."]),
        ("The problem", [
            "These models were written to make the fights interesting, but nobody had checked if they really work, and "
            "nobody could see what they learn. A boss that reads the player too well, or one that guesses badly, makes the "
            "fight unfair: too hard for some players and too easy for others. The difficulty of the game is the same for "
            "everybody inside each level, while players are very different. At the same time, the information about the "
            "players (accounts, saved games, records, problems that they report) was in different places, so the team "
            "could not use it to decide anything."]),
        ("Affected population", [
            "The players of UMBRAL: the people who test the game at events and with friends, and the players with an "
            f"account. When we wrote this documentation the game had {n['real_players']} accounts with model data and "
            f"{n['real_made']:,} predictions made by the game model during real fights. The development team is also "
            "affected, because it needs evidence to change the difficulty, and so is the teacher who evaluates the "
            "process."]),
        ("Supporting evidence", [
            "Research on games shows that the difficulty must match the skill of the player: when the challenge is much "
            "higher than the skill the player feels anxiety, and when it is much lower the player gets bored "
            "(Csikszentmihalyi, 1990; Chen, 2007). This is the reason for dynamic difficulty adjustment (Hunicke, 2005; "
            "Zohaib, 2018) and for player modeling (Yannakakis &amp; Togelius, 2018). In our own game, the people who "
            "tested it said that some fights were \"very easy\" and others too hard, and in our measurements with an "
            "automatic player the same boss cost between 4 and 13 lives depending on the world. Our domain expert (J.A, "
            "the creator of the game) said in the Activity 1 review that his main worry was that the bosses could feel "
            "unfair for some players."]),
        ("Consequences", [
            "Without this project, the bosses would keep using models that nobody has evaluated; a mistake in them (like "
            "the wrong order of probabilities that we found in Activity 1) would stay hidden; players who die too much "
            "would leave the game; and the team would change the difficulty by opinion, without data and without a "
            "record of why it changed."]),
        ("Justification", [
            "The project is relevant because the bosses that learn are the main idea of UMBRAL, and a learning boss is only "
            "a good idea if it is fair. It is also a complete case of applied machine learning: real data from a product, a "
            "model that has to work fast on phones, and decisions that affect people. The intelligent system contributes "
            "in three ways. First, the CRISP-ML process measures the models with players that they never saw and chooses "
            "the best one with a rule fixed before training. Second, the information system brings the data of the players "
            "together and puts the model inside a real decision: when a play session arrives, the model reads the player "
            "and proposes a fairer difficulty, and a person approves or rejects it. Third, monitoring tells us when the "
            "model gets worse, so it can be trained again."]),
    ]


GENERAL_OBJECTIVE = ("Develop and publish an intelligent information system that models how each player of UMBRAL fights, "
                     "using machine learning built with the CRISP-ML process, and uses that model to support fair "
                     "difficulty decisions for each player, with authentication, traceability and monitoring.")


def specific_objectives(n: dict) -> list[dict]:
    rv = f"{n['review'][0]} of {n['review'][1]} checks passed on the public URL" if n["review"] else "not run yet"
    return [
        {"code": "SO1", "area": "Data preparation",
         "objective": "Collect the play traces of the game in an anonymous log and prepare a dataset without data leakage "
                      "(cleaning rules, new features and a split by player).",
         "indicator": "Documented cleaning rules; no player in two parts of the data.",
         "result": f"{n['rows']['actions']:,} actions, {n['rows']['hits']:,} boss attacks and {n['rows']['dodges']:,} "
                   f"dodges ({n['source']}); split by player {n['split']['train']}/{n['split']['val']}/"
                   f"{n['split']['test']}.", "status": "Met"},
        {"code": "SO2", "area": "Model development",
         "objective": "Train and compare at least five algorithms for each task with group cross-validation, and record "
                      "every experiment.",
         "indicator": "Experiment log with every model, its settings and its results.",
         "result": f"{n['n_exp']} experiments in reports/experiments.csv.", "status": "Met"},
        {"code": "SO3", "area": "Model evaluation",
         "objective": "Choose the models with a rule fixed before training and check them once with test players.",
         "indicator": "Next action: 10 points above the majority class and not below the game model; hit: lower Brier "
                      "score than the game model.",
         "result": f"{n['chosen_a']}: {pct(n['acc'])} (majority {pct(n['acc_major'])}, game {pct(n['acc_game'])}); "
                   f"{n['chosen_b']}: Brier {n['brier']:.3f} (game {n['brier_game']:.3f}).", "status": "Met"},
        {"code": "SO4", "area": "System implementation",
         "objective": "Build a web information system with registration, sign-in, three roles and the business processes "
                      "of players and accounts, with server-side inference stored with its model version.",
         "indicator": "Automatic tests of security, authorization, rules, workflow and monitoring.",
         "result": f"Flask + PostgreSQL system with 5 processes, 10 tables and {n['tests']} automatic tests that pass.",
         "status": "Met"},
        {"code": "SO5", "area": "Integration in a decision",
         "objective": "Turn the model output into a difficulty recommendation that a person approves or rejects, and "
                      "check the result with the next session.",
         "indicator": "Recommendations stored with the inference, the person who decided and the outcome.",
         "result": "Working on the public system: the first analyzed session produced a recommendation to lower the "
                   "difficulty, waiting for a decision.", "status": "Met"},
        {"code": "SO6", "area": "Validation of the solution",
         "objective": "Publish the system and validate it with one account of each role, including permissions, the "
                      "complete workflow and persistence.",
         "indicator": "Operational review of the deployed system.", "result": rv[:1].upper() + rv[1:] + ".",
         "status": "Met" if n["review"] and n["review"][0] == n["review"][1] else "Pending"},
        {"code": "SO7", "area": "Monitoring and maintenance",
         "objective": "Monitor errors, response times, accuracy against real actions and data drift, and retrain the "
                      "models every week.",
         "indicator": "Monitoring page and weekly retraining job.",
         "result": "Running. Drift (PSI) needs 200 actions in 30 days, so it is not computed yet; training still uses "
                   "simulated players until the real log reaches its minimum.", "status": "Partly met"},
    ]


def framework() -> list[tuple[str, list[str]]]:
    return [
        ("Background: difficulty, flow and player modeling", [
            "Csikszentmihalyi (1990) described flow, the state in which a person is fully involved in an activity because "
            "the challenge matches the skill. Chen (2007) applied this idea to games: a game keeps players in flow only if "
            "it adapts to different players. Dynamic difficulty adjustment (DDA) changes the game while people play "
            "(Hunicke, 2005); Zohaib (2018) reviews many DDA methods and shows that most of them use a model of the player. "
            "Player modeling, which means learning a computational model of how a person plays, is a central topic of "
            "artificial intelligence in games (Yannakakis &amp; Togelius, 2018). <b>Relationship with the project:</b> our "
            "bosses are a form of DDA, and our models are player models; the information system adds a person who "
            "approves each difficulty change."]),
        ("Related projects and studies", [
            "Drachen et al. (2009) grouped players of <i>Tomb Raider: Underworld</i> by their behavior data, and showed that "
            "play traces can describe different player types. Missura and Gärtner (2009) predicted the right difficulty "
            "for each player from short game sessions, which is very close to our decision rule. Li et al. (2010) used a "
            "contextual bandit to personalize content, the same family of method as the hit model of our game. "
            "<b>Relationship:</b> we follow the same ideas, but we evaluate them inside a real game, against the models "
            "that the game already uses, and with a person in the decision."]),
        ("Theoretical foundations of the models", [
            "Gradient boosting builds many small trees, each one correcting the errors of the previous ones (Friedman, "
            "2001); the histogram version that we use comes from LightGBM (Ke et al., 2017) and accepts missing values. "
            "Random forests (Breiman, 2001) and logistic regression were the other candidates, and a Markov table of the "
            "last actions was the model of the game. We measured probabilities with the Brier score (Brier, 1950) and the "
            "online models with prequential evaluation, which first predicts and then learns (Dawid, 1984). All the "
            "models were built with scikit-learn (Pedregosa et al., 2011). <b>Relationship:</b> these choices explain why "
            "the chosen next-action model is gradient boosting and why the hit model is evaluated by its probabilities."]),
        ("Process model: CRISP-ML", [
            "CRISP-DM organized data mining in phases (Chapman et al., 2000). CRISP-ML(Q) adapts it to machine learning, "
            "with six phases and quality checks in each one: business and data understanding, data engineering, model "
            "engineering, evaluation, deployment, and monitoring and maintenance (Studer et al., 2021). "
            "<b>Relationship:</b> it is the method of the whole project (see Methodology)."]),
        ("Information systems and machine learning in production", [
            "An information system collects, stores and processes data to support the decisions of an organization "
            "(Laudon &amp; Laudon, 2020). A model in production needs more than the model: data checks, configuration, "
            "monitoring and versions, or it becomes technical debt (Sculley et al., 2015; Amershi et al., 2019). Data "
            "changes over time (concept drift), so models must be watched and updated (Gama et al., 2014); we measure "
            "it with the Population Stability Index (Siddiqi, 2006). <b>Relationship:</b> this is why the system stores "
            "every inference with its model version, compares predictions with real actions and retrains every week."]),
        ("Security of web applications", [
            "The OWASP Top 10 lists the most common risks of web applications, such as broken access control and "
            "cryptographic failures (OWASP Foundation, 2021). Passwords must be stored with a slow hash with salt, such as "
            "scrypt (Percival &amp; Josefsson, 2016). <b>Relationship:</b> the system checks access on the server for every "
            "route and object, uses CSRF tokens, locks accounts after failed attempts and stores only scrypt hashes."]),
        ("Technical concepts and tools", [
            "Flask for the web application (Pallets Projects, 2024), SQLAlchemy and PostgreSQL on Supabase for the data "
            "(Supabase, 2024), Render for publishing (Render, 2024), Godot for the game (Godot Engine, 2024), and GitHub "
            "Actions for the tests and the weekly retraining. The dataset is our own anonymous play log, plus simulated "
            "players that follow the rules of the game while the real log grows (Arboleda Londoño et al., 2026)."]),
    ]


def methodology(n: dict) -> list[tuple[str, str]]:
    return [
        ("Phase 1: Business and data understanding",
         "We described the problem and the people, translated the goals of the game into four machine learning goals "
         "(next action, hit probability, dodge position, speed on phones), listed the data sources and their risks, and "
         "fixed six success criteria before training. J.A, the creator of the game, was the domain expert."),
        ("Phase 2: Data engineering",
         f"We added an anonymous play log to the game (Supabase, no names or emails). We removed duplicated packages, "
         f"changed the value 9999 to missing, limited long pauses, and created features from the game: the last two "
         f"actions with a 4-second reset, the history of the player and the probabilities of the game model. We split "
         f"the data by player ({n['split']['train']}/{n['split']['val']}/{n['split']['test']}) so the same person is never in "
         f"training and test. The real log did not reach its minimum yet, so this version uses simulated players and "
         f"says SIMULATED next to every number."),
        ("Phase 3: Model engineering",
         f"We trained seven kinds of models for the next action and six for the attacks, with grid search and group "
         f"cross-validation by player, and saved {n['n_exp']} experiments with their settings and results."),
        ("Phase 4: Evaluation",
         f"We chose with validation players and used the test players only once. Gradient boosting reached "
         f"{pct(n['acc'])} against {pct(n['acc_game'])} for the game model and {pct(n['acc_major'])} for the simplest "
         f"guess; the logistic regression for attacks had the best Brier score. We checked overfitting, bias by "
         f"difficulty and mode, and reviewed the results with the domain expert."),
        ("Phase 5: Deployment",
         f"We saved each model with its preprocessing and a model card, and published a Flask application on Render with "
         f"the six phases, a demo and an API (about {n['next_ms']:.0f} ms per prediction). In Activity 2 we extended the "
         f"same application into the information system: accounts and roles, five business processes, a PostgreSQL "
         f"database on Supabase, and the model inside the play-session process, where it supports a difficulty decision."),
        ("Phase 6: Monitoring and maintenance",
         "The system stores every error and every inference (user, model version, latency), compares each prediction with "
         "the real next action, measures drift with the PSI, and GitHub Actions retrains the models every Monday. A "
         "simulated test with a change in week 4 showed that the PSI finds the change before the accuracy falls."),
        ("Development and integration of the information system",
         "We designed the processes of players and accounts (accounts, saved games, play sessions and difficulty, records, "
         "support) and their rules, the database model and the permissions of each role. The model service uses the same "
         "feature code as training and gives the model the earlier sessions of the player, as in training (we found this "
         "difference by comparing features row by row). We wrote automatic tests, published the system and ran an "
         "operational review with one account of each role. In Activity 3 we wrote this documentation on the homepage, "
         "the final report and the presentation."),
    ]


# planned = our plan at the start of each activity; actual = dates in the Git history of the game and of this repository
SCHEDULE = [
    # activity, responsible, planned, actual, deliverable
    ("In-game online models (next action, dodge, contextual bandit) and visible ML", "J.A, J.P",
     "2026-10-19 – 2026-10-31", "2026-10-21 – 2026-10-31", "Bosses that learn in the game"),
    ("Accounts, records and anonymous play log on Supabase", "J.A", "2026-11-02 – 2026-11-07",
     "2026-11-07 – 2026-11-08", "SQL scripts and play log"),
    ("Activity 1, phases 1–2: problem, data, cleaning and split", "J.A (domain expert), J.Q",
     "2026-11-09 – 2026-11-11", "2026-11-14", "Data report and phase pages"),
    ("Activity 1, phases 3–4: training, comparison, evaluation, expert review", "J.A", "2026-11-11 – 2026-11-13",
     "2026-11-14", "22 experiments, chosen models"),
    ("Activity 1, phases 5–6: Flask app on Render, monitoring, weekly retraining", "J.A, J.Q",
     "2026-11-12 – 2026-11-14", "2026-11-14", "Published app and GitHub Actions"),
    ("Activity 1 report R3A2 Traces in Data", "J.Q, J.A", "2026-11-14", "2026-11-14", "PDF report"),
    ("Activity 2: design and implementation of the information system", "J.P", "2026-11-14", "2026-11-14",
     "Accounts, roles, processes, model in the workflow"),
    ("Activity 2: deployment (Supabase + Render), tests and operational review", "J.P, J.Q", "2026-11-14",
     "2026-11-14", "Public system, automatic tests, review 13/13"),
    ("Activity 2 report R3A3 Speaking Models and manuals", "J.P", "2026-11-14", "2026-11-14",
     "PDF report, technical and user manuals"),
    ("Activity 3: PGC documentation on the homepage and final report", "J.Q, J.P", "2026-11-15", "2026-11-15",
     "PGC pages and final PDF"),
    ("Activity 3: 20-minute presentation in English", "J.A, J.P, J.Q", "2026-11-16", "Pending", "Video on the homepage"),
]

CONTRIBUTIONS = [
    ("J.A", "Jeison Arboleda Londoño",
     "Creator and lead developer of UMBRAL and domain expert. Wrote the in-game models (next-action predictor, dodge "
     "model, contextual bandit) and the bosses that use them, the anonymous play log and the Supabase accounts. In "
     "Activity 1 built the CRISP-ML pipeline (data, 22 experiments, evaluation, model card), the Flask application, "
     "the monitoring test and the weekly retraining. Owner of the model."),
    ("J.P", "Juan Pablo Lugo Cárdenas",
     "In the game, worlds II and III, animated bosses and the first visible ML (heat maps, summary, predictive "
     "counterattack). In Activity 2 designed and built the information system: authentication and roles, the five "
     "business processes, the model service with the history of the player, the decision rule, monitoring, tests, the "
     "deployment on Render and Supabase, the operational review and the report R3A3."),
    ("J.Q", "Johan Sebastián Quintero Ramírez",
     "In the game, the start menu, controller support, adaptive tutorial texts and the English translation tool. In "
     "the project, the website and the reports: the phase pages and the report of Activity 1, the review of tickets "
     "and opinions in the monitoring routine, the PGC documentation and the final report of Activity 3."),
]

REFERENCES = [
    "Amershi, S., Begel, A., Bird, C., DeLine, R., Gall, H., Kamar, E., Nagappan, N., Nushi, B., &amp; Zimmermann, T. "
    "(2019). Software engineering for machine learning: A case study. In <i>Proceedings of the 41st International "
    "Conference on Software Engineering: Software Engineering in Practice</i> (pp. 291–300). IEEE. "
    "https://doi.org/10.1109/ICSE-SEIP.2019.00042",
    "Arboleda Londoño, J., Lugo Cárdenas, J. P., &amp; Quintero Ramírez, J. S. (2026). <i>UMBRAL anonymous play log and "
    "simulated players</i> [Data set]. https://github.com/jeisonlondono35-cloud/umbral-crisp-ml",
    "Breiman, L. (2001). Random forests. <i>Machine Learning, 45</i>(1), 5–32. https://doi.org/10.1023/A:1010933404324",
    "Brier, G. W. (1950). Verification of forecasts expressed in terms of probability. <i>Monthly Weather Review, "
    "78</i>(1), 1–3. https://doi.org/10.1175/1520-0493(1950)078%3C0001:VOFEIT%3E2.0.CO;2",
    "Chapman, P., Clinton, J., Kerber, R., Khabaza, T., Reinartz, T., Shearer, C., &amp; Wirth, R. (2000). <i>CRISP-DM "
    "1.0: Step-by-step data mining guide</i>. SPSS.",
    "Chen, J. (2007). Flow in games (and everything else). <i>Communications of the ACM, 50</i>(4), 31–34. "
    "https://doi.org/10.1145/1232743.1232769",
    "Csikszentmihalyi, M. (1990). <i>Flow: The psychology of optimal experience</i>. Harper &amp; Row.",
    "Dawid, A. P. (1984). Present position and potential developments: Some personal views. Statistical theory: The "
    "prequential approach. <i>Journal of the Royal Statistical Society: Series A, 147</i>(2), 278–292. "
    "https://doi.org/10.2307/2981683",
    "Drachen, A., Canossa, A., &amp; Yannakakis, G. N. (2009). Player modeling using self-organization in Tomb Raider: "
    "Underworld. In <i>2009 IEEE Symposium on Computational Intelligence and Games</i> (pp. 1–8). IEEE. "
    "https://doi.org/10.1109/CIG.2009.5286500",
    "Friedman, J. H. (2001). Greedy function approximation: A gradient boosting machine. <i>The Annals of Statistics, "
    "29</i>(5), 1189–1232. https://doi.org/10.1214/aos/1013203451",
    "Gama, J., Žliobaitė, I., Bifet, A., Pechenizkiy, M., &amp; Bouchachia, A. (2014). A survey on concept drift "
    "adaptation. <i>ACM Computing Surveys, 46</i>(4), Article 44. https://doi.org/10.1145/2523813",
    "Godot Engine. (2024). <i>Godot Engine</i> (Version 4) [Computer software]. https://godotengine.org",
    "Hunicke, R. (2005). The case for dynamic difficulty adjustment in games. In <i>Proceedings of the 2005 ACM SIGCHI "
    "International Conference on Advances in Computer Entertainment Technology</i> (pp. 429–433). ACM. "
    "https://doi.org/10.1145/1178477.1178573",
    "Ke, G., Meng, Q., Finley, T., Wang, T., Chen, W., Ma, W., Ye, Q., &amp; Liu, T.-Y. (2017). LightGBM: A highly "
    "efficient gradient boosting decision tree. In <i>Advances in Neural Information Processing Systems 30</i> "
    "(pp. 3146–3154). Curran Associates.",
    "Laudon, K. C., &amp; Laudon, J. P. (2020). <i>Management information systems: Managing the digital firm</i> "
    "(16th ed.). Pearson.",
    "Li, L., Chu, W., Langford, J., &amp; Schapire, R. E. (2010). A contextual-bandit approach to personalized news "
    "article recommendation. In <i>Proceedings of the 19th International Conference on World Wide Web</i> "
    "(pp. 661–670). ACM. https://doi.org/10.1145/1772690.1772758",
    "Missura, O., &amp; Gärtner, T. (2009). Player modeling for intelligent difficulty adjustment. In <i>Discovery "
    "Science: 12th International Conference</i> (pp. 197–211). Springer. https://doi.org/10.1007/978-3-642-04747-3_17",
    "OWASP Foundation. (2021). <i>OWASP Top 10:2021</i>. https://owasp.org/Top10/",
    "Pallets Projects. (2024). <i>Flask documentation</i>. https://flask.palletsprojects.com/",
    "Pedregosa, F., Varoquaux, G., Gramfort, A., Michel, V., Thirion, B., Grisel, O., Blondel, M., Prettenhofer, P., "
    "Weiss, R., Dubourg, V., Vanderplas, J., Passos, A., Cournapeau, D., Brucher, M., Perrot, M., &amp; Duchesnay, É. "
    "(2011). Scikit-learn: Machine learning in Python. <i>Journal of Machine Learning Research, 12</i>, 2825–2830.",
    "Percival, C., &amp; Josefsson, S. (2016). <i>The scrypt password-based key derivation function</i> (RFC 7914). "
    "Internet Engineering Task Force. https://doi.org/10.17487/RFC7914",
    "Render. (2024). <i>Render documentation</i>. https://render.com/docs",
    "Sculley, D., Holt, G., Golovin, D., Davydov, E., Phillips, T., Ebner, D., Chaudhary, V., Young, M., Crespo, J.-F., "
    "&amp; Dennison, D. (2015). Hidden technical debt in machine learning systems. In <i>Advances in Neural Information "
    "Processing Systems 28</i> (pp. 2503–2511). Curran Associates.",
    "Siddiqi, N. (2006). <i>Credit risk scorecards: Developing and implementing intelligent credit scoring</i>. Wiley.",
    "Studer, S., Bui, T. B., Drescher, C., Hanuschkin, A., Winkler, L., Peters, S., &amp; Müller, K.-R. (2021). Towards "
    "CRISP-ML(Q): A machine learning process model with quality assurance methodology. <i>Machine Learning and "
    "Knowledge Extraction, 3</i>(2), 392–413. https://doi.org/10.3390/make3020020",
    "Supabase. (2024). <i>Supabase documentation</i>. https://supabase.com/docs",
    "Yannakakis, G. N., &amp; Togelius, J. (2018). <i>Artificial intelligence and games</i>. Springer. "
    "https://doi.org/10.1007/978-3-319-63519-4",
    "Zohaib, M. (2018). Dynamic difficulty adjustment (DDA) in computer games: A review. <i>Advances in Human-Computer "
    "Interaction, 2018</i>, Article 5681652. https://doi.org/10.1155/2018/5681652",
]

SECTIONS = [("title", "Project Title"), ("problem", "Problem Statement and Justification"),
            ("general", "General Objective"), ("specific", "Specific Objectives"),
            ("framework", "Reference Framework"), ("method", "Methodology and Development Schedule"),
            ("references", "Bibliographic References")]


def context() -> dict:
    """Everything the /pgc page and the final report need."""
    n = numbers()
    return {"title": PROJECT_TITLE, "short": SHORT_TITLE, "team": TEAM, "n": n, "problem": problem(n),
            "general": GENERAL_OBJECTIVE, "specific": specific_objectives(n), "framework": framework(),
            "method": methodology(n), "schedule": SCHEDULE, "contrib": CONTRIBUTIONS, "refs": REFERENCES,
            "sections": SECTIONS, "video": VIDEO_URL}
