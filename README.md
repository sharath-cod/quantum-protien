# 🧬 Quantum AI Protein Folding Solver
**Team AI Nexus | AI & DS, 4th Semester**

---

## 🚀 How to Run Locally (3 steps only!)

### Step 1 — Install packages
```bash
pip install -r requirements.txt
```

### Step 2 — Run the app
```bash
python app.py
```

### Step 3 — Open browser
```
http://localhost:5000
```

> ⚠️ Local dev needs `firebase_service_account.json` in the project folder.
> Production uses the `FIREBASE_KEY` environment variable on Render.

---

## 📁 Project Structure
```
protein-folding/
├── app.py                        ← Backend (Flask + Firebase + VQE)
├── requirements.txt              ← Python packages
├── render.yaml                   ← Render deploy config
├── .gitignore                    ← Keeps secrets off GitHub
├── README.md                     ← This file
├── firebase_service_account.json ← LOCAL ONLY — never pushed to GitHub
└── templates/
    └── index.html                ← Frontend (HTML + CSS + JS)
```

---

## 🌐 Live Deploy

| Service | Purpose | URL |
|---------|---------|-----|
| Render | Flask backend | https://protein-folding.onrender.com |
| Firebase Firestore | Database | console.firebase.google.com |
| UptimeRobot | Keep alive ping | uptimerobot.com |

### Deploy Steps
1. Push to GitHub (firebase_service_account.json is gitignored)
2. Connect repo to Render
3. Add `FIREBASE_KEY` environment variable on Render (paste full JSON)
4. Set UptimeRobot to ping your URL every 5 mins

---

## ✨ Features
- ✅ Protein sequence input with live amino acid visualization
- ✅ AI secondary structure prediction (Helix/Sheet/Coil/Turn)
- ✅ Quantum VQE energy minimization simulation
- ✅ Interactive 3D protein structure visualization (drag to rotate!)
- ✅ Energy convergence chart (20 VQE iterations)
- ✅ Quantum state probability distribution
- ✅ Energy landscape visualization
- ✅ Save/load analyses to Firebase Firestore
- ✅ Firebase Authentication (per-user data)
- ✅ 6 built-in example proteins (Insulin, Beta Amyloid, GFP...)
- ✅ Theory page explaining all concepts

---

## 🔌 API Endpoints

| Method | URL | Auth | What it does |
|--------|-----|------|-------------|
| GET | `/` | No | Opens the web app |
| POST | `/api/analyze` | Yes | Run full AI + Quantum analysis |
| GET | `/api/results` | Yes | Get all saved analyses |
| DELETE | `/api/results/<id>` | Yes | Delete a saved analysis |
| POST | `/api/mutate` | Yes | Point mutation analysis |
| GET | `/api/examples` | No | Get example protein sequences |

---

## 🧪 Try These Sequences

**Beta Amyloid (Alzheimer's related):**
```
DAEFRHDSGYEVHHQKLVFFAEDVGSNKGAIIGLMVGGVVIA
```

**Insulin A-chain:**
```
GIVEQCCTSICSLYQLENYCN
```

**Pure Alpha Helix:**
```
AELMAELMAELMAELM
```

**GFP Fragment:**
```
MSKGEELFTGVVPILVELDGDVNGHKFSVSGEGEGDATYGKLT
```

---

## 🗄️ Database — Firebase Firestore

All analysis results are saved to Firestore under the `protein_results` collection.
Each document stores: uid, sequence, ai_result, quantum_result, disease_risk, timestamps.

View data: Firebase Console → quantum-4232d → Firestore Database → protein_results

---

## 📊 How the AI Works
1. Analyzes amino acid sequence properties (hydrophobicity, charge, size)
2. Applies Chou-Fasman propensity rules for structure prediction
3. Scores Alpha Helix, Beta Sheet, Beta Turn, and Random Coil probabilities
4. Identifies helix and sheet regions in the sequence

## ⚛️ How the Quantum VQE Works
1. Builds Hamiltonian (energy operator) from amino acid interactions
2. Initializes parameterized quantum circuit (ansatz)
3. Uses exact diagonalization via scipy.linalg.eigh for ground state
4. Runs VQE iterations with COBYLA optimizer for convergence curve
5. Returns minimum energy state = predicted stable structure

---

## 🎓 Technologies Used
| Layer | Technology | Purpose |
|-------|-----------|---------|
| Frontend | HTML + CSS + JS | User interface |
| Backend | Python + Flask | API server |
| Database | Firebase Firestore | Cloud database |
| Auth | Firebase Auth | User authentication |
| AI Module | Chou-Fasman (Pure Python) | Sequence analysis |
| Quantum Module | Qiskit VQE | Energy minimization |
| Hosting | Render | Backend deployment |
| Uptime | UptimeRobot | Free tier keep-alive |
