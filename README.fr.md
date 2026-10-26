# LLM ChatHub

Un client de tchat complet dans ComfyUI, branché sur **LM Studio**, **Ollama (local)**
ou **Ollama Cloud**, avec :

- un **panneau de chat** dans la barre latérale (streaming, historique, édition) ;
- des **nodes** pour brancher la réponse du modèle sur n'importe quel workflow
  (prompt → sampler, refus, légendes, traduction, prompt-engineering...) ;
- un **prompt de départ** : instructions système + contexte de référence + images + RAG ;
- une **historisation persistante** : les conversations sont enregistrées en base SQLite,
  reprises au redémarrage, éditables, régénérables, tronquables — comme dans ChatGPT ;
- un **RAG de documents** (dossier de fichiers texte) avec embeddings du provider et
  repli automatique sur une recherche BM25 mots-clés ;
- l'**envoi d'images** aux modèles de vision (Qwen-VL, Gemma 3, LLaVA, ...).

Aucune dépendance à installer : tout est déjà présent dans ComfyUI
(`requests`, `aiohttp`, `numpy`, `Pillow`).

> Depuis la version 1.1, **toute l'interface est en anglais** (libellés du panneau,
> options des nodes, messages d'erreur). Les workflows enregistrés avant gardent leurs
> valeurs françaises : elles restent acceptées (`aucun`, `effacer l'historique`,
> `premiere image`, `chaque image du lot`...).

---

## Exemples

Deux sessions reelles avec un modele local LM Studio. Le node **Chat** se place entre
le node **Session** et le graphe : sa sortie `reply` alimente *Preview as Text*, et le
JSON renvoye est ce que le workflow transforme en image finale.

**1 — Ecriture de prompt structure.** Le panneau contient le prompt systeme et la
demande ("a picture of a cat black on a roof, in jdon for BB8X") ; le node du graphe pose
la meme question et renvoie le JSON de scene (canvas, camera, items, bbox) qui pilote le
rendu. Meme conversation, deux points d'entree :

![Panneau de chat et chaine de nodes ecrivant un JSON de scene pour un rendu photorealiste](examples/example-1-photoreal-prompt.png)

**2 — Iteration sur place.** Meme session, un tour de plus : "change the style in anime
90's". Le modele reecrit tout le JSON (nouveaux `items`, nouveau `caption`, meme
composition) et le run suivant rend la version anime — sans copier-coller entre deux
runs :

![Meme chaine apres une demande de style anime annees 90, avec la nouvelle image rendue](examples/example-2-anime-restyle.png)

---

## 1. Installation

Copie le dossier `ComfyUI-LLM-ChatHub` dans `ComfyUI/custom_nodes/` puis redémarre
ComfyUI. Un nouvel onglet **ChatHub** apparaît dans la barre latérale, et la catégorie
**LLM ChatHub** dans le menu des nodes (clic droit → Add Node).

Données utilisateur (base de conversations + images) :

```
ComfyUI/user/llm_chathub/
├── llm_chathub.sqlite3   # conversations, messages, index RAG, réglages
├── settings.json         # réglages + clés API
├── images/               # images envoyées au modèle
└── documents/            # documents importés depuis le panneau (source du RAG)
```

> La variable d'environnement `LLM_CHATHUB_HOME` permet de déplacer ce dossier.

## 2. Configurer un provider

Onglet **ChatHub** → bouton **Settings**.

| Provider | URL de base par défaut | Clé API |
|---|---|---|
| `lmstudio` | `http://127.0.0.1:1234/v1` | non (LM Studio : onglet *Developer* → *Start Server*) |
| `ollama` | `http://127.0.0.1:11434` | non (`ollama serve` + `ollama pull <modele>`) |
| `ollama_cloud` | `https://ollama.com` | **oui** (https://ollama.com/settings/keys) |

1. Choisis le provider.
2. Le champ **model** liste tous les modèles que le provider sélectionné annonce :
   ouvre la liste et choisis-en un, ou tape un nom qui n'y figure pas. La liste se
   recharge toute seule quand tu changes de provider : les modèles apparaissent donc
   dès que tu choisis Ollama ou Ollama Cloud. **Reload models** la récupère à nouveau
   (après un `ollama pull`, ou après avoir chargé un autre modèle dans LM Studio) et
   **Test connection** vérifie seulement que le serveur répond. Si le champ est vide,
   le premier modèle de la liste est retenu automatiquement.
3. Colle la clé API seulement pour Ollama Cloud, puis **Save**.

Changer de provider prend effet tout de suite : il devient le provider des nouvelles
conversations **et** de celle affichée, donc le choix survit à un aller-retour vers
*Apps* / *Templates*. La ligne de statut indique quel modèle la conversation utilise
désormais ; **Save** ne sert plus qu'aux autres champs (température, prompt, RAG…).
**+ New** ouvre immédiatement la nouvelle conversation, et si elle ne peut pas être
créée l'erreur s'affiche au lieu de ne rien faire.

Chaque provider n'annonce que **ses** modèles. Un Ollama local liste aussi les
modèles cloud qu'il sait relayer (`…-cloud`, ou toute entrée portant un
`remote_host`) : ils sont masqués dans la liste **Ollama (local)**, pour qu'un choix
local tourne bien sur cette machine, et ils apparaissent côté **Ollama Cloud**. Les
base URL Ollama terminées par `/v1` ou `/api` sont acceptées : les chemins natifs sont
reconstruits depuis la racine, donc un `https://ollama.com/v1` copié continue de
fonctionner.

La clé peut aussi venir d'une variable d'environnement : `OLLAMA_API_KEY`,
`LMSTUDIO_API_KEY` / `LM_API_TOKEN`.

## 3. Utiliser le panneau de chat

**Settings** occupe tout le panneau tant qu'il est ouvert : l'historique, les pastilles
de statut et la zone de saisie s'effacent pour que toutes les options tiennent à
l'écran au lieu d'être dans une petite boîte avec ascenseur. **Hide settings** ramène
la conversation (et le champ de saisie). Sur une fenêtre très basse les réglages
défilent encore : il y a simplement moins de place que de contrôles.

1. **+ New** crée une conversation.
2. Ouvre **Settings** pour saisir :
   - le **prompt de depart** (instructions système, ex. « Tu écris des prompts de
     storyboard en anglais, format JSON »),
   - le **contexte de reference** (texte libre toujours inclus : style, charte,
     glossaire...),
   - la température, les max tokens et le nombre de **tours d'historique** envoyés.
3. Pour donner des documents au modele : coche **RAG on**, puis dans **Settings** →
   section *RAG documents* clique **Add documents** et choisis tes fichiers
   (`.txt .md .json .csv .srt .py .html`... plusieurs à la fois). Ils sont copiés dans
   `ComfyUI/user/llm_chathub/documents/`, listés dans le panneau (taille + `x` pour
   retirer) et **indexés automatiquement** : la réponse cite la source utilisée.
   Le champ dossier accepte aussi un chemin complet (ex. `D:\projets\docs`) pour
   indexer une arborescence existante, puis **Index documents**. Le bouton
   **Test RAG** montre les extraits que le modele recevrait pour une question.
   Les PDF/DOCX binaires ne sont pas lus : convertis-les en texte d'abord.
4. Écris ton message, joins une ou plusieurs images avec **Image**, **Send**.
   La réponse s'affiche en streaming.
5. Sur chaque message : **Copy**, **Edit** (relance la suite), **Regenerate**,
   **Delete**, **Truncate here**.

Le sélecteur en haut liste toutes les conversations enregistrées ; tout survit à un
redémarrage de ComfyUI.

### Modeles qui « reflechissent »

Les modeles de raisonnement (Qwen3, DeepSeek-R1, Gemma « thinking », ...) streament
leur chaine de pensee separement de la reponse. ChatHub l'affiche dans un bloc
repliable **Model reasoning** et ne la melange jamais a la reponse.

Attention : sur ces modeles le raisonnement consomme le budget de `Max tokens`. Si la
reponse revient vide avec la mention "The model reasoned without replying", augmente
`Max tokens` (2000 a 4000 est un bon point de depart). Une reponse vide n'est jamais
renvoyee au modele dans les tours suivants.

## 4. Utiliser les nodes

| Node | Rôle |
|---|---|
| **LLM ChatHub - Provider** | choisit provider / URL / modèle / température / max tokens. |
| **LLM ChatHub - Session + starting prompt** | ouvre (ou vide) une conversation persistante et fixe le prompt système, le contexte de référence et le RAG. Le bouton **RAG upload (files)** importe des documents directement depuis le node. |
| **LLM ChatHub - Chat** | envoie un message (+ image optionnelle + contexte optionnel) et renvoie la réponse. |
| **LLM ChatHub - RAG Indexer** | indexe un dossier de documents. |
| **LLM ChatHub - RAG Retrieve** | renvoie les extraits pertinents pour une requête. |
| **LLM ChatHub - History (read)** | relit la conversation dans le graphe. |
| **LLM ChatHub - History (edit)** | efface / tronque / change le prompt système. |

Chaîne type :

```
Provider ─► Session ─► Chat(reply) ─► ton workflow (CLIPTextEncode, sampler, ...)
                        ▲
                        └── RAG Retrieve(context)   (ou le champ context du Chat)
```

### `panel (saved settings)` veut bien dire le panneau

Avec le choix « panneau », la connexion d'un node (**URL**, **modèle**, clé API) est
celle enregistrée dans le panneau, et non les valeurs restées dans ses propres
widgets — elles gardent l'URL du provider utilisé avant, et les honorer envoyait une
conversation « Ollama Cloud » sur le port de LM Studio. `temperature`, `max_tokens` et
`history_limit` restent réglables par node. Pour imposer une URL ou un modèle précis,
choisis le provider explicitement au lieu de l'entrée panneau.

Changer de provider dans le panneau rejoue le node **Provider** au *Run* suivant,
même graphe inchangé : ComfyUI réutilise normalement la sortie en cache d'un node,
ce qui conservait le provider enregistré au dernier lancement. Le panneau **ChatHub**
se rafraîchit aussi tout seul après un run (une requête légère tant que l'onglet est
visible), donc une conversation ouverte par un node apparaît sans quitter l'onglet.

### Bouton « RAG upload (files) » du node Session

Sous `rag_top_k`, le bouton **RAG upload (files)** ouvre le sélecteur de fichiers du
navigateur : les documents choisis sont copiés dans
`ComfyUI/user/llm_chathub/documents/` (même dossier que le panneau), le champ
`rag_folder` est rempli avec ce dossier, `rag_enabled` passe à *on* et l'index est
reconstruit dans la foulée. Pendant l'indexation le bouton affiche la progression
(`RAG: 3/7`), puis une notification récapitule les documents et extraits indexés. Le
champ `rag_folder` reste utilisable tel quel pour indexer une arborescence existante.

Le bouton réindexe **tout le dossier** : un import ajoute les fichiers puis remplace
l'index (les documents déjà importés restent pris en compte). Les conversations qui
avaient `rag_enabled` actif bénéficient du nouvel index sans autre réglage.

Notes utiles :

- Pour envoyer `reply` dans une node existante : glisse la sortie `reply` sur le
  widget texte voulu (ex. `text` de *CLIP Text Encode*) — il se transforme en entrée —
  ou clic droit sur le widget → *Convert text to input*.
- Le champ `session_id` accepte un identifiant libre (`story_01`) ou `auto` : dans ce
  cas chaque node **Chat** garde sa propre conversation (clé = node id).
- `history_limit = 0` dans la Session signifie « tout l'historique disponible ».
- `image_mode = every image in the batch` envoie une requête par frame d'un batch
  d'images (l'ancienne valeur `chaque image du lot` reste acceptée).
- Le paramètre `persist` désactivé n'écrit rien en base (utile pour du batch).

## 5. API HTTP

Toutes les routes sont sous `/llm_chathub`. Les réponses sont du JSON avec un champ
`ok`. Les routes `chat`, `regenerate` et `rag/index` renvoient un flux
`text/event-stream` avec les évènements `delta` (texte de la réponse), `reasoning`
(chaine de pensee d'un modele de raisonnement), `progress` (indexation), `result`
(résumé final) et `error`.

| Méthode | Route | Corps / paramètres |
|---|---|---|
| GET | `/state` | — (réglages publics, conversations, index) |
| POST | `/settings` | patch des réglages (`active_provider`, `providers`, `chat`, `rag`) |
| POST | `/models` | `{provider, base_url, api_key}` |
| GET | `/conversations` | — |
| GET | `/conversation` | `?id=<id>` |
| POST | `/conversation/new` | `{id?, provider, model, system_prompt, reference, rag_*...}` (`id` renvoie 409 s'il existe déjà) |
| POST | `/conversation/update` | `{id, ...champs}` |
| POST | `/conversation/delete` | `{id}` |
| POST | `/chat` | `{conversation_id, text, images:[dataUrl], overrides:{}}` |
| POST | `/regenerate` | `{conversation_id, message_id}` |
| POST | `/cancel` | `{conversation_id}` |
| POST | `/messages/edit` | `{conversation_id, message_id, content}` |
| POST | `/messages/delete` | `{conversation_id, message_id, from_here}` |
| POST | `/messages/truncate` | `{conversation_id, keep}` |
| POST | `/upload` | `{name, data}` → image stockée |
| GET | `/image/{name}` | — |
| POST | `/rag/index` | `{folder?, provider, base_url, embed_model, chunk_size, overlap}` (`folder` absent = documents importés) |
| GET | `/rag/files` | — (documents importés) |
| POST | `/rag/files` | `{files:[{name, data(base64)}]}` → copie dans `documents/` |
| POST | `/rag/files/delete` | `{name}` |
| POST | `/rag/clear` | — |
| POST | `/rag/test` | `{query, top_k}` |
| GET | `/health` | — |

### curl

```bash
# une question, réponse complète (le flux SSE est renvoyé tel quel)
curl -N -X POST http://127.0.0.1:8188/llm_chathub/chat \
  -H 'Content-Type: application/json' \
  -d '{"conversation_id":"conv_demo","text":"Explique le CFG en 3 lignes"}'

# indexer un dossier
curl -N -X POST http://127.0.0.1:8188/llm_chathub/rag/index \
  -H 'Content-Type: application/json' \
  -d '{"folder":"D:/docs","embed_model":"nomic-embed-text"}'

# rechercher dans l'index
curl -X POST http://127.0.0.1:8188/llm_chathub/rag/test \
  -H 'Content-Type: application/json' -d '{"query":"contraintes de style","top_k":3}'
```

### JavaScript (navigateur ou Node 18+)

```js
const base = "http://127.0.0.1:8188";

const res = await fetch(`${base}/llm_chathub/chat`, {
  method: "POST",
  headers: { "Content-Type": "application/json" },
  body: JSON.stringify({ conversation_id: "conv_demo", text: "Bonjour !" }),
});

const reader = res.body.getReader();
const decoder = new TextDecoder();
let buffer = "";
while (true) {
  const { done, value } = await reader.read();
  if (done) break;
  buffer += decoder.decode(value, { stream: true });
  const parts = buffer.split("\n\n");
  buffer = parts.pop();
  for (const part of parts) {
    const [head, ...rest] = part.split("\n");
    const data = rest.map((l) => l.replace(/^data:/, "")).join("");
    if (!data) continue;
    if (head.startsWith("event: delta")) process.stdout.write(JSON.parse(data));
  }
}
```

### Python

```python
import json
import requests

BASE = "http://127.0.0.1:8188/llm_chathub"

def chat(question, conversation_id="conv_python"):
    with requests.post(
        f"{BASE}/chat",
        json={"conversation_id": conversation_id, "text": question},
        stream=True,
        timeout=600,
    ) as response:
        response.raise_for_status()
        for line in response.iter_lines(decode_unicode=True):
            if not line or not line.startswith("data:"):
                continue
            payload = json.loads(line[5:].strip())
            if isinstance(payload, str):
                print(payload, end="", flush=True)

chat("Donne-moi 5 idees de prompts cinematographiques")
```

## 6. Dépannage

- **« Impossible de joindre lmstudio ... »** : lance LM Studio → *Developer* → *Start
  Server*, ou `ollama serve`.
- **HTTP 401/403 sur ollama_cloud** : clé API absente ou invalide.
- **HTTP 404** : le nom du modèle n'existe pas sur le serveur — utilise
  *Charger les modeles*.
- **Un modèle manque dans la liste Ollama (local)** : c'est un modèle cloud relayé par
  ton daemon (`…-cloud`) — il se choisit dans **Ollama Cloud**, là où il appartient.
- **La liste de modèles est vide** : le provider répond mais n'annonce aucun modèle —
  démarre le serveur LM Studio, fais un `ollama pull`, ou vérifie la clé Ollama Cloud,
  puis *Reload models*.
- **Un document n'est pas pris en compte** : seuls les formats texte sont lus (pas de
  PDF/DOCX), et les fichiers trop courts (moins de 20 caractères utiles) sont ignorés.
- **Le RAG répond mal / n'utilise pas d'embeddings** : indique un modèle d'embedding
  (`nomic-embed-text` pour Ollama, `text-embedding-*` pour LM Studio). Sans modèle, le
  repli mots-clés (BM25) prend le relais automatiquement.
- **Images non prises en compte** : le modèle sélectionné doit être un modèle de vision.
- **Réponse vide alors que le modèle a « réfléchi »** : augmente `Max tokens`, le
  raisonnement a mangé le budget.
- **Accents bizarres (`cinÃ©ma`)** : une conversation enregistrée avant le correctif
  d'encodage peut contenir du texte corrompu et le modèle le recopie ; repartez sur une
  nouvelle conversation (**+ New**).
- **Les réponses sont longues à arriver** : le panneau affiche les tokens au fil de
  l'eau ; le node Chat, lui, attend la fin de la génération.
- Logs : `ComfyUI/user/comfyui.log`, plus l'onglet console du navigateur
  (`[LLM ChatHub]`).

## 7. Licence

MIT.
