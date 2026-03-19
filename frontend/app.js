// ============================================================
// Ekie Legal Router — Chat Frontend
// ============================================================
function resolveApiUrl() {
  var host = window.location.hostname;
  if (host === "localhost" || host === "127.0.0.1") {
    return "http://localhost:8080";
  }
  return "https://ekie-legal-router-570069708764.europe-west1.run.app";
}

const API_URL = resolveApiUrl();

let conversationId = null;

const chat = document.getElementById("chat");
const form = document.getElementById("chatForm");
const userInput = document.getElementById("userInput");
const sendBtn = document.getElementById("sendBtn");
const domaineIndicator = document.getElementById("domaineIndicator");

// Boot message
addBotMessage("Bonjour, je suis l'assistant juridique Ekie. Decrivez votre situation, je vous poserai quelques questions avant de generer un brief pour votre avocat.");

form.addEventListener("submit", handleSubmit);

async function handleSubmit(e) {
  e.preventDefault();
  const msg = userInput.value.trim();
  if (!msg) return;

  addUserMessage(msg);
  userInput.value = "";
  setLoading(true);

  try {
    const response = await fetch(API_URL + "/chat", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        conversation_id: conversationId,
        message: msg,
      }),
    });

    if (!response.ok) {
      var errData = await response.json().catch(function() { return null; });
      throw new Error(errData && errData.error ? errData.error : "Erreur serveur (" + response.status + ")");
    }

    var data = await response.json();
    conversationId = data.conversation_id;

    // Update domaine indicator
    if (data.domaine) {
      domaineIndicator.innerHTML =
        "Domaine : <span>" + data.domaine + "</span>" +
        " &middot; Confiance : <span>" + Math.round((data.confiance || 0) * 100) + "%</span>";
    }

    if (data.action === "question") {
      addBotMessage(data.message);
    } else if (data.action === "brief") {
      addBriefMessage(data.brief);
    }
  } catch (err) {
    addBotMessage("Erreur : " + (err.message || "Une erreur inattendue est survenue."));
  } finally {
    setLoading(false);
  }
}

function addUserMessage(text) {
  var div = document.createElement("div");
  div.className = "message message--user";
  div.innerHTML =
    '<div class="avatar avatar--user">V</div>' +
    '<div class="bubble"></div>';
  div.querySelector(".bubble").textContent = text;
  chat.appendChild(div);
  scrollToBottom();
}

function addBotMessage(text) {
  var div = document.createElement("div");
  div.className = "message message--bot";
  div.innerHTML =
    '<div class="avatar avatar--bot">E</div>' +
    '<div class="bubble"></div>';
  div.querySelector(".bubble").textContent = text;
  chat.appendChild(div);
  scrollToBottom();
}

function addBriefMessage(brief) {
  var wrapper = document.createElement("div");
  wrapper.className = "message message--bot";
  wrapper.style.maxWidth = "100%";

  var html = '<div class="avatar avatar--bot">E</div>';
  html += '<div class="brief-container">';
  html += '<div class="bubble" style="margin-bottom:0.75rem; font-weight:600;">Voici le brief pour votre avocat :</div>';
  html += '<div class="brief-grid">';

  // Urgence
  html += '<div class="card card--urgence urgence-' + brief.urgence + '">';
  html += '<div class="card-header">Urgence</div>';
  html += '<div class="card-body"><span class="urgence-badge">' + brief.urgence + '</span></div></div>';

  // Domaine
  html += '<div class="card"><div class="card-header">Domaine</div><div class="card-body">';
  html += '<span class="domaine-tag">' + esc(brief.domaine) + '</span>';
  html += '<div class="sous-domaine">' + esc(brief.sous_domaine) + '</div></div></div>';

  // Resume
  html += '<div class="card"><div class="card-header">Resume de la situation</div>';
  html += '<div class="card-body">' + esc(brief.resume_situation) + '</div></div>';

  // Points cles
  html += '<div class="card"><div class="card-header">Points cles</div><div class="card-body"><ul>';
  brief.points_cles.forEach(function(p) { html += '<li>' + esc(p) + '</li>'; });
  html += '</ul></div></div>';

  // References
  html += '<div class="card"><div class="card-header">References legales</div><div class="card-body">';
  brief.references_legales.forEach(function(ref) {
    html += '<div class="ref-item">';
    html += '<div class="ref-source">' + esc(ref.source) + '</div>';
    html += '<div class="ref-extrait">' + esc(ref.extrait) + '</div>';
    html += '<div class="ref-score">Pertinence : ' + Math.round(ref.pertinence * 100) + '%</div>';
    html += '</div>';
  });
  html += '</div></div>';

  // Questions a creuser
  html += '<div class="card"><div class="card-header">Questions a creuser</div><div class="card-body"><ol>';
  brief.questions_a_creuser.forEach(function(q) { html += '<li>' + esc(q) + '</li>'; });
  html += '</ol></div></div>';

  // Delais
  if (brief.delais_importants) {
    html += '<div class="card"><div class="card-header">Delais importants</div>';
    html += '<div class="card-body">' + esc(brief.delais_importants) + '</div></div>';
  }

  // Recommandation
  html += '<div class="card card--recommandation"><div class="card-header">Recommandation immediate</div>';
  html += '<div class="card-body">' + esc(brief.recommandation_immediate) + '</div></div>';

  html += '</div></div>';

  wrapper.innerHTML = html;
  chat.appendChild(wrapper);
  scrollToBottom();

  // Disable further input after brief
  userInput.placeholder = "Brief genere. Rechargez pour une nouvelle conversation.";
  userInput.disabled = true;
  sendBtn.disabled = true;
}

function setLoading(loading) {
  sendBtn.disabled = loading;
  if (loading) {
    var div = document.createElement("div");
    div.className = "message message--bot";
    div.id = "typing";
    div.innerHTML =
      '<div class="avatar avatar--bot">E</div>' +
      '<div class="bubble"><div class="typing"><span></span><span></span><span></span></div></div>';
    chat.appendChild(div);
    scrollToBottom();
  } else {
    var typing = document.getElementById("typing");
    if (typing) typing.remove();
  }
}

function scrollToBottom() {
  chat.scrollTop = chat.scrollHeight;
}

function esc(str) {
  var div = document.createElement("div");
  div.textContent = str;
  return div.innerHTML;
}
