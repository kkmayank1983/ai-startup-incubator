/**
 * AI NEXUS - INTEGRATED UI PAINTER
 * Manages Progress, Resizing, Animations, and Navigation.
 */

document.addEventListener("DOMContentLoaded", () => {
  console.log("AI Nexus Platform Initialized...");

  // 1. DATA CONTEXT
  // Get the current session ID from the <body> tag (e.g., <body data-current-session="t1_s1">)
  const currentSessionId = document.body.getAttribute("data-current-session");

  // 2. PAGE GUARD (The Gatekeeper)
  // If we are on a session page, check if the student has permission to be here.
  // 2. PAGE GUARD
  if (currentSessionId && typeof NexusCore !== "undefined") {
    if (!NexusCore.canAccess(currentSessionId)) {
      // Remove the "Security Clearance" alert about previous missions
      // Just redirect to the local dashboard
      window.location.href = "term2.html"; 
      return;
    }
  }

  // 3. UI PAINTING (Derived State)
  // Automatically colors cards and sidebar based on NexusCore progress.

  const updateUI = () => {
    if (typeof NexusCore === "undefined") return;

    // ==========================================
    // 1. EXISTING SESSION CARD LOGIC
    // ==========================================
    document.querySelectorAll("[data-session-id]").forEach((el) => {
      const id = el.getAttribute("data-session-id");
      const status = NexusCore.getStatus(id);
      const isSidebarItem = el.tagName === "LI";
      const link = el.tagName === "A" ? el : el.querySelector("a");

      el.classList.remove(
        "locked-level",
        "active-level",
        "completed-level",
        "sidebar-locked",
        "active",
      );
      if (link) link.style.pointerEvents = "auto";

      const oldLock = el.querySelector(".dynamic-lock");
      if (oldLock) oldLock.remove();

      if (status === "LOCKED") {
        el.classList.add(isSidebarItem ? "sidebar-locked" : "locked-level");
        if (link) {
          link.style.pointerEvents = "none";
          link.setAttribute("tabindex", "-1");
        }
        if (isSidebarItem) {
          const lockIcon = document.createElement("i");
          lockIcon.className = "fas fa-lock lock-mini dynamic-lock";
          el.appendChild(lockIcon);
        } else {
          const btn = el.querySelector(".level-btn");
          if (btn) btn.innerHTML = 'Locked <i class="fas fa-lock"></i>';
        }
      } else {
        if (!isSidebarItem && status === "UNLOCKED")
          el.classList.add("active-level");

        // Use the currentSessionId variable defined at the top of your script.js
        if (
          typeof currentSessionId !== "undefined" &&
          id === currentSessionId
        ) {
          el.classList.add(isSidebarItem ? "active" : "active-level");
        } else if (status === "COMPLETED") {
          el.classList.add(
            isSidebarItem ? "completed-sidebar" : "completed-level",
          );
        }

        if (!isSidebarItem) {
          const btn = el.querySelector(".level-btn");
          if (btn) {
            btn.innerHTML =
              status === "COMPLETED"
                ? 'Review Mission <i class="fas fa-check"></i>'
                : 'Start Mission <i class="fas fa-play"></i>';
          }
        }
      }
    });

    // ==========================================
    // 2. TERM CARD LOGIC (for terms.html)
    // ==========================================
    for (let i = 1; i <= 4; i++) {
      const termCard = document.querySelector(`.t${i}-card`);
      if (termCard) {
        const isUnlocked = NexusCore.isTermUnlocked(i);
        const statusLabel = termCard.querySelector(".card-status");

        if (!isUnlocked) {
          termCard.classList.add("locked-card");
          termCard.classList.remove("active-mission");
          termCard.style.pointerEvents = "none";
          if (statusLabel) statusLabel.innerText = "LOCKED";
        } else {
          termCard.classList.remove("locked-card");
          termCard.classList.add("active-mission");
          termCard.style.pointerEvents = "auto";
          if (statusLabel && statusLabel.innerText === "LOCKED") {
            statusLabel.innerText = "ACTIVE MISSION";
            statusLabel.classList.add("status-active");
          }
        }
      }
    }

    // ==========================================
    // 3. NEW FUTURE-PROOF SIDEBAR LOGIC
    // ==========================================
    let currentTerm = 2;
    if (typeof currentSessionId !== "undefined" && currentSessionId) {
      currentTerm = parseInt(currentSessionId.split("_")[0].replace("t", ""));
    }

    for (let i = 1; i <= 4; i++) {
      const sidebarSection = document.querySelector(
        `[data-sidebar-term="${i}"]`,
      );

      if (sidebarSection) {
        const isUnlocked = NexusCore.isTermUnlocked(i);
        const header = sidebarSection.querySelector("h3");
        const icon = header ? header.querySelector("i") : null;
        const list = sidebarSection.querySelector("ul");

        if (isUnlocked) {
          sidebarSection.classList.remove("locked");

          if (icon && icon.classList.contains("fa-lock")) {
            icon.classList.remove("fa-lock");
            icon.classList.add(
              list ? "fa-chevron-down" : "fa-external-link-alt",
            );
            icon.style.color = "#10b981";
          }

          if (header) {
            header.style.cursor = "pointer";

            if (list) {
              if (i === currentTerm) {
                list.style.display = "block";
                if (icon && icon.classList.contains("fa-chevron-down"))
                  icon.style.transform = "rotate(180deg)";
              } else {
                list.style.display = "none";
              }

              header.onclick = () => {
                if (list.style.display === "none") {
                  list.style.display = "block";
                  if (icon) icon.style.transform = "rotate(180deg)";
                } else {
                  list.style.display = "none";
                  if (icon) icon.style.transform = "rotate(0deg)";
                }
              };
            } else {
              header.title = `Enter Term ${i} Dashboard`;
              header.onclick = () => {
                window.location.href = `../term-${i}/term${i}.html`;
              };
            }
          }
          // --- START REPLACING AT APPROX LINE 191 ---
        } else {
          // This term is not part of the current standalone website
          sidebarSection.classList.add("locked");
          if (header) {
            header.style.cursor = "default"; // Change from 'not-allowed'
            header.onclick = () => {
              console.log(`Term ${i} is managed on a separate platform.`);
              // Optional: You could redirect to the other website URL here instead of an alert
            };
          }
        }
        // --- END REPLACING ---
      }
    }
  };

  // Call it immediately!
  updateUI();
  const celebrationOverlay = document.getElementById("completion-celebration");
  const celebrationParticles = document.getElementById("celebration-particles");
  const celebrationColors = [
    "#2563eb",
    "#0ea5e9",
    "#22c55e",
    "#38bdf8",
    "#a5f3fc",
    "#e2e8f0",
    "#8b5cf6",
  ];

  const createParticle = (type) => {
    const particle = document.createElement("div");
    particle.className = `particle ${type}`;
    const size = type === "confetti" ? 10 : 5;
    particle.style.width = `${size}px`;
    particle.style.height = `${size}px`;
    const color =
      celebrationColors[Math.floor(Math.random() * celebrationColors.length)];
    particle.style.background = color;
    particle.style.left = `${Math.random() * 100}%`;
    particle.style.top = `${Math.random() * -20}%`;
    particle.style.transform = `rotate(${Math.random() * 360}deg)`;
    return particle;
  };

  const generateCelebrationParticles = () => {
    if (!celebrationParticles) return;
    celebrationParticles.innerHTML = "";

    for (let i = 0; i < 28; i += 1) {
      const type = i % 3 === 0 ? "sparkle" : "confetti";
      const particle = createParticle(type);
      particle.style.animationDuration = `${2 + Math.random() * 1.5}s`;
      particle.style.left = `${Math.random() * 100}%`;
      particle.style.animationDelay = `${Math.random() * 0.5}s`;
      celebrationParticles.appendChild(particle);
    }
  };

  const triggerCompletionCelebration = () => {
    if (!celebrationOverlay) {
      window.location.href = "term2.html";
      return;
    }
    celebrationOverlay.classList.add("active");
    generateCelebrationParticles();
  };

  window.closeCelebrationAndNavigate = () => {
    if (celebrationOverlay) {
      celebrationOverlay.classList.remove("active");
    }
    window.location.href = "term2.html";
  };

  // 4. COMPLETION LOGIC
  const nextBtn = document.getElementById("next-btn");
  const sessionCheck = document.getElementById("session-check");

  if (nextBtn) {
    // Visual feedback for the checkbox (Your FIX 6)
    if (sessionCheck) {
      sessionCheck.addEventListener("change", () => {
        if (sessionCheck.checked) {
          nextBtn.innerHTML =
            'Complete & Next Session <i class="fas fa-arrow-right"></i>';
          nextBtn.style.transform = "scale(1.05)";
        } else {
          nextBtn.innerHTML =
            'Mark Complete & Next <i class="fas fa-arrow-right"></i>';
          nextBtn.style.transform = "scale(1)";
        }
      });
    }

    // Logic to actually save progress
    nextBtn.addEventListener("click", (e) => {
      if (sessionCheck && !sessionCheck.checked) {
        showWarningModal(); // FIX: Replaced alert with professional modal
        return;
      }
      if (currentSessionId) {
        NexusCore.complete(currentSessionId);
        triggerCompletionCelebration();
      }
    });
  }

  // 5. SIDEBAR RESIZE LOGIC
  const resizer = document.getElementById("sidebar-resizer");
  const sidebar = document.querySelector(".sidebar");

  if (resizer && sidebar) {
    resizer.addEventListener("mousedown", (e) => {
      document.body.classList.add("resizing");
      document.addEventListener("mousemove", resize);
      document.addEventListener("mouseup", stopResize);
    });

    function resize(e) {
      let newWidth = e.clientX;
      if (newWidth > 200 && newWidth < 600) {
        sidebar.style.width = newWidth + "px";
      }
    }

    function stopResize() {
      document.body.classList.remove("resizing");
      document.removeEventListener("mousemove", resize);
      document.removeEventListener("mouseup", stopResize);
    }
  }

  // 6. ANIMATIONS & BREADCRUMBS
  // Content Fade-in
  const content = document.querySelector(".content");
  if (content) {
    content.style.opacity = 0;
    content.style.transition = "opacity 0.5s ease-in";
    setTimeout(() => (content.style.opacity = 1), 100);
  }

  // Nav Link Active Highlighting
  const currentPath = window.location.pathname.split("/").pop() || "index.html";
  document.querySelectorAll(".nav-links a").forEach((link) => {
    if (link.getAttribute("href") === currentPath) {
      link.classList.add("active");
    }
  });
});
/* --- MCQ LOGIC --- */
document.querySelectorAll(".mcq-option").forEach((option) => {
  option.addEventListener("click", function () {
    const parent = this.parentElement;
    const isCorrect = this.getAttribute("data-correct") === "true";
    const feedback = this.getAttribute("data-feedback");

    // Prevent multiple clicks in the same question
    if (parent.classList.contains("answered")) return;
    parent.classList.add("answered");

    // Visual feedback
    if (isCorrect) {
      this.classList.add("mcq-correct");
      showMcqModal("Correct!", feedback, "success");
    } else {
      this.classList.add("mcq-wrong");
      showMcqModal("Review Needed", feedback, "error");
    }

    // Disable all options in this question
    parent.querySelectorAll(".mcq-option").forEach((btn) => {
      btn.style.cursor = "default";
    });
  });
});

function showMcqModal(title, text, type) {
  const modal = document.getElementById("mcq-modal");
  const titleEl = document.getElementById("mcq-modal-title");
  const textEl = document.getElementById("mcq-modal-feedback");
  const iconEl = document.getElementById("mcq-modal-icon");

  titleEl.innerText = title;
  textEl.innerText = text;

  if (type === "success") {
    iconEl.innerHTML =
      '<i class="fas fa-check-circle" style="color: #22c55e;"></i>';
    titleEl.style.color = "#166534";
  } else {
    iconEl.innerHTML =
      '<i class="fas fa-exclamation-circle" style="color: #ef4444;"></i>';
    titleEl.style.color = "#991b1b";
  }

  modal.style.display = "flex";
  setTimeout(() => {
    modal.querySelector(".mcq-modal-content").style.transform = "scale(1)";
  }, 10);
}

function closeMcqModal() {
  const modal = document.getElementById("mcq-modal");
  modal.querySelector(".mcq-modal-content").style.transform = "scale(0.9)";
  setTimeout(() => {
    modal.style.display = "none";
  }, 200);
}
/* --- MCQ INTERACTION LOGIC --- */
document.querySelectorAll(".mcq-btn").forEach((btn) => {
  btn.addEventListener("click", function () {
    const isCorrect = this.getAttribute("data-correct") === "true";
    const feedback = this.getAttribute("data-feedback");

    // Remove previous status colors from this question only
    this.parentElement.querySelectorAll(".mcq-btn").forEach((b) => {
      b.classList.remove("correct", "wrong");
    });

    // Add new status color
    this.classList.add(isCorrect ? "correct" : "wrong");

    // Show Modal
    showMcqModal(isCorrect, feedback);
  });
});

function showMcqModal(isCorrect, text) {
  const modal = document.getElementById("mcq-modal");
  const statusBar = document.getElementById("mcq-modal-status-bar");
  const title = document.getElementById("mcq-modal-title");
  const icon = document.getElementById("mcq-modal-icon");
  const feedback = document.getElementById("mcq-modal-feedback");

  // Use Pastel Accent Colors
  const color = isCorrect ? "#22c55e" : "#ef4444";

  statusBar.style.backgroundColor = color;
  title.innerText = isCorrect ? "Diagnostic Passed!" : "System Alert";
  title.style.color = color;
  feedback.innerText = text;
  icon.innerHTML = isCorrect
    ? `<i class="fas fa-check-circle" style="color:${color}"></i>`
    : `<i class="fas fa-exclamation-triangle" style="color:${color}"></i>`;

  modal.style.display = "flex";
  setTimeout(() => {
    modal.querySelector(".mcq-modal-content").style.transform = "scale(1)";
  }, 10);
}

function closeMcqModal() {
  const modal = document.getElementById("mcq-modal");
  modal.querySelector(".mcq-modal-content").style.transform = "scale(0.9)";
  setTimeout(() => {
    modal.style.display = "none";
  }, 200);
}
/* --- WARNING MODAL HELPERS --- */
function showWarningModal() {
  const modal = document.getElementById("warning-modal");
  modal.style.display = "flex";
  setTimeout(() => {
    modal.querySelector(".mcq-modal-content").style.transform = "scale(1)";
  }, 10);
}

function closeWarningModal() {
  const modal = document.getElementById("warning-modal");
  modal.querySelector(".mcq-modal-content").style.transform = "scale(0.9)";
  setTimeout(() => {
    modal.style.display = "none";
  }, 200);
}
document.addEventListener("DOMContentLoaded", function () {
  const toggle = document.querySelector(".homework-toggle");
  const content = document.querySelector(".homework-content");
  const arrow = document.querySelector(".toggle-arrow");

  if (toggle) {
    toggle.addEventListener("click", function () {
      content.classList.toggle("open");
      arrow.classList.toggle("rotate");
    });
  }
});
// ... (keep all your existing code at the top of script.js)

// ... (keep the rest of your script.js logic unchanged)
