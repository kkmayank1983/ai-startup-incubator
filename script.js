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
  if (currentSessionId && typeof NexusCore !== "undefined") {
    if (!NexusCore.canAccess(currentSessionId)) {
      alert(
        "Security Clearance Required: Please complete previous missions first.",
      );
      window.location.href = "term1.html";
      return;
    }
  }

  // 3. UI PAINTING (Derived State)
  // Automatically colors cards and sidebar based on NexusCore progress.

  const updateUI = () => {
    if (typeof NexusCore === "undefined") return;

    document.querySelectorAll("[data-session-id]").forEach((el) => {
      const id = el.getAttribute("data-session-id");
      const status = NexusCore.getStatus(id);
      const isSidebarItem = el.tagName === "LI";
      const link = el.tagName === "A" ? el : el.querySelector("a");

      // 1. Reset all state classes
      el.classList.remove(
        "locked-level",
        "active-level",
        "completed-level",
        "sidebar-locked",
        "active",
      );
      if (link) link.style.pointerEvents = "auto";

      // 2. Cleanup old icons
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
          // Append to 'el' (the <li>) so it's relative to the whole row
          el.appendChild(lockIcon);
        } else {
          const btn = el.querySelector(".level-btn");
          if (btn) btn.innerHTML = 'Locked <i class="fas fa-lock"></i>';
        }
      } else {
        // --- STATUS IS UNLOCKED OR COMPLETED ---

        // 1. ANIMATION LOGIC:
        // Only add 'active-level' if the session is NOT completed.
        // This ensures Green cards don't "pop" or "glow" on hover.
        if (!isSidebarItem && status === "UNLOCKED") {
          el.classList.add("active-level");
        }

        // 2. BLUE COLOR LOGIC:
        // Only turn BLUE if this is the specific page the student is currently viewing.
        if (id === currentSessionId) {
          el.classList.add(isSidebarItem ? "active" : "active-level");
        }

        // 3. GREEN COLOR LOGIC:
        // If finished but NOT the current page, show the static GREEN state.
        else if (status === "COMPLETED") {
          el.classList.add(
            isSidebarItem ? "completed-sidebar" : "completed-level",
          );
        }

        // 4. Update Roadmap Button Text
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
  };
  updateUI();
  // --- CELEBRATION LOGIC ---
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
      window.location.href = "term1.html";
      return;
    }
    celebrationOverlay.classList.add("active");
    generateCelebrationParticles();
  };

  window.closeCelebrationAndNavigate = () => {
    if (celebrationOverlay) celebrationOverlay.classList.remove("active");
    window.location.href = "term1.html";
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
    // Logic to actually save progress
    nextBtn.addEventListener("click", (e) => {
      if (sessionCheck && !sessionCheck.checked) {
        showWarningModal(); 
        return;
      }
      if (currentSessionId) {
        NexusCore.complete(currentSessionId);
        // CHANGE HERE: Call celebration instead of immediate redirect
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
