/**
 * NEXUS CORE ENGINE
 * Logic for curriculum sequence and progress persistence.
 */
const NexusCore = (() => {
  const STORAGE_KEY = "nexus_student_progress";

  const storage = {
    save: (data) => localStorage.setItem(STORAGE_KEY, JSON.stringify(data)),
    load: () =>
      JSON.parse(localStorage.getItem(STORAGE_KEY)) || { completed: [] },
  };

  const curriculum = [
    "t1_s1",
    "t1_s2",
    "t1_s3",
    "t1_s4",
    "t1_s5",
    "t1_s6",
    "t1_s7",
    "t1_s8",
  ];

  let progress = storage.load();

  return {
    getStatus: (id) => {
      if (progress.completed.includes(id)) return "COMPLETED";
      const index = curriculum.indexOf(id);
      if (index === 0) return "UNLOCKED"; // S1 always available
      const prev = curriculum[index - 1];
      return progress.completed.includes(prev) ? "UNLOCKED" : "LOCKED";
    },

    complete: (id) => {
      if (!progress.completed.includes(id)) {
        progress.completed.push(id);
        storage.save(progress);
        window.dispatchEvent(new CustomEvent("nexus_update"));
      }
    },

    canAccess: (id) => {
      const status = NexusCore.getStatus(id);
      return status === "COMPLETED" || status === "UNLOCKED";
    },
  };
})();
