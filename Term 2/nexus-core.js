const NexusCore = (() => {
  // 1. UNIQUE STORAGE: Change 'term2' to 'term3', etc., for each new site
  const STORAGE_KEY = "nexus_term2_standalone_progress";

  const storage = {
    save: (data) => localStorage.setItem(STORAGE_KEY, JSON.stringify(data)),
    load: () =>
      JSON.parse(localStorage.getItem(STORAGE_KEY)) || { completed: [] },
  };

  // 2. LOCAL CURRICULUM: Only include sessions present on THIS website
  const curriculum = [
    "t2_s1",
    "t2_s2",
    "t2_s3",
    "t2_s4",
    "t2_s5",
    "t2_s6",
    "t2_s7",
    "t2_s8",
  ];

  let progress = storage.load();

  return {
    getStatus: (id) => {
      if (progress.completed.includes(id)) return "COMPLETED";
      const index = curriculum.indexOf(id);
      // If it's the first session of THIS site, it's always unlocked
      if (index === 0) return "UNLOCKED";
      const prev = curriculum[index - 1];
      return progress.completed.includes(prev) ? "UNLOCKED" : "LOCKED";
    },

    // 3. STANDALONE UNLOCK: Always return true for the current term
    isTermUnlocked: (termNumber) => {
      // If this is the Term 2 website, Term 2 is always "unlocked"
      if (termNumber === 2) return true;
      return false; // Other terms don't exist here
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
