"""
AI Incubator - Setup Check  (v2)
Run this if a script fails on 'import torch'.
It does NOT import torch, so it still works when torch is broken.

    python check_setup.py

Send the WHOLE output to your teacher.
"""
import sys, platform, ctypes, os, glob

print("=" * 60)
print("AI INCUBATOR - SETUP CHECK v2")
print("=" * 60)

# ---------- 1. Which Python is actually running? ----------
print("\n[1] PYTHON")
print(f"   version    : {sys.version.split()[0]}")
print(f"   bits       : {platform.architecture()[0]}")
print(f"   exe        : {sys.executable}")
in_venv = sys.prefix != sys.base_prefix
print(f"   in a venv  : {in_venv}")
if not in_venv:
    print("   !! You are NOT inside the venv. Activate it first.")

# ---------- 2. Windows version ----------
print("\n[2] WINDOWS")
print(f"   {platform.platform()}")
if os.name == "nt":
    v = sys.getwindowsversion()
    print(f"   build      : {v.major}.{v.minor}.{v.build}")
    if v.major < 10:
        print("   !! PyTorch 2.x needs Windows 10 or newer.")

# ---------- 3. CPU features ----------
print("\n[3] CPU")
print(f"   {platform.processor()}")
avx = avx2 = None
if os.name == "nt":
    k32 = ctypes.windll.kernel32
    avx = bool(k32.IsProcessorFeaturePresent(39))   # AVX
    avx2 = bool(k32.IsProcessorFeaturePresent(40))  # AVX2
    print(f"   AVX        : {avx}")
    print(f"   AVX2       : {avx2}")

# ---------- 4. Microsoft C++ runtime ----------
print("\n[4] MICROSOFT C++ RUNTIME")
missing = []
if os.name == "nt":
    sysdir = os.path.join(os.environ.get("WINDIR", r"C:\Windows"), "System32")
    for d in ["vcruntime140.dll", "vcruntime140_1.dll", "msvcp140.dll", "concrt140.dll"]:
        ok = os.path.exists(os.path.join(sysdir, d))
        print(f"   {d:20} {'found' if ok else 'MISSING'}")
        if not ok:
            missing.append(d)

# ---------- 5. Where is torch, and does it belong to this Python? ----------
print("\n[5] TORCH INSTALL")
site = os.path.join(sys.prefix, "Lib", "site-packages", "torch")
print(f"   expected   : {site}")
print(f"   exists     : {os.path.isdir(site)}")
libdir = os.path.join(site, "lib")
dlls = sorted(glob.glob(os.path.join(libdir, "*.dll"))) if os.path.isdir(libdir) else []
print(f"   DLL count  : {len(dlls)}")

# ---------- 6. Try loading the DLLs one by one ----------
print("\n[6] LOADING TORCH DLLs (this is where it breaks)")
first_fail = None
if dlls:
    if hasattr(os, "add_dll_directory"):
        try:
            os.add_dll_directory(libdir)
        except Exception:
            pass
    for path in dlls:
        name = os.path.basename(path)
        try:
            ctypes.WinDLL(path)
            print(f"   OK      {name}")
        except OSError as e:
            code = getattr(e, "winerror", "?")
            print(f"   FAILED  {name}   (WinError {code})")
            if first_fail is None:
                first_fail = (name, code)
else:
    print("   no DLLs found - torch is not installed here")

# ---------- 7. Verdict ----------
print("\n" + "=" * 60)
print("WHAT TO DO")
print("=" * 60)

py = sys.version_info

if not os.path.isdir(site):
    print(">> torch is not installed in this environment. Run:")
    print("   pip install torch --index-url https://download.pytorch.org/whl/cpu")
elif not in_venv:
    print(">> You are not in the venv. Activate it, then run this again:")
    print("   .\\venv\\Scripts\\Activate.ps1")
elif missing:
    print(">> Install Microsoft Visual C++ Redistributable (x64), then REBOOT:")
    print("   https://aka.ms/vs/17/release/vc_redist.x64.exe")
elif platform.architecture()[0] != "64bit":
    print(">> 32-bit Python detected. PyTorch needs 64-bit Python.")
elif avx2 is False:
    print(">> THIS IS THE PROBLEM: your CPU has no AVX2 support.")
    print("   Modern PyTorch cannot run on this machine, no matter what you reinstall.")
    if avx:
        print("   Your CPU does have AVX, so an older PyTorch may work.")
        if py >= (3, 12):
            print(f"   BUT you have Python {py.major}.{py.minor}, which old PyTorch does not support.")
            print("   You would need Python 3.11 AND torch 1.13.1.")
        else:
            print("   Try:  pip install torch==1.13.1")
    print("   Otherwise: ask your teacher for the Google Colab link.")
elif first_fail:
    print(f">> {first_fail[0]} failed to load (WinError {first_fail[1]}).")
    print("   Everything else looks fine, so this is usually antivirus blocking it.")
    print("   Add this folder as an exclusion, then try again:")
    print(f"   {libdir}")
else:
    print(">> All torch DLLs loaded fine here.")
    print("   If 'import torch' still fails, send this whole output to your teacher.")
