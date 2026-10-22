import json
from pathlib import Path
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

r = Path("results/warp")
published = json.loads((r / "planar_interface_cpu/result.json").read_text())
area = json.loads((r / "planar_area_long_cpu/result.json").read_text())
fig, axes = plt.subplots(1, 2, figsize=(10, 4), constrained_layout=True)
for data, ratio, label in [
    (published, 1, "Uniform"),
    (published, 3, "Published-style trace"),
    (area, 3, "Experimental area trace"),
]:
    rows = [w for w in data["wave_convergence"] if w["ratio"] == ratio]
    x = [1 / w["coarse_n"] for w in rows]
    axes[0].loglog(x, [w["electric_relative_l2"] for w in rows], "o-", label=label)
    axes[1].loglog(
        x, [w["initial_projection_relative_l2"] for w in rows], "o-", label=label
    )
for ax in axes:
    ax.set_xlabel("Coarse cell width (normalized units)")
    ax.grid(True, which="both", alpha=0.25)
axes[0].set_ylabel("Electric relative L2 error")
axes[0].legend()
axes[1].set_ylabel("Initial trace projection relative L2")
axes[0].set_title("After propagation")
axes[1].set_title("Representation error before propagation")
fig.suptitle("3D planar-interface CPU control — composite Gauss not yet certified")
fig.savefig(r / "planar_area_long_cpu/convergence.png", dpi=160)
