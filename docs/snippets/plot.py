import io  # markdown-exec: hide
import matplotlib  # markdown-exec: hide
matplotlib.use("Agg")  # markdown-exec: hide
matplotlib.rcParams["svg.hashsalt"] = "tops-docs"  # markdown-exec: hide
import matplotlib.pyplot as plt  # markdown-exec: hide
def _tops_docs_show(*args, **kwargs):  # markdown-exec: hide
    for num in plt.get_fignums():  # markdown-exec: hide
        buf = io.StringIO()  # markdown-exec: hide
        plt.figure(num).savefig(buf, format="svg", bbox_inches="tight", metadata={"Date": None})  # markdown-exec: hide
        svg = buf.getvalue()  # markdown-exec: hide
        print('<div class="tops-figure">' + svg[svg.find("<svg"):] + "</div>")  # markdown-exec: hide
    plt.close("all")  # markdown-exec: hide
plt.show = _tops_docs_show  # markdown-exec: hide
