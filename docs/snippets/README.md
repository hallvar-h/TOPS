Files in this folder are included into pages with the snippets extension
(`--8<-- "name"`), not published as pages of their own.

`plot.py` goes as the first line of an executed block that draws with
matplotlib:

    ```python exec="on" source="above" html="1"
    --8<-- "plot.py"
    import matplotlib.pyplot as plt
    plt.plot([0, 1], [0, 1])
    plt.show()
    ```

It switches matplotlib to the Agg backend and replaces `plt.show()` so that
every open figure is printed as inline SVG. Each of its lines carries the
`# markdown-exec: hide` marker, so readers see only the code after it, which
is the same code they would run themselves.
