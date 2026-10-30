# About

TOPS (**T**iny **O**pen **P**ower System **S**imulator) is developed at
SINTEF Energy Research as a research and teaching tool. It was previously
called **DynPSSimPy**.

The aim is a simple, lightweight simulator that is easy to install, run and
modify. Performance is not the main priority. The core uses only numpy and
scipy.

!!! warning "Research software"

    TOPS is developed as part of ongoing research and contains experimental
    features. Check results against a reference where it matters.

## Citing

If you use TOPS in your research, please cite
[this paper](https://ieeexplore.ieee.org/document/9494770):

```bibtex
@inproceedings{haugdal2021tops,
  author    = {Haugdal, Hallvar and Uhlen, Kjetil},
  title     = {An Open Source Power System Simulator in Python for Efficient
               Prototyping of WAMPAC Applications},
  booktitle = {2021 IEEE Madrid PowerTech},
  year      = {2021},
  doi       = {10.1109/PowerTech46648.2021.9494770}
}
```

## Contributing

Bug reports and pull requests are welcome on
[GitHub](https://github.com/hallvar-h/TOPS). Pull request titles follow
[Conventional Commits](https://www.conventionalcommits.org/) (`fix: ...`,
`feat: ...`, `docs: ...`), and releases are made with Release Please. The
[README](https://github.com/hallvar-h/TOPS#contributing-and-releases) has
the details.

To work on the documentation:

```bash
uv sync --group docs
uv run mkdocs serve
```

The pages are in `docs/`. Code blocks marked `exec="on"` run during the
build, so `uv run mkdocs build --strict` also checks that the examples still
work.

## Licence

TOPS is released under the
[MIT licence](https://github.com/hallvar-h/TOPS/blob/main/LICENSE).

## Contact

[Hallvar Haugdal](mailto:hallvhau@gmail.com), or open an
[issue](https://github.com/hallvar-h/TOPS/issues).
